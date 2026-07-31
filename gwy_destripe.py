"""
Stripe removal: MDSR (Fourier filtering) and GSR (variational).

`mdsr` implements the destriping method of

    X. Liang, Y. Zang, D. Dong, L. Zhang, M. Fang, X. Yang, A. Arranz,
    J. Ripoll, H. Hui, J. Tian, "Stripe artifact elimination based on
    nonsubsampled contourlet transform for light sheet fluorescence
    microscopy", J. Biomed. Opt. 21(10), 106005 (2016),

following the reference implementation of the General-Stripe-Removal
project (N. Rottmayer, `Matlab-Stripe-Removal/Algorithms/MDSR.m`).

The method has three steps:

  1. decompose the image with a nonsubsampled contourlet transform (NSCT)
     into shift-invariant subbands of different scale and direction,
  2. in every high-pass subband, damp the frequencies that carry stripes
     running in the given direction - with a damping width that shrinks as
     the subband's own orientation moves away from the stripe direction,
     so subbands that cannot hold the stripes are barely touched,
  3. reconstruct the image from the damped subbands.

Both stages of the NSCT (the nonsubsampled pyramid and the nonsubsampled
directional filter bank) are *nonsubsampled*: they are plain linear
shift-invariant filters, with no decimation and hence no aliasing. The
damping is a multiplication in the frequency domain as well. The whole
method is therefore one linear filter, and is implemented as one: the
analysis filter of each subband, its damping and its synthesis filter are
accumulated into a single frequency mask, applied in one FFT round trip.
That is exact - not an approximation of the subband loop - and it makes the
filter fast enough for a live preview (a 512x512 image takes milliseconds
instead of the ~4 s the paper reports for the explicit subband loop).

What differs from the reference implementation: the filter bank is built
directly in the frequency domain, as raised-cosine rings and angular wedges
that sum to exactly one, instead of the 'maxflat' / 'dmaxflat7' filter banks
of Cunha's NSCT toolbox. The structure of the method, the damping equations
and the parameters are those of the paper; the transfer functions of the
filter bank are not bit-identical to the toolbox's.

One correction to the paper: Eq. (1) defines the damping coordinate as
`u cos(pi/2 + theta0) + v sin(pi/2 + theta0)`, which puts the zero of the
groove on the line *perpendicular* to the stripe frequencies. The text
around it ("the bottom is the line with angle pi/2 + theta0 ... where the
value of w is 0"), and the reference implementation, both put the groove on
the stripe frequencies themselves. That is what is implemented here: the
damped coordinate is the frequency component *along* the stripes,
`u cos(theta0) + v sin(theta0)`.

`gsr` implements the general stripe remover of

    N. Rottmayer, C. Redenbach, F. O. Fahrbach, "A universal and effective
    variational method for destriping: application to light-sheet
    microscopy, FIB-SEM, and remote sensing images", Opt. Express 33(3),
    5800 (2025),

ported from the same project's `Python-Stripe-Removal/
GeneralStripeRemover.py` (PyTorch) to numpy, for the 2D case. It splits the
image into a clean part u and a stripe part s with u + s = u0 by minimizing

    mu1*||grad u||_{2,1} + i_[0,1](u) + ||grad_theta s||_1 + mu2*||s||_1

with the primal-dual hybrid gradient method (extrapolated dual, PDHGMp).
The first term says a clean image has few strong edges, the box indicator
keeps u in the value range of the input, the third says stripes vary little
*along* their own direction, and the last says only a small part of the
image is struck by stripes. mu1 sets the strength of the removal, mu2 the
caution about touching real structure.

Since the box indicator constrains u to [0, 1], the image is normalized to
that range before the iteration and mapped back afterwards - AFM heights
are in nanometres and would otherwise be clipped to nothing. The
recommended parameters therefore transfer directly.
"""

import numpy as np

# Reference defaults (General-Stripe-Removal/ProcessingScript.m): 8
# directions, 5 levels, sigma in 5..25, sigma_a = 0.3, max_angle = 45 deg.
# The paper instead uses sigma_a = 0.8 rad and filters every direction.
#
# sigma is counted in frequency BINS, so its effect depends on the image
# width: the 5..25 range was tuned on light-sheet images of ~1000 px and is
# aggressive on a 512 px AFM scan, where the stripes sit within a couple of
# bins of fx = 0 anyway. Hence the lower default here - raise it while
# watching the preview if stripes survive.
DEFAULTS = dict(angle=0.0, directions=8, levels=5, sigma=5.0,
                sigma_a=0.3, max_angle=45.0)

# GSR defaults from the paper's Supplement section 2: mu1 = 1/3 and
# mu2 = 1/300 are "a good starting point", and the intervals
# mu1 in [0.1, 0.5], mu2 in [0.0016, 0.017] "were never exceeded". The
# supplement recommends 10000 iterations for a fully converged result and
# notes 5000 often suffice; the default here is lower so that the preview
# stays interactive - raise it before applying if the result still moves.
GSR_DEFAULTS = dict(angle=0.0, mu1=1.0 / 3.0, mu2=1.0 / 300.0,
                    iterations=600)


def _freq_grids(shape):
    """Frequency grids in cycles per pixel, in fftshift-ed layout."""
    ny, nx = shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    return np.meshgrid(fx, fy)


def _wrap_half_pi(angles):
    """Wrap angles into [-pi/2, pi/2) - orientations are modulo pi."""
    return (angles + np.pi / 2) % np.pi - np.pi / 2


def directional_wedges(shape, directions=8):
    """
    The angular filters of the nonsubsampled directional filter bank.

    `directions` raised-cosine wedges centered on the frequency angles
    pi*l/directions, each one octave-of-angle wide. Neighbouring wedges
    overlap as cos^2 / sin^2, so they sum to exactly one everywhere and the
    decomposition is inverted by simply summing the subbands.

    Returns [(center angle, mask), ...] with the masks in fftshift-ed
    layout. A wedge is centered on the fy axis (frequency angle pi/2)
    whenever `directions` is even, which is the wedge that carries
    horizontal stripes.
    """
    FX, FY = _freq_grids(shape)
    phi = np.arctan2(FY, FX)
    step = np.pi / directions
    wedges = []
    for level in range(directions):
        center = np.pi * level / directions
        w = _wrap_half_pi(phi - center)
        mask = np.where(np.abs(w) < step,
                        np.cos(np.pi * w / (2.0 * step)) ** 2, 0.0)
        wedges.append((center, mask))
    return wedges


def pyramid_rings(shape, levels=5):
    """
    The bandpass rings of the nonsubsampled pyramid.

    Octave-wide raised-cosine rings: ring i covers the radial frequencies
    around 0.5/2**i cycles/px. Returns (rings, lowpass) with
    `sum(rings) + lowpass == 1` everywhere, so this too is inverted by
    summing.
    """
    FX, FY = _freq_grids(shape)
    radius = np.hypot(FX, FY)
    rings = []
    prev = np.ones_like(radius)
    for level in range(1, levels + 1):
        cut = 0.5 / 2 ** level
        low = np.ones_like(radius)
        band = (radius > cut / 2) & (radius < cut)
        low[band] = np.cos(np.pi / 2 * np.log2(2 * radius[band] / cut)) ** 2
        low[radius >= cut] = 0.0
        rings.append(prev - low)
        prev = low
    return rings, prev


def max_levels(shape):
    """Deepest pyramid that still resolves its lowest ring on this image."""
    return max(1, int(np.floor(np.log2(min(shape)))) - 2)


def mdsr_mask(shape, angle=0.0, directions=8, levels=5, sigma=5.0,
              sigma_a=0.3, max_angle=45.0, n_ref=None):
    """
    The composite frequency mask of the MDSR filter, in fftshift-ed layout.

    Args:
        shape: (ny, nx) of the image the mask is applied to.
        angle: direction of the stripes in degrees, measured from the x
            axis: 0 = horizontal stripes (the usual AFM scan-line
            artifact), 90 = vertical stripes.
        directions: number of directional subbands per scale (a power of
            two, as the directional filter bank is a binary tree).
        levels: number of pyramid scales. Stripes coarser than the last
            scale stay in the low-pass residual, which is never filtered -
            so raise this until the low-pass holds no visible stripes.
        sigma: width of the damping groove, in frequency bins of the image
            axis along the stripes (same units as the reference
            implementation, which recommends 5..25 - see DEFAULTS). It
            trades stripe removal against real structure that is itself
            elongated along the scan lines: the groove reaches roughly
            2.5*sigma bins, i.e. it takes out near-horizontal features
            longer than about nx/(2.5*sigma) pixels.
        sigma_a: how fast the damping narrows as a subband's orientation
            moves away from the stripes, in radians.
        max_angle: subbands whose orientation differs from the stripes by
            more than this (degrees) are left untouched.
        n_ref: number of pixels along the stripe axis to use for the bin
            scale of `sigma`. Defaults to the size of `shape`; pass the
            unpadded size when the mask is built for a padded image, so
            that `sigma` keeps its meaning.

    Returns:
        A float array in [0, 1]. The DC bin is exactly 1, so the filter
        can never shift the mean height.
    """
    theta = np.deg2rad(angle)
    FX, FY = _freq_grids(shape)

    # Frequency component ALONG the stripes: this is the coordinate that
    # crosses the ridge of stripe energy, so damping it removes the
    # stripes. In bins, like the reference implementation.
    if n_ref is None:
        n_ref = shape[1] if abs(np.cos(theta)) >= abs(np.sin(theta)) \
            else shape[0]
    t = (FX * np.cos(theta) + FY * np.sin(theta)) * n_ref

    rings, lowpass = pyramid_rings(shape, levels)
    wedges = directional_wedges(shape, directions)
    ring_sum = np.sum(rings, axis=0)

    # A subband of frequency angle phi holds image features oriented at
    # phi + pi/2, so the stripe-carrying subband sits at theta + pi/2.
    #
    # The damping depends on the subband's direction but not on its scale
    # (sigma_bar below has no level index, exactly as in the reference
    # implementation), so all the rings of one direction can be summed
    # first. `levels` therefore acts through the low-pass residual: it sets
    # how coarse a stripe still escapes the filter.
    mask = lowpass.copy()
    for center, wedge in wedges:
        dangle = abs(_wrap_half_pi(center - (theta + np.pi / 2)))
        if dangle > np.deg2rad(max_angle):
            damp = 1.0                       # subband left untouched
        else:
            sigma_bar = sigma * np.exp(-dangle ** 2 / (2 * sigma_a ** 2))
            damp = 1.0 - np.exp(-t ** 2 / (2 * max(sigma_bar, 1e-6) ** 2))
        # every scale of this direction gets the same damping
        mask += ring_sum * wedge * damp

    ny, nx = shape
    mask[ny // 2, nx // 2] = 1.0             # keep the mean height
    return mask


def mdsr(data, angle=0.0, directions=8, levels=5, sigma=5.0, sigma_a=0.3,
         max_angle=45.0, pad=False):
    """
    Remove stripe artifacts with the multidirectional stripe remover.

    See `mdsr_mask` for the parameters. `pad` mirrors the image before
    filtering and crops afterwards: the FFT treats the image as periodic,
    and a scan whose two opposite edges do not match rings along the stripe
    notch without it. It is off by default, which is the behaviour of the
    reference implementation.

    Returns the destriped image (same shape and dtype as `data`).
    """
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape
    levels = int(min(levels, max_levels(data.shape)))
    n_ref = nx if abs(np.cos(np.deg2rad(angle))) >= \
        abs(np.sin(np.deg2rad(angle))) else ny

    if pad:
        work = np.pad(data, ((ny // 2, ny - ny // 2), (nx // 2, nx - nx // 2)),
                      mode="reflect")
    else:
        work = data

    mask = mdsr_mask(work.shape, angle=angle, directions=directions,
                     levels=levels, sigma=sigma, sigma_a=sigma_a,
                     max_angle=max_angle, n_ref=n_ref)
    spectrum = np.fft.fftshift(np.fft.fft2(work))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * mask)))

    if pad:
        out = out[ny // 2:ny // 2 + ny, nx // 2:nx // 2 + nx]
    return out


# ---------------------------------------------------------------------------
# GSR - general stripe remover (variational)
# ---------------------------------------------------------------------------

# Stripe directions the difference operator supports, as (row, col) pixel
# steps: straight, 1-in-2 and diagonal. Any angle is mapped onto the
# closest of these by flipping and transposing the image, exactly as the
# reference implementation does.
GSR_STEPS = ((1, 0), (2, 1), (1, 1))


def _shift_diff(v, a, b):
    """Forward difference along the (a, b) pixel step: v[i+a, j+b] - v[i, j],
    zero where the step leaves the image."""
    ny, nx = v.shape
    out = np.zeros_like(v)
    if a < ny and b < nx:
        out[:ny - a, :nx - b] = v[a:, b:] - v[:ny - a, :nx - b]
    return out


def _shift_diff_t(w, a, b):
    """Adjoint of `_shift_diff` (its transpose as a matrix)."""
    ny, nx = w.shape
    out = np.zeros_like(w)
    if a < ny and b < nx:
        out[a:, b:] += w[:ny - a, :nx - b]
        out[:ny - a, :nx - b] -= w[:ny - a, :nx - b]
    return out


def _acc_shift_diff_t(out, w, a, b, scale, buf):
    """out += scale * (adjoint of `_shift_diff`) applied to w, without
    allocating (`buf` is scratch of the same shape)."""
    ny, nx = w.shape
    if a >= ny or b >= nx:
        return
    inner = (slice(0, ny - a), slice(0, nx - b))
    np.multiply(w[inner], scale, out=buf[inner])
    out[a:, b:] += buf[inner]
    out[inner] -= buf[inner]


def _gsr_orientation(angle):
    """Map a stripe angle onto a supported difference step.

    Returns (step, flip_rows, flip_cols, transpose): apply the flips and the
    transpose to the image, run with `step`, then undo them.
    """
    theta = np.deg2rad(angle)
    # stripe direction in array axes: (rows, cols) = (y, x)
    d = np.array([np.sin(theta), np.cos(theta)])
    flip_rows, flip_cols = d[0] < 0, d[1] < 0
    d = np.abs(d)
    transpose = d[1] > d[0]
    if transpose:
        d = d[::-1]
    steps = np.array(GSR_STEPS, dtype=float)
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    step = GSR_STEPS[int(np.argmin(np.linalg.norm(steps - d[None, :], axis=1)))]
    return step, bool(flip_rows), bool(flip_cols), bool(transpose)


def _soft(x, threshold):
    """Soft shrinkage: sign(x) * max(|x| - threshold, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def gsr_split(data, angle=0.0, mu1=1.0 / 3.0, mu2=1.0 / 300.0,
              iterations=600, proj=True):
    """
    Split `data` into (clean image, stripes) with the general stripe remover.

    The two add up to the input exactly. See the module docstring for the
    objective function.

    Args:
        angle: direction of the stripes in degrees, 0 = horizontal scan
            lines. The difference operator supports steps of 0, 26.6 and 45
            degrees (plus every flip and transpose of those), so other
            angles are snapped to the nearest one - as in the reference
            implementation.
        mu1: strength of the stripe removal. Larger removes more, and
            starts to smooth and to eat stripe-like structures.
        mu2: caution about touching real structure: larger keeps the stripe
            image sparser, so less is removed. The ratio of the two matters
            more than either alone.
        iterations: primal-dual steps. The result keeps improving with
            more; the paper recommends 10000 for a converged solution.
        proj: keep the clean image inside the value range of the input.

    Returns:
        (clean, stripes), both in the units of `data`.
    """
    data = np.asarray(data, dtype=np.float64)
    lo, hi = float(np.min(data)), float(np.max(data))
    span = hi - lo
    if span <= 0:                       # flat image: nothing to destripe
        return data.copy(), np.zeros_like(data)

    step, flip_rows, flip_cols, transpose = _gsr_orientation(angle)
    f = (data - lo) / span              # the box constraint needs [0, 1]
    if flip_rows:
        f = f[::-1, :]
    if flip_cols:
        f = f[:, ::-1]
    if transpose:
        f = f.T
    f = np.ascontiguousarray(f)

    u, s = _gsr_core(f, step, mu1, mu2, int(iterations), proj)

    if transpose:
        u = u.T
    if flip_cols:
        u = u[:, ::-1]
    if flip_rows:
        u = u[::-1, :]
    clean = np.ascontiguousarray(u) * span + lo
    # The iteration keeps u + s = f, but only to single precision; taking
    # the stripes as the remainder makes the split exact in the units of
    # the data, and it is the same quantity the previews show.
    return clean, data - clean


def _gsr_core(f, step, mu1, mu2, iterations, proj):
    """PDHGMp iteration on an image already normalized to [0, 1] and
    oriented so that the stripes run along `step`.

    Written with preallocated buffers and in single precision (as the
    reference implementation's torch tensors are), because the iteration
    count makes this the only expensive part of the module.

    Two simplifications of the dual updates, both exact:
    `p - soft(p, t)` is `clip(p, -t, t)`, and the coupled shrinkage
    `p * max(|p| - t, 0) / |p|` subtracted from p is the projection of the
    gradient vector onto the disc of radius t.
    """
    tau = 0.35
    sigma = tau
    ts = np.float32(tau * sigma)
    a, b = step
    t1 = np.float32(mu1 / sigma)         # radius for the gradient dual
    t2 = np.float32(1.0 / sigma)         # for the stripe difference
    t3 = np.float32(mu2 / sigma)         # for the stripe image
    half = np.float32(0.5)
    two = np.float32(2.0)

    f = np.ascontiguousarray(f, dtype=np.float32)
    ny, nx = f.shape

    # dual variables: b1x/b1y for the total variation of u, b2 for the
    # difference of s along the stripes, b3 for the sparsity of s
    b1x, b1y = np.zeros_like(f), np.zeros_like(f)
    b2, b3 = np.zeros_like(f), np.zeros_like(f)
    b1xbar, b1ybar = np.zeros_like(f), np.zeros_like(f)
    b2bar, b3bar = np.zeros_like(f), np.zeros_like(f)
    old1x, old1y = np.empty_like(f), np.empty_like(f)
    old2, old3 = np.empty_like(f), np.empty_like(f)

    u = f.copy()
    s = np.zeros_like(f)
    tmp, buf, norm = (np.empty_like(f) for _ in range(3))
    inner = (slice(0, ny - a), slice(0, nx - b))

    for _ in range(iterations):
        # primal step
        _acc_shift_diff_t(u, b1xbar, 1, 0, -ts, buf)
        _acc_shift_diff_t(u, b1ybar, 0, 1, -ts, buf)
        _acc_shift_diff_t(s, b2bar, a, b, -ts, buf)
        s -= ts * b3bar

        # back onto the constraint u + s = f, then into the value range
        np.subtract(f, s, out=tmp)
        tmp -= u
        tmp *= half
        u += tmp
        s += tmp
        if proj:
            np.minimum(u, 0.0, out=tmp)
            s += tmp
            np.subtract(u, 1.0, out=tmp)
            np.maximum(tmp, 0.0, out=tmp)
            s += tmp
            np.clip(u, 0.0, 1.0, out=u)

        old1x[...], old1y[...] = b1x, b1y
        old2[...], old3[...] = b2, b3

        # dual step: the gradient dual, projected onto the disc of radius t1
        b1x += _shift_diff(u, 1, 0)
        b1y += _shift_diff(u, 0, 1)
        np.hypot(b1x, b1y, out=norm)
        np.maximum(norm, t1, out=norm)
        np.divide(t1, norm, out=norm)
        b1x *= norm
        b1y *= norm
        # the stripe difference along the stripes, and the stripe image
        b2[...] = 0.0
        b2[inner] = s[a:, b:] - s[inner]
        b2 += old2
        np.clip(b2, -t2, t2, out=b2)
        np.add(old3, s, out=b3)
        np.clip(b3, -t3, t3, out=b3)

        # extrapolation of the dual variables
        for new, old, bar in ((b1x, old1x, b1xbar), (b1y, old1y, b1ybar),
                              (b2, old2, b2bar), (b3, old3, b3bar)):
            np.multiply(new, two, out=bar)
            bar -= old

    return u.astype(np.float64), s.astype(np.float64)


def gsr(data, angle=0.0, mu1=1.0 / 3.0, mu2=1.0 / 300.0, iterations=600,
        proj=True):
    """The destriped image; see `gsr_split` for the parameters."""
    return gsr_split(data, angle=angle, mu1=mu1, mu2=mu2,
                     iterations=iterations, proj=proj)[0]
