"""
Stripe removal: MDSR (Fourier filtering), GSR (variational) and DeStripe
(spectrum denoising).

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

`destripe_chen` implements the third method,

    S.-w. W. Chen, J.-L. Pellequer, "DeStripe: frequency-based algorithm
    for removing stripe noises from AFM images", BMC Struct. Biol. 11, 7
    (2011), doi:10.1186/1472-6807-11-7,

which is the only one of the three written for AFM. The other two decide
what a stripe is from a model (a frequency band, an energy); this one
decides it from the image's own spectrum: the log-amplitude spectrum of a
striped image carries the stripes as a few abnormally bright, line-shaped
groups of pixels, and DeStripe finds those pixels statistically and pulls
them down to the level of their neighbours. Nothing but the image is
needed - not even the stripe direction.

The steps, following the paper's Implementation section:

  1. LogF = log|FFT(image)|. Everything happens on that image.
  2. Heterogeneity H = normalized Laplacian x normalized intensity, in
     [0, 1]: a pixel is suspicious when it is both bright and abruptly
     brighter than its surroundings.
  3. Global sampling. A threshold Href is read off the 20-bin histogram of
     H (the first bin, walking towards higher H from the peak of the
     longest run of populated bins, that holds half the peak's count or
     less). With Iref = (max + mean)/2 of the intensities of the quiet
     pixels (H <= Href), the candidates are Pn1 = {H > Href and I > Iref}.
  4. Divide and conquer. Pn1 is split by a disk around the spectrum
     origin - the intensity-weighted inertia tensor of Pn1 gives its
     initial radius, and the disk grows in tenths of that radius until the
     fraction of candidates inside falls to `density`. Inside (C0) an
     anisotropic Gaussian is least-squares fitted to the intensities and
     only pixels above the fit stay candidates; outside (Pn2) a pixel
     stays a candidate if its local variance exceeds the variance of the
     quiet pixels.
  5. Both sets are thinned by their own histogram threshold and then kept
     only where they look like a line: a row or column that is more than
     two thirds candidates, or a run of `min_run` consecutive candidates.
  6. CVAR test. For each surviving pixel the mean and variance of the
     *non-candidate* pixels in a (2*window+1)^2 neighbourhood are taken,
     and the pixel is pulled down to that mean if it exceeds it by more
     than `cvar_k` standard deviations. Clusters are worked from their
     boundary inwards, so interior pixels see already-restored values.
  7. The filter is Phi = exp(restored LogF)/exp(LogF), in (0, 1], and the
     result is the inverse FFT of the spectrum times Phi. The phase is
     untouched, and because Phi <= 1 the method can only ever take energy
     out of the image.

Where the paper leaves a step underdetermined, the choice made here is
marked with a comment: the direction of the histogram walk in step 3, the
normalization of the inertia tensor, the "VAR test" of step 4 (the paper
names it in the flow chart but never defines it, so the variance of the
quiet pixels is used as the reference), and the region of interest used by
the line criteria in step 5.

One deliberate deviation, `keep_mean`: the paper lets the origin of the
spectrum be restored like any other pixel, and reports that for a SEM image
most of the striping was in fact the amplitude at the origin. For an AFM
height map the amplitude at the origin is the mean height, and scaling it
moves the whole surface up or down without touching a single stripe, so it
is left alone by default.
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

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

# DeStripe (Chen & Pellequer 2011). The paper fixes every one of these by
# trial and error and takes only the image as input; they are exposed here
# because AFM scans differ more than the paper's set did, and because
# seeing them is the only way to know what the method is doing.
#   window   NS of the (2NS+1)^2 CVAR neighbourhood; the paper uses 1.
#   cvar_k   how far above its neighbours a pixel must sit to be pulled
#            down, in standard deviations. The paper writes the condition
#            as I - ave > (coefficient) * std but does not print the value.
#   density  the fraction of candidates at which the central disk stops
#            growing (0.85 in the paper).
#   min_run  length of a run of candidates that counts as a line (4).
CHEN_DEFAULTS = dict(window=1, cvar_k=1.0, density=0.85, min_run=4,
                     keep_mean=True)

# The discrete Laplacian of the paper's Table 1.
_LAPLACIAN = np.array([[-1.0, -1.0, -1.0],
                       [-1.0, 8.0, -1.0],
                       [-1.0, -1.0, -1.0]])

# A row or column that is more than this fraction candidates is a line.
_LINE_FRACTION = 2.0 / 3.0


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


# ---------------------------------------------------------------------------
# DeStripe - noisy pixels of the log-amplitude spectrum (Chen & Pellequer)
# ---------------------------------------------------------------------------


def heterogeneity(image):
    """
    The paper's H: normalized Laplacian times normalized intensity, in
    [0, 1]. High where a pixel is both bright and abruptly brighter than
    its neighbours - which is what a stripe looks like in the spectrum.
    """
    lap = ndimage.convolve(image, _LAPLACIAN, mode="nearest")
    return _unit(lap) * _unit(image)


def _unit(a):
    """Offset and scale to [0, 1]; constant input becomes zeros."""
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)


def _histogram_threshold(values, bins, value_range=None):
    """
    The paper's threshold rule (steps 2-5 of 'Global sampling of pixels').

    Take the longest run of consecutive populated bins, and walk from its
    most populated bin towards higher values until a bin holds at most half
    of the peak's count; the threshold is the middle of that bin.

    (The paper says only "in the direction of increasing heterogeneity";
    starting the walk at the peak rather than at the start of the run is
    this implementation's reading - starting earlier would stop at the
    first sparse bin on the way *up* to the peak.)
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return np.inf
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    populated = counts > 0
    if not populated.any():
        return np.inf

    # longest run of populated bins
    best_start = best_len = start = length = 0
    for k, p in enumerate(populated):
        if p:
            if length == 0:
                start = k
            length += 1
            if length > best_len:
                best_start, best_len = start, length
        else:
            length = 0
    group = slice(best_start, best_start + best_len)

    peak = best_start + int(np.argmax(counts[group]))
    limit = 0.5 * counts[peak]
    for k in range(peak, best_start + best_len):
        if counts[k] <= limit:
            return 0.5 * (edges[k] + edges[k + 1])
    return float(edges[best_start + best_len])       # never falls off


def _global_sample(logf, bins=20):
    """Step 3: the first, global set of candidate pixels Pn1."""
    h = heterogeneity(logf)
    h_ref = _histogram_threshold(h, bins, value_range=(0.0, 1.0))
    quiet = logf[h <= h_ref]
    if quiet.size == 0:                  # everything is heterogeneous
        quiet = logf.ravel()
    i_ref = 0.5 * (float(quiet.max()) + float(quiet.mean()))
    var_ref = float(quiet.var())
    return (h > h_ref) & (logf > i_ref), h, var_ref


def _inertia(mask, intensity):
    """
    Center, principal widths and orientation of the candidate cloud, from
    the moment-of-inertia tensor with intensity for mass.

    The tensor is divided by the total mass (the paper does not say so, but
    the radius it derives - sqrt(sx + sy) - is only a radius if the
    eigenvalues are mean squared distances rather than sums).
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    w = intensity[mask]
    w = np.maximum(w, 0.0)               # log amplitudes can be negative
    total = float(w.sum())
    if total <= 0:
        w, total = np.ones_like(w), float(w.size)
    i0, j0 = (idx * w[:, None]).sum(axis=0) / total
    di = idx[:, 0] - i0
    dj = idx[:, 1] - j0
    tensor = np.array([[float((w * di * di).sum()), float((w * di * dj).sum())],
                       [float((w * di * dj).sum()), float((w * dj * dj).sum())]
                       ]) / total
    evals, evecs = np.linalg.eigh(tensor)
    evals = np.maximum(evals, 1e-12)
    # theta rotates the axes onto the eigenvectors of the tensor
    theta = float(np.arctan2(evecs[1, -1], evecs[0, -1]))
    sx, sy = float(evals[-1]), float(evals[0])
    sx_p = sx * np.cos(theta) ** 2 + sy * np.sin(theta) ** 2
    sy_p = sx * sy / sx_p if sx_p > 0 else sy
    return (i0, j0), float(np.sqrt(sx + sy)), max(sx_p, 1e-12), max(sy_p, 1e-12)


def _central_disk(mask, shape, center, radius, density=0.85):
    """
    Step 4: grow a disk around the origin of the spectrum in tenths of
    `radius` while the candidates inside it are denser than `density`.
    """
    ny, nx = shape
    i0, j0 = center
    ii, jj = np.ogrid[:ny, :nx]
    dist2 = (ii - i0) ** 2 + (jj - j0) ** 2
    step = max(radius / 10.0, 1.0)
    limit = np.hypot(ny, nx)
    r = step
    disk = dist2 <= r * r
    while r < limit:
        grown = dist2 <= (r + step) ** 2
        inside = int(grown.sum())
        if inside == 0 or float(mask[grown].sum()) / inside <= density:
            break
        r += step
        disk = grown
    return disk


def _gaussian_residual(logf, c0, center, sx_p, sy_p):
    """
    Step 4, central region: least-squares fit of the anisotropic Gaussian

        I(i0,j0) * exp(-c1*(i-i0)^2/sx' - c2*(j-j0)^2/sy')

    to the intensities of C0, and the residual I - fit. `c1` and `c2` are
    free, so they absorb whatever scale the widths carry.

    The origin itself is left out of the fit. The paper calls the fitted
    I(i0,j0) "the restored intensity at (i0,j0)", which it can only be if
    the fit is a prediction of that value rather than a copy of it - with
    the origin in, least squares would simply chase the spike it is meant
    to judge.

    Returns (residual over the whole image, the fitted Gaussian).
    """
    ci, cj = logf.shape[0] // 2, logf.shape[1] // 2
    c0 = c0.copy()
    c0[ci, cj] = False
    idx = np.argwhere(c0)
    if idx.size == 0:                    # nothing but the origin: no fit
        return None, None
    i0, j0 = center
    di2 = (idx[:, 0] - i0) ** 2 / sx_p
    dj2 = (idx[:, 1] - j0) ** 2 / sy_p
    values = logf[c0]
    guess = np.array([float(values.max()), 1.0, 1.0])

    def residual(p):
        return values - p[0] * np.exp(-p[1] * di2 - p[2] * dj2)

    if idx.shape[0] > 3:
        try:
            guess = least_squares(residual, guess, method="lm",
                                  max_nfev=200).x
        except Exception:
            pass                          # keep the initial guess
    ny, nx = logf.shape
    ii, jj = np.ogrid[:ny, :nx]
    fit = guess[0] * np.exp(-guess[1] * (ii - i0) ** 2 / sx_p
                            - guess[2] * (jj - j0) ** 2 / sy_p)
    return logf - fit, fit


def _var_test(image, window=1, cvar_k=1.0):
    """
    The flow chart's VAR test: a pixel is noisy if it stands more than
    `cvar_k` standard deviations above the mean of its (2*window+1)^2
    neighbourhood.

    (The paper names this test but only ever writes down its *constrained*
    version, the CVAR test of step 6 - constrained meaning that noisy
    neighbours are left out of the mean and variance. This is that same
    test without the constraint, which is the reading the name asks for.)
    """
    size = 2 * int(window) + 1
    mean = ndimage.uniform_filter(image, size=size, mode="nearest")
    sq = ndimage.uniform_filter(image * image, size=size, mode="nearest")
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return image - mean > cvar_k * std


def _line_screen(mask, roi, min_run=4):
    """
    Step 5: keep the candidates that look like a line - a row or column of
    the region of interest that is more than two thirds candidates, or a
    run of `min_run` consecutive candidates.

    (The paper says "the region of interest" without defining it; the row
    and column extents of `roi` are used, so the fraction is measured
    against the part of the row that the region actually covers.)
    """
    mask = mask & roi
    if not mask.any():
        return mask
    keep = np.zeros_like(mask)

    counts = mask.sum(axis=1)
    extent = roi.sum(axis=1)
    rows = counts > _LINE_FRACTION * np.maximum(extent, 1)
    keep[rows, :] = mask[rows, :]
    counts = mask.sum(axis=0)
    extent = roi.sum(axis=0)
    cols = counts > _LINE_FRACTION * np.maximum(extent, 1)
    keep[:, cols] |= mask[:, cols]

    run = max(int(min_run), 1)
    # an opening with a line of `run` pixels keeps exactly the runs that
    # are at least that long
    keep |= ndimage.binary_opening(mask, structure=np.ones((1, run), bool))
    keep |= ndimage.binary_opening(mask, structure=np.ones((run, 1), bool))
    return keep & roi


def _cvar_restore(logf, noisy, window=1, cvar_k=1.0):
    """
    Step 6: pull every noisy pixel down to the mean of its non-noisy
    neighbours when it sits more than `cvar_k` standard deviations above
    them.

    Pixels are visited in order of their distance to a non-noisy pixel, so
    the boundary of a cluster is restored before its interior and the
    interior has something to average over - the paper's "the test was
    performed starting at the boundary pixels of each cluster".
    """
    out = logf.copy()
    if not noisy.any():
        return out
    usable = ~noisy
    ns = max(int(window), 1)
    ny, nx = logf.shape

    # distance to the nearest non-noisy pixel, as a visiting order
    order_key = ndimage.distance_transform_cdt(noisy, metric="chessboard")
    idx = np.argwhere(noisy)
    for i, j in idx[np.argsort(order_key[noisy], kind="stable")]:
        i0, i1 = max(i - ns, 0), min(i + ns + 1, ny)
        j0, j1 = max(j - ns, 0), min(j + ns + 1, nx)
        good = usable[i0:i1, j0:j1]
        if not good.any():
            usable[i, j] = True          # nothing to compare against
            continue
        values = out[i0:i1, j0:j1][good]
        ave = float(values.mean())
        std = float(np.sqrt(values.var()))
        if out[i, j] - ave > cvar_k * std:
            out[i, j] = ave
        usable[i, j] = True              # restored: usable from now on
    return out


def destripe_chen_filter(data, window=1, cvar_k=1.0, density=0.85,
                         min_run=4, keep_mean=True):
    """
    The DeStripe filter of Chen & Pellequer (2011) for `data`.

    Returns (phi, noisy, logf): the filter image Phi in (0, 1] (the
    paper's F-image - the fraction of the spectrum amplitude that is kept
    at each frequency), the mask of the pixels found noisy, and the
    log-amplitude spectrum, all in fftshift-ed layout.
    """
    data = np.asarray(data, dtype=np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(data))
    amplitude = np.abs(spectrum)
    tiny = max(float(amplitude.max()), 1.0) * 1e-12
    logf = np.log(np.maximum(amplitude, tiny))

    # (var_ref is the paper's varref. It is computed there but never used
    # in any equation, and it is not needed by the reading of the VAR test
    # taken here either - see _var_test.)
    pn1, h, _var_ref = _global_sample(logf)
    noisy = np.zeros_like(pn1)
    origin = (data.shape[0] // 2, data.shape[1] // 2)
    origin_fit = None
    inertia = _inertia(pn1, logf)
    if inertia is not None:
        center, radius, sx_p, sy_p = inertia
        disk = _central_disk(pn1, logf.shape, center, radius, density)
        c0 = pn1 & disk
        pn2 = pn1 & ~disk

        if c0.any():
            # central region: only the pixels above the fitted Gaussian
            residual, fit = _gaussian_residual(logf, c0, center, sx_p, sy_p)
            if fit is not None:
                if c0[origin]:
                    origin_fit = float(fit[origin])
                cn1 = c0 & (residual > 0)
                if cn1.any():
                    cn1 &= h > _histogram_threshold(h[cn1], 10)
                    noisy |= _line_screen(cn1, disk, min_run)

        if pn2.any():
            pn2 &= _var_test(logf, window, cvar_k)
            if pn2.any():
                pn2 &= h > _histogram_threshold(h[pn2], 10)
                noisy |= _line_screen(pn2, ~disk, min_run)

    if keep_mean:
        # the amplitude at the origin is the mean height, and scaling it
        # moves the whole surface instead of removing a stripe
        noisy[origin] = False
        origin_fit = None

    restored = _cvar_restore(logf, noisy, window, cvar_k)
    if origin_fit is not None and origin_fit < logf[origin]:
        # the paper's "I(i0,j0) is the restored intensity at (i0,j0)": the
        # origin is set by the Gaussian model of its neighbourhood, not by
        # the local mean of the CVAR test
        restored[origin] = origin_fit
        noisy[origin] = True
    phi = np.exp(np.minimum(restored - logf, 0.0))
    return phi, noisy, logf


def destripe_chen_split(data, window=1, cvar_k=1.0, density=0.85,
                        min_run=4, keep_mean=True):
    """
    Split `data` into (clean image, stripes) with DeStripe. The two add up
    to the input exactly.

    Args:
        window: half-width NS of the (2NS+1)^2 neighbourhood of the CVAR
            test. The paper uses 1.
        cvar_k: how far above its neighbours a spectral pixel must sit to
            be pulled down to their mean, in standard deviations. Lower
            removes more.
        density: the candidate density at which the central disk stops
            growing. Larger keeps the disk smaller.
        min_run: how many candidates in a row make a line. Larger is more
            conservative - isolated bright pixels are then left alone.
        keep_mean: leave the amplitude at the origin of the spectrum, i.e.
            the mean height, untouched (see the module docstring).

    Returns:
        (clean, stripes), both in the units of `data`.
    """
    data = np.asarray(data, dtype=np.float64)
    phi, _, _ = destripe_chen_filter(data, window=window, cvar_k=cvar_k,
                                     density=density, min_run=min_run,
                                     keep_mean=keep_mean)
    spectrum = np.fft.fftshift(np.fft.fft2(data))
    clean = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * phi)))
    return clean, data - clean


def destripe_chen(data, window=1, cvar_k=1.0, density=0.85, min_run=4,
                  keep_mean=True):
    """The destriped image; see `destripe_chen_split` for the parameters."""
    return destripe_chen_split(data, window=window, cvar_k=cvar_k,
                               density=density, min_run=min_run,
                               keep_mean=keep_mean)[0]
