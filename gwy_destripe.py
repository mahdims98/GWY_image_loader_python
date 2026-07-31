"""
Multidirectional stripe removal (MDSR).

Implements the destriping method of

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
