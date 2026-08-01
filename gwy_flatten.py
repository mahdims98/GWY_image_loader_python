"""
Flatten a scan without letting the scan's own features bend the background.

Levelling an AFM image means fitting the artifact - tilt, bow, drift, the
slow wander of the z scanner - and subtracting it. The fit is the whole
problem. Fit it to *every* pixel, as a plain plane fit or a row-by-row
polynomial does, and the features on the surface are fitted too: a row that
crosses a tall cell has its baseline pulled up by the cell, the fit
compensates by pushing that row down, and the result is a dark trench
running along each side of every raised object and a cell whose top is no
longer flat. On a sample of well-separated nanobubbles this shows up as
stripes; on a sample of cells that cover a third of the frame it shows up as
cells whose surfaces are visibly uneven, which is exactly the structure one
is trying to read.

The fix is not a better polynomial. It is to fit the background *only to the
background*, which means knowing where the background is before fitting it.
That is what this module does, following

    Y. Wang, T. Lu, X. Li and H. Wang, "Automated image
    segmentation-assisted flattening of atomic force microscopy images",
    Beilstein J. Nanotechnol. 2018, 9, 975-985. doi:10.3762/bjnano.9.91

in two steps:

  1. Segment the foreground. A threshold gives a first, always-too-small
     outline of each feature - a threshold necessarily cuts a bump partway
     up its flank - and the outline is then pushed outwards along the flank
     until it reaches flat ground (`expand_contour`). The area enclosed
     becomes an exclusion mask. Convex features (bumps, cells) and concave
     ones (pits, holes) are both handled; the paper detects a pit by
     complementing the image, which is what `detect="concave"` does.

  2. Fit the remaining pixels. Every masked pixel is simply left out of the
     least-squares problem, so nothing about a feature can enter the fitted
     surface. The fit is a polynomial along each scan line (`fit="rows"`,
     the paper's curve fitting), along each column, or over the whole image
     at once (`fit="surface"`).

The second step has a sliding-window form for backgrounds too complicated
for one polynomial (the paper's SWCF and SWSF). A window of `window` pixels
is stepped one pixel at a time; at each position a polynomial is fitted to
the background points inside it and its value is recorded at every point the
window covers; each pixel ends up with one value per window that covered it,
and those are averaged. Raising the polynomial order instead would be the
obvious alternative and it is the wrong one - a high-order fit oscillates
between the points that constrain it (Runge). A low order in a short window
does not. The paper measures the trade-off in window size: smaller windows
follow the background more closely, larger ones start leaving a corrugation
behind, and at the limit of a window as big as the image the sliding fit is
the plain fit again.

Two things here are not in the paper, and both are about images that its
nanobubbles never produced:

  * a window can land entirely inside a cell, where there is no background
    to fit at all. Such a window is dropped (it needs `MIN_FILL` of its area
    to be background), and a pixel covered by no surviving window falls back
    to the whole-image fit. Without this the fit is an extrapolation from
    whatever few pixels happened to be unmasked, which is worse than not
    fitting at all.
  * a fit is rejected outright when it would be extrapolating rather than
    interpolating - when the background it has left is bunched at one end of
    the line or one corner of the window, so that the polynomial it supports
    says nothing about the rest. A whole-line fit then drops its order until
    one of them is supportable, down to a constant; a window simply drops
    out and the whole-image fit stands in. Nothing is damped or smoothed
    towards a guess: either the data supports the fit or it does not.

The contour expansion is also not the paper's. Wang et al. evolve an active
contour under the image's gradient field, with two coefficients that set how
continuous and how smooth the contour stays. Here the mask instead grows one
pixel at a time and a pixel joins only while the local gradient says the
flank is still falling - same purpose, reaching the foot of the feature
rather than stopping partway up it, one parameter instead of two, and no
dependency beyond scipy. What it cannot do is hold a smooth contour across a
noisy edge, so `grow` adds a plain margin afterwards.

Nothing here rescales z. The image is returned as `data - background`, a
subtraction, so heights measured on the result are the heights that were
measured on the sample.
"""

import numpy as np
from scipy import ndimage

import gwy_balance as gb

# Defaults, shared with the GUI so both agree on what the buttons mean.
DEFAULTS = {
    "detect": "convex",        # which features to exclude
    "threshold": "otsu",       # how the first outline is found
    "neighbourhood": 25.0,     # adaptive window, % of the shorter side
    "sensitivity": 3.0,        # threshold offset, in robust sigmas
    "expand": 8,               # contour expansion steps (0 = off)
    "edge": 1.0,               # gradient gate, in robust sigmas
    "grow": 2,                 # plain margin added afterwards, px
    "min_area": 0.05,          # smallest feature kept, % of the frame
    "fit": "rows",             # rows / columns / surface
    "order": 3,                # polynomial order
    "window": 0,               # sliding window, px (0 = whole line/image)
    "passes": 2,               # segment, flatten, segment again, refit
}

DETECT = ("convex", "concave", "both")
THRESHOLDS = ("adaptive", "otsu")
FITS = ("rows", "columns", "surface")

EDGE_SIGMA = 1.5    # smoothing of the gradient field used by the expansion, px
MIN_FILL = 0.15     # share of a window that must be background for it to count
CHUNK = 4_000_000   # elements per batch of local normal equations
SLACK = 4.0         # how far past a fully covered fit's reach we will go
PROBE = 33          # points the reach is measured at, per axis
LOCAL_PROBE = 3     # ... and per axis for the one reach test per pixel
EPS = 1e-10         # keeps a normal matrix invertible, and nothing more


# --------------------------------------------------------------- segmentation


def _robust_sigma(values):
    """Median absolute deviation, scaled to a standard deviation."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _drop_small(mask, min_area):
    """Discard connected regions covering less than `min_area` % of the frame."""
    if not mask.any():
        return mask
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    areas = ndimage.sum(mask, labels, np.arange(1, count + 1))
    keep = 1 + np.flatnonzero(areas >= min_area / 100.0 * mask.size)
    return np.isin(labels, keep) if keep.size else np.zeros_like(mask)


def _local_median(data, size):
    """
    Median over a `size`-wide neighbourhood, taken on a coarse grid and
    interpolated back up.

    A median filter that visits every pixel with a window tens of pixels
    across is minutes of work for an answer that, by construction, has no
    structure finer than the window. Sampling one point in `size // 8` and
    interpolating between them gives the same surface for a thousandth of
    the cost.

    The edges need care, and the obvious choices are both wrong. A window
    sitting on the boundary can only look inwards, so on a bowed scan its
    median is measured from pixels that are all on one side of the point it
    belongs to; the residual then carries a rim right round the frame,
    several times the noise, and the rim is reported as features. Repeating
    the edge value drags the median inwards, and mirroring folds the trend
    back on itself, which is worse - it turns a slope into a fold. Odd
    reflection instead continues the trend straight through the boundary,
    so a window there is symmetric in value as well as in position and a
    straight slope passes through it untouched. Measured on a bowed test
    image, the share of background pixels left far enough out to be called
    features went from 7.6 % (mirrored) and 1.2 % (repeated) to none.
    """
    step = max(1, size // 8)
    coarse = data[::step, ::step]
    width = max(3, int(round(size / step)))
    width += 1 - width % 2
    # reflect cannot pad by more than the array holds, which only bites on
    # an image a few windows across
    pad = max(0, min(width // 2, min(coarse.shape) - 1))
    if pad:
        coarse = np.pad(coarse, pad, mode="reflect", reflect_type="odd")
    smooth = ndimage.median_filter(coarse, size=width, mode="nearest")
    if pad:
        smooth = smooth[pad:-pad, pad:-pad]
    # Coarse sample k is original pixel k * step, so pixel r is coarse r/step.
    grid = np.meshgrid(np.arange(data.shape[0]) / step,
                       np.arange(data.shape[1]) / step, indexing="ij")
    return ndimage.map_coordinates(smooth, grid, order=1, mode="nearest")


def adaptive_threshold(data, neighbourhood=DEFAULTS["neighbourhood"],
                       sensitivity=DEFAULTS["sensitivity"]):
    """
    Split the image into what stands above its surroundings and what sits
    below them.

    Each pixel is compared with the mean of a neighbourhood around it rather
    than with one number for the whole image, which is what makes the
    threshold adaptive: a tilted or bowed background moves the local mean
    with it and drops out of the comparison. The offset is set from the
    spread of the residual itself, measured robustly, so it does not need to
    be given in nanometres and does not change meaning between images.

    The neighbourhood has to be larger than the features - a window smaller
    than a cell sees the middle of that cell as its own background and marks
    only the rim.

    Two details keep it honest, and both are departures from the usual
    local-mean recipe.

    The neighbourhood statistic is a median, not a mean. A mean is dragged
    towards any feature inside the window, so the background beside a deep
    pit is left sitting well above the residual's centre - far enough that a
    threshold reports most of the frame as a convex feature, which is the
    opposite of the truth. A median ignores a minority of outliers entirely,
    so the trend beside a pit is the trend. It is taken on a coarse grid and
    interpolated back, which costs nothing: a statistic over a window tens
    of pixels wide has no detail worth sampling every pixel.

    The two sides are then measured against different spreads: the convex
    side against the spread of the residual *below* the centre, the concave
    side against the spread above it - in each case the side that the
    features being looked for cannot have contaminated.

    What neither trick can survive is a feature wider than the
    neighbourhood, where more than half the window is feature and the median
    follows it. That is the regime of cells rather than nanobubbles, and it
    is what `threshold="otsu"` is for.

    Args:
        data (np.ndarray): A 2D image.
        neighbourhood (float): Window size as a percentage of the shorter
            side of the image.
        sensitivity (float): Threshold offset in robust standard deviations
            of the residual.

    Returns:
        tuple: `(above, below)`, boolean masks of the convex and the concave
        pixels.
    """
    data = np.asarray(data, dtype=float)
    size = max(3, int(round(neighbourhood / 100.0 * min(data.shape))))
    empty = np.zeros(data.shape, dtype=bool)

    residual = data - _local_median(data, size)
    residual -= np.median(residual)
    low, high = residual[residual < 0.0], residual[residual > 0.0]
    if low.size == 0 or high.size == 0:
        return empty, empty.copy()
    # 1.4826 * the median absolute deviation of one side alone
    spread_low = 1.4826 * float(np.median(-low))
    spread_high = 1.4826 * float(np.median(high))
    if spread_low <= 0.0 or spread_high <= 0.0:
        return empty, empty.copy()
    return (residual > sensitivity * spread_low,
            residual < -sensitivity * spread_high)


def expand_contour(data, mask, steps=DEFAULTS["expand"],
                   edge=DEFAULTS["edge"]):
    """
    Push the outline of a feature outwards until it reaches flat ground.

    A threshold cuts a bump partway up its flank, so the mask it gives is
    always smaller than the feature. What is left outside the mask is the
    foot of the feature, and the foot is the worst possible thing to include
    in a background fit: it is the steepest part of the error.

    The mask therefore grows one pixel at a time, and a candidate pixel is
    taken only while the image is still sloping there - the gradient
    magnitude is above what the flat parts of the image show. Growth stops
    on its own when the frontier runs out of slope, so `steps` is a limit
    and not a target.

    Args:
        data (np.ndarray): The image the mask belongs to.
        mask (np.ndarray): Boolean starting mask.
        steps (int): Largest number of pixels the outline may travel.
        edge (float): How far above the background's own gradient a pixel
            has to be to count as flank, in robust standard deviations.
            Larger values stop the growth earlier.

    Returns:
        np.ndarray: The grown mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if steps <= 0 or not mask.any() or mask.all():
        return mask

    gradient = ndimage.gaussian_gradient_magnitude(
        np.asarray(data, dtype=float), EDGE_SIGMA)
    outside = gradient[~mask]
    level = float(np.median(outside)) + edge * _robust_sigma(outside)

    flank = gradient > level
    for _ in range(int(steps)):
        frontier = ndimage.binary_dilation(mask) & ~mask & flank
        if not frontier.any():
            break
        mask = mask | frontier
    return mask


def segment_foreground(data, detect=DEFAULTS["detect"],
                       threshold=DEFAULTS["threshold"],
                       neighbourhood=DEFAULTS["neighbourhood"],
                       sensitivity=DEFAULTS["sensitivity"],
                       expand=DEFAULTS["expand"], edge=DEFAULTS["edge"],
                       grow=DEFAULTS["grow"],
                       min_area=DEFAULTS["min_area"]):
    """
    Mark everything that is sample rather than background.

    Threshold, drop the specks, push each outline out to the foot of its
    feature, fill what is enclosed, and add a margin. Holes are filled after
    the expansion as well as before it: a dip in the middle of a cell is
    still cell, and letting the fit see it would put a piece of the sample
    back into the background.

    Args:
        data (np.ndarray): A 2D image.
        detect (str): `convex`, `concave` or `both`.
        threshold (str): `adaptive` (local mean, see `adaptive_threshold`)
            or `otsu` (one threshold for the whole image, on a heavily
            smoothed copy - the split that works on images whose features
            are large, such as cells).
        neighbourhood, sensitivity: Passed to `adaptive_threshold`; unused
            when `threshold` is `otsu`.
        expand, edge: Passed to `expand_contour`.
        grow (int): Plain dilation applied at the end, in pixels.
        min_area (float): Smallest feature kept, as a percentage of the
            frame.

    Returns:
        np.ndarray: A boolean mask, True on the foreground.
    """
    data = np.asarray(data, dtype=float)
    if threshold not in THRESHOLDS:
        raise ValueError(f"unknown threshold {threshold!r}, expected one of "
                         f"{list(THRESHOLDS)}")
    if detect not in DETECT:
        raise ValueError(f"unknown detect {detect!r}, expected one of "
                         f"{list(DETECT)}")

    if threshold == "otsu":
        above = gb.segment_cells(data, min_area=min_area / 100.0)
        below = gb.segment_cells(-data, min_area=min_area / 100.0)
    else:
        above, below = adaptive_threshold(data, neighbourhood, sensitivity)

    parts = []
    if detect in ("convex", "both"):
        parts.append(above)
    if detect in ("concave", "both"):
        parts.append(below)

    mask = np.zeros(data.shape, dtype=bool)
    for part in parts:
        part = _drop_small(ndimage.binary_fill_holes(part), min_area)
        part = ndimage.binary_fill_holes(expand_contour(data, part, expand,
                                                        edge))
        mask |= part
    if grow > 0 and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=int(grow))
    return mask


# ------------------------------------------------------------------- fitting


def _terms(fit, order):
    """Exponent pairs (x, y) of the polynomial, constant term first."""
    order = max(0, int(order))
    if fit == "rows":
        return [(k, 0) for k in range(order + 1)]
    if fit == "columns":
        return [(0, k) for k in range(order + 1)]
    if fit == "surface":
        return [(i, total - i)
                for total in range(order + 1)
                for i in range(total, -1, -1)]
    raise ValueError(f"unknown fit {fit!r}, expected one of {list(FITS)}")


def _window_shape(fit, window):
    """Sliding window as (height, width) in pixels; always odd."""
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    return {"rows": (1, w), "columns": (w, 1), "surface": (w, w)}[fit]


def _probe_matrix(terms, n=PROBE):
    """The polynomial's basis evaluated across the whole window, used to ask
    what the fit does where there was nothing to fit to."""
    ux = np.linspace(-1.0, 1.0, n) if max(i for i, _ in terms) else np.zeros(1)
    uy = np.linspace(-1.0, 1.0, n) if max(j for _, j in terms) else np.zeros(1)
    gx, gy = np.meshgrid(ux, uy, indexing="ij")
    return np.column_stack([(gx ** i * gy ** j).ravel() for i, j in terms])


def _reach(inverse, terms, count, spots=PROBE):
    """
    How far the fit is being asked to see.

    The variance of a least-squares prediction at a point is proportional to
    b(u)' M^-1 b(u), with b the polynomial's basis and M its normal matrix.
    Where the fit is interpolating between points it actually has, that
    number is small; where it is extrapolating across a gap, it grows
    without bound. Scaled by the number of points, it is comparable between
    fits, which is what lets one fit be rejected in favour of a lower order.
    """
    probe = _probe_matrix(terms, spots)
    return float(np.einsum("ij,jk,ik->i", probe, inverse, probe).max()) * count


def _ideal_reach(terms, coords, spots=PROBE):
    """`_reach` of the same fit on a window with nothing masked out - the
    best any fit of this order can do, and so the yardstick for the rest."""
    design = np.column_stack([coords[0] ** i * coords[1] ** j
                              for i, j in terms])
    return _reach(np.linalg.inv(design.T @ design), terms, design.shape[0],
                  spots)


def _solve(design, weight, values, terms, eps, limit):
    """Weighted least squares, or None when it would be an extrapolation.

    `eps` only keeps the normal matrix invertible; it is far too small to
    pull the answer anywhere, and it is kept off the constant term, which
    carries the background level and is the one thing the fit is always
    entitled to."""
    total = float(weight.sum())
    if total < design.shape[1]:
        return None
    matrix = design.T @ (weight[:, None] * design)
    size = design.shape[1]
    if size > 1:
        index = np.arange(1, size)
        matrix[index, index] += eps * total
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    if _reach(inverse, terms, total) > limit:
        return None
    return inverse @ (design.T @ (weight * values))


def _fill_gaps(coeffs, level):
    """Replace the rows that could not be fitted by interpolating their
    coefficients from the rows that could, so the fitted background stays
    continuous down the image instead of stepping wherever a row was too
    covered to fit."""
    good = np.isfinite(coeffs[:, 0])
    if not good.any():
        coeffs[:] = 0.0
        coeffs[:, 0] = level
        return coeffs
    index = np.arange(coeffs.shape[0])
    for c in range(coeffs.shape[1]):
        coeffs[:, c] = np.interp(index, index[good], coeffs[good, c])
    return coeffs


def _global_background(data, weight, fit, order):
    """
    One polynomial per scan line, or one over the whole image, fitted to the
    pixels with a non-zero weight.

    The order is not fixed. A scan line that crosses a cell for four fifths
    of its length has a handful of background pixels bunched at one end, and
    a cubic through them says nothing whatever about the other end - it says
    something enormous, and subtracting it wrecks the line. So each line is
    offered the requested order first and drops to the next one down
    whenever `_reach` says the fit would be extrapolating, all the way to a
    constant, which is the level of the background pixels it has and is
    always supportable. Lines with no background at all are left out and
    interpolated from their neighbours.

    Rejecting a fit outright is why nothing needs damping here: `EPS` is in
    the normal matrix to keep it invertible and is small enough to leave a
    well-covered fit exact.
    """
    ny, nx = data.shape
    present = weight > 0
    level = float(np.median(data[present])) if present.any() else 0.0

    if fit == "columns":
        return _global_background(data.T, weight.T, "rows", order).T

    if fit == "rows":
        x = np.linspace(-1.0, 1.0, nx)
        design = np.vander(x, order + 1, increasing=True)
        zero = np.zeros(nx)
        limits = [SLACK * _ideal_reach(_terms("rows", p), (x, zero))
                  for p in range(order + 1)]
        coeffs = np.full((ny, order + 1), np.nan)
        for y in range(ny):
            for p in range(order, -1, -1):
                fitted = _solve(design[:, :p + 1], weight[y], data[y],
                                _terms("rows", p), EPS, limits[p])
                if fitted is not None:
                    coeffs[y, :p + 1] = fitted
                    coeffs[y, p + 1:] = 0.0
                    break
        return _fill_gaps(coeffs, level) @ design.T

    grid_y, grid_x = np.meshgrid(np.linspace(-1.0, 1.0, ny),
                                 np.linspace(-1.0, 1.0, nx), indexing="ij")
    flat_x, flat_y = grid_x.ravel(), grid_y.ravel()
    values, w = data.ravel(), weight.ravel()
    for p in range(order, -1, -1):
        terms = _terms("surface", p)
        design = np.column_stack([flat_x ** i * flat_y ** j for i, j in terms])
        limit = SLACK * _ideal_reach(terms, (flat_x, flat_y))
        fitted = _solve(design, w, values, terms, EPS, limit)
        if fitted is not None:
            return (design @ fitted).reshape(ny, nx)
    return np.full(data.shape, level)


def _sliding_background(data, weight, fit, order, window):
    """
    The paper's sliding-window fit, in closed form.

    Done literally - fit a polynomial at every window position, in a loop -
    this is one least-squares problem per pixel, which for a 512 x 512 image
    is a quarter of a million fits. It does not have to be done that way.
    Every entry of a window's normal equations is a sum of the same quantity
    over the window (w, w*z, each weighted by a power of the offset from the
    centre), and a sum over a sliding window is a correlation, so all of the
    normal equations for the whole image come out of a handful of passes of
    `correlate1d`. Only the solve is then per pixel, and it is a batch of
    tiny symmetric systems that numpy does in one call.

    The averaging step is the same trick a second time. Window `c`
    contributes its polynomial's value at every pixel it covers, so pixel
    `j` receives one term per window, each a coefficient of that window
    times a fixed power of `j - c` - a convolution of the coefficient
    images. The denominator counts the windows that reached each pixel,
    which handles the edges of the image and the dropped windows at once.

    A window is dropped when it holds less than `MIN_FILL` of its area in
    background, and also when `_reach` says its fit would be extrapolating -
    the same rule the whole-image fit uses, and just as necessary here. A
    window whose background sits along one edge is not rare on a scan full
    of cells, and the cubic it supports says nothing about the rest of the
    window: measured on a yeast scan, dropping only the underfilled windows
    still left a fitted background swinging over 700 nm on an image whose
    features are 40 nm tall. The reach test is what makes the sliding fit
    usable on anything other than well-separated features.

    The extra right-hand sides in the solve are what make that cheap: the
    same LU factorisation that gives the coefficients gives M^-1 b for each
    probe point, and the reach falls out of a dot product.

    Returns:
        tuple: `(background, covered)`, the second being False where no
        window survived and the caller has to fall back.
    """
    terms = _terms(fit, order)
    wy, wx = _window_shape(fit, window)
    hy, hx = (wy - 1) // 2, (wx - 1) // 2
    ux = np.arange(-hx, hx + 1) / hx if hx else np.zeros(1)
    uy = np.arange(-hy, hy + 1) / hy if hy else np.zeros(1)
    kx = [ux ** a for a in range(2 * max(i for i, _ in terms) + 1)]
    ky = [uy ** b for b in range(2 * max(j for _, j in terms) + 1)]

    def moments(field, keys):
        """{(a, b): sum over the window of field * ux**a * uy**b}"""
        along_x, out = {}, {}
        for a, b in keys:
            if a not in along_x:
                along_x[a] = ndimage.correlate1d(field, kx[a], axis=1,
                                                 mode="constant", cval=0.0)
            out[(a, b)] = ndimage.correlate1d(along_x[a], ky[b], axis=0,
                                              mode="constant", cval=0.0)
        return out

    ny, nx = data.shape
    size = len(terms)
    pairs = [[(i1 + i2, j1 + j2) for i2, j2 in terms] for i1, j1 in terms]
    gram = moments(weight, {key for row in pairs for key in row} | {(0, 0)})
    rhs = moments(weight * data, terms)

    counts = gram[(0, 0)]
    valid = counts >= max(size + 1, int(round(MIN_FILL * wy * wx)))

    # One probe point per axis end is enough to catch an extrapolation, and
    # a coarse probe keeps the number of right-hand sides down.
    spots = LOCAL_PROBE if fit == "surface" else 2 * LOCAL_PROBE - 1
    probe = _probe_matrix(terms, spots)
    full = np.meshgrid(ux, uy, indexing="ij")
    limit = SLACK * _ideal_reach(terms, (full[0].ravel(), full[1].ravel()),
                                 spots)

    identity = np.eye(size)
    diagonal = np.arange(size)

    coeffs = np.zeros((ny, nx, size))
    rows = max(1, int(CHUNK // max(1, nx * size * (size + probe.shape[0]))))
    for y0 in range(0, ny, rows):
        cut = slice(y0, min(y0 + rows, ny))
        block = np.empty((cut.stop - cut.start, nx, size, size))
        for k in range(size):
            for l in range(size):
                block[..., k, l] = gram[pairs[k][l]][cut]
        block[..., diagonal, diagonal] += EPS * counts[cut][..., None]
        vector = np.stack([rhs[t][cut] for t in terms], axis=-1)
        bad = ~valid[cut]
        block[bad] = identity
        vector[bad] = 0.0
        answer = np.linalg.solve(
            block, np.concatenate(
                [vector[..., None],
                 np.broadcast_to(probe.T, block.shape[:-1] + (probe.shape[0],))],
                axis=-1))
        coeffs[cut] = answer[..., 0]
        seen = np.einsum("pk,...kp->...p", probe, answer[..., 1:]).max(axis=-1)
        valid[cut] &= seen * counts[cut] <= limit

    used = valid.astype(float)
    total = np.zeros((ny, nx))
    for k, (i, j) in enumerate(terms):
        term = ndimage.convolve1d(coeffs[..., k] * used, kx[i], axis=1,
                                  mode="constant", cval=0.0)
        total += ndimage.convolve1d(term, ky[j], axis=0, mode="constant",
                                    cval=0.0)
    reach = ndimage.convolve1d(used, np.ones(wx), axis=1, mode="constant",
                               cval=0.0)
    reach = ndimage.convolve1d(reach, np.ones(wy), axis=0, mode="constant",
                               cval=0.0)

    covered = reach > 0
    background = np.zeros((ny, nx))
    np.divide(total, reach, out=background, where=covered)
    return background, covered


def fit_background(data, mask, fit=DEFAULTS["fit"], order=DEFAULTS["order"],
                   window=DEFAULTS["window"], report=None):
    """
    Fit the background of an image, using only the pixels outside `mask`.

    Args:
        data (np.ndarray): A 2D image.
        mask (np.ndarray): Boolean; True where the image is foreground and
            must be kept out of the fit.
        fit (str): `rows` (a polynomial along each scan line, the paper's
            curve fitting), `columns` (the same down each column) or
            `surface` (one polynomial over the whole image).
        order (int): Polynomial order.
        window (int): Sliding window in pixels; 0 fits each line, or the
            image, in one piece.
        report (dict): Optional; filled with `covered`, the fraction of the
            image the sliding fit could reach.

    Returns:
        np.ndarray: The fitted background, the same shape as `data`.
    """
    data = np.asarray(data, dtype=float)
    weight = (~np.asarray(mask, dtype=bool)).astype(float)
    if fit not in FITS:
        raise ValueError(f"unknown fit {fit!r}, expected one of {list(FITS)}")

    background = _global_background(data, weight, fit, order)
    covered = 1.0
    if window and int(window) > 1:
        local, reach = _sliding_background(data, weight, fit, order,
                                           int(window))
        covered = float(reach.mean())
        background = np.where(reach, local, background)
    if report is not None:
        report["covered"] = covered
    return background


def flatten(data, detect=DEFAULTS["detect"], threshold=DEFAULTS["threshold"],
            neighbourhood=DEFAULTS["neighbourhood"],
            sensitivity=DEFAULTS["sensitivity"], expand=DEFAULTS["expand"],
            edge=DEFAULTS["edge"], grow=DEFAULTS["grow"],
            min_area=DEFAULTS["min_area"], fit=DEFAULTS["fit"],
            order=DEFAULTS["order"], window=DEFAULTS["window"],
            passes=DEFAULTS["passes"]):
    """
    Segment the foreground, fit the background to what is left, subtract it.

    There is a chicken and egg here and it has to be handled or nothing else
    works. Segmenting needs a reasonably flat image: on a raw scan the drift
    between one line and the next is routinely larger than the cells sitting
    on it, and a segmentation of that image marks the bright scan lines, not
    the sample. The mask then covers whole rows, those rows have no
    background left to fit, and the result is worse than doing nothing.

    So the first mask is taken from a plainly flattened copy - the same fit
    over every pixel, features and all. That copy is not a good image (it is
    exactly the one this module exists to avoid: the features are dented and
    trenched) but it is a good image to *segment*, because the artifact is
    gone from it and what is left standing is the sample. It is used for
    that and thrown away.

    Each further pass segments the properly flattened result of the one
    before, which is better again. The paper does the same in its last
    section: segment, flatten with that mask, segment the flattened image.
    The final fit is always taken on the original data, never on an
    already-flattened copy, so the passes refine the mask and never stack
    subtractions.

    Args:
        data (np.ndarray): A 2D image.
        detect, threshold, neighbourhood, sensitivity, expand, edge, grow,
        min_area: Passed to `segment_foreground`.
        fit, order, window: Passed to `fit_background`.
        passes (int): How many times to segment and refit.

    Returns:
        dict: With `data` (the flattened image), `background` (what was
        subtracted), `mask` (the foreground of the last pass), `coverage`
        (the fraction of the frame it holds), `covered` (the fraction the
        sliding fit reached; 1.0 when no window is used) and `starved` (the
        fraction of scan lines the mask left no background on).
    """
    data = np.asarray(data, dtype=float)
    mask = np.zeros(data.shape, dtype=bool)
    report = {"covered": 1.0}

    # the plain fit over every pixel: only ever used to find the features
    background = _global_background(data, np.ones_like(data), fit, order)

    for _ in range(max(1, int(passes))):
        mask = segment_foreground(
            data - background, detect=detect, threshold=threshold,
            neighbourhood=neighbourhood, sensitivity=sensitivity,
            expand=expand, edge=edge, grow=grow, min_area=min_area,
        )
        background = fit_background(data, mask, fit=fit, order=order,
                                    window=window, report=report)

    axis = 0 if fit == "columns" else 1
    starved = float(np.mean((~mask).sum(axis=axis) == 0))
    return {
        "data": data - background,
        "background": background,
        "mask": mask,
        "coverage": float(mask.mean()),
        "covered": report["covered"],
        "starved": 0.0 if fit == "surface" else starved,
    }
