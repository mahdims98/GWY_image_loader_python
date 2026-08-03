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


Finding the features by shape instead of by height
--------------------------------------------------

Both of the paper's thresholds ask the same question - is this pixel high?
- and that question has a wrong answer on any object whose parts sit at
different levels, which is most real samples. An object tilted in the scan,
an object with a dip in its middle, two objects of the same kind on an
uneven substrate: one threshold takes a bite out of every one of them, and
the bite it takes is put back into the background fit. The levelling this
module exists to do is then bent by exactly the features it was trying to
leave out.

`threshold="shape"` asks a different question, and it is the question
`gwy_segment` - the segmentation behind the 3D viewer - was written to
answer. It never looks at a height. It looks at where the height *changes*:
the gradient is large along the rim of an object and small anywhere the
surface is merely smooth, whatever level that smooth surface sits at, so
thresholding it gives a set of walls, the frame is divided by those walls
into patches, and a patch is a feature when it is large enough and smoother
inside than the frame is on average. A watershed then hands each feature
the half of its rim that faces it, so an object keeps its own flank and its
neighbour keeps the other.

Because no height is consulted, two objects of the same kind at different
levels are found equally well, and that is the case a single threshold
cannot survive. Put two discs on a textured field, one 3 nm up and one 90:
the outlines return both of them entire, while Otsu - which has to put its
one cut somewhere - keeps the tall one and loses every pixel of the short
one, and the adaptive threshold loses it too. Neither is misbehaving. There
is simply no height that separates a 3 nm disc from the substrate and a
90 nm disc from the same substrate at the same time, and the whole of the
lost disc then goes into the background fit.

Three things to know before choosing it.

  * Nothing about it knows which way is up, so a pit is found by the same
    rim that finds a bump and `detect` has nothing to select between. It is
    ignored when this route is in use.
  * `smoothness` decides which side of a boundary is the sample, and it
    starts out assuming the sample is the smooth thing on a textured field.
    That is a fact about a specimen and not about specimens; on a substrate
    rougher than what lies on it the assumption is the wrong way round, and
    raising the setting past 1 swaps the two over. It is the one control
    here that repays a look at the contour before the result is trusted.
  * It answers what a segmentation should answer and not what a fit needs
    to hear, and the difference matters. Its job is to enclose the whole
    sample; a fit's need is for whatever is left over to be enough to fit
    through. Over 30 of the scans here it marked more than 85 % of the
    frame on 20 and left at least one scan line with no background at all
    on 23, against 10 and 11 for Otsu and none for the adaptive threshold -
    not because it was wrong, but because on these samples the cells really
    do cover the frame. That is why it is offered rather than assumed:
    `otsu` is still the default.

`threshold="adaptive"` and `threshold="otsu"` are unchanged. A local median
is the cheaper answer on a frame of nanobubbles all at one level, and
Otsu's single split is what handles a feature so large the local median
follows it.

Three settings drive the shape route, against the thirteen `gwy_segment`
offers its own editor. The ten it is not given are fixed at that module's
defaults, and they are of two kinds: the ones about telling one object from
another, which a background fit never asks since it only ever wants the
union, and the ones belonging to the two detectors that are not run.
`gwy_segment` also looks for thin ridges and for small raised specks; a
background fit wants the things that have area, and anything with a rim is
already walled off by the outlines.

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

A second paper,

    J. Zhang, A. Biswas, J. Rade, C. Shukla, J. Ren, A. Sarkar,
    A. Krishnamurthy and A. Balu, "Artifact Removal and Image Restoration in
    AFM: A Structured Mask-Guided Directional Inpainting Approach",
    arXiv:2602.04051 (2026),

builds a whole pipeline - classify the scan, segment the artifacts with a
network, inpaint them - whose flattening stage ("Smart Flatten", its section
2.4) is the same idea as Wang's with three additions, and those three are
here:

  * the fit can run down columns as well as along rows, it can do both in
    turn (`fit="both"`: a polynomial off every row, then one off every
    column of what is left), and it can pick the direction itself
    (`fit="auto"`, see `choose_direction`);
  * regions can be excluded by hand, on top of whatever the segmentation
    found (`exclude`). The paper is blunt about why: an automatic mask does
    not always cover a step edge or an unusual structure, and one that is
    left in bends the baseline. In the GUI this is a rectangle dragged on
    the image.

Its remaining pieces are not here, because this module already does the same
job better or is not trying to do it at all: its exclusion mask is a global
`|z - mean| > k*sigma` threshold plus a dilation, its fallback for a scan
line with too few background pixels is to subtract that line's median, and
the rest of the paper - the ResNet classifier, the segmentation network and
the Telea inpainting - is about repairing artifacts rather than levelling.

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
import gwy_segment as gs

# Defaults, shared with the GUI so both agree on what the buttons mean.
DEFAULTS = {
    "detect": "convex",        # which features to exclude (not for `shape`)
    "threshold": "otsu",       # how the first outline is found
    "detail": gs.DEFAULTS["detail"],          # shape: edge scale, % of frame
    "edge_level": gs.DEFAULTS["edge_level"],  # shape: wall level, robust sigmas
    "smoothness": gs.DEFAULTS["smoothness"],  # shape: patch against the frame
    "feature_size": 3.0,       # otsu blur, % of the shorter side
    "neighbourhood": 25.0,     # adaptive window, % of the shorter side
    "sensitivity": 3.0,        # threshold offset, in robust sigmas
    "expand": 8,               # contour expansion steps (0 = off)
    "edge": 1.0,               # gradient gate, in robust sigmas
    "grow": 2,                 # plain margin added afterwards, px
    "min_area": 0.05,          # smallest feature kept, % of the frame
    "fit": "rows",             # rows / columns / both / surface / auto
    "order": 3,                # polynomial order
    "window": 0,               # sliding window, px (0 = whole line/image)
    "passes": 1,               # times to look for the features
}

DETECT = ("convex", "concave", "both")
THRESHOLDS = ("shape", "adaptive", "otsu")
FITS = ("rows", "columns", "both", "surface", "auto")
LINES = ("rows", "columns")   # the fits that are one polynomial per scan line

EDGE_SIGMA = 1.5    # smoothing of the gradient field used by the expansion, px
MIN_FILL = 0.15     # share of a window that must be background for it to count
CHUNK = 4_000_000   # elements per batch of local normal equations
SLACK = 4.0         # how far past a fully covered fit's reach we will go
PROBE = 33          # points the reach is measured at, per axis
LOCAL_PROBE = 3     # ... and per axis for the one reach test per pixel
EPS = 1e-10         # keeps a normal matrix invertible, and nothing more
SEED_ORDER = 2      # per-line polynomial that reveals the features; 2 was
                    # measured best over 18 scans, see `seed_background`


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


def _square_side(min_area, x_real, y_real):
    """
    A smallest-feature area, as the side of the square of that area.

    This module measures a size as a share of the frame's *area*, because
    that is what a fit cares about - how much of the image a feature takes
    away from it. `gwy_segment` measures every size as a length, a share of
    the frame's longer side, because its filters are convolutions and a
    convolution has a width. The two say the same thing about the same
    feature, and this is where one is turned into the other.
    """
    frame = max(x_real, y_real)
    if frame <= 0.0:
        return 0.0
    side = float(np.sqrt(max(0.0, min_area) / 100.0 * x_real * y_real))
    return 100.0 * side / frame


def outline_mask(data, detail=DEFAULTS["detail"],
                 edge_level=DEFAULTS["edge_level"],
                 smoothness=DEFAULTS["smoothness"],
                 min_area=DEFAULTS["min_area"], dx=None, dy=None):
    """
    Mark the features by their outlines, without looking at their heights.

    The work is `gwy_segment.find_outlines`, reached through
    `gwy_segment.segment`; the module docstring above says why an edge is a
    better boundary than a height, and that module's says how the edges are
    found and how a patch is judged. What is decided here is only what to
    ask it for.

    One detector, and no separation of one object from the next: a fit never
    asks which object a pixel belongs to, only whether it belongs to one, so
    everything `gwy_segment` does to tell two touching objects apart is work
    whose answer would be thrown away. What comes back is the union of the
    regions - which is also why `detect` has nothing to select here. The
    outlines find a pit by the same rim that finds a bump.

    Args:
        data (np.ndarray): A 2D image.
        detail (float): The scale the edges are measured at, as a percentage
            of the frame - roughly the width of the thinnest rim worth
            seeing. Larger ignores fine texture and rounds off corners.
        edge_level (float): How far above the frame's typical edge strength
            a rim has to be, in robust sigmas. Lower walls off more.
        smoothness (float): A patch is a feature when its own edge strength
            is below this multiple of the frame's median - when it is
            smoother than the frame is on average. This is the setting that
            decides which side of a boundary is the sample, and it is worth
            knowing which way round: it assumes the sample is the smooth
            thing and the field is the textured one. On a substrate rougher
            than what lies on it, raise it past 1 to swap the two over. 0
            turns the test off and keeps every patch that is large enough,
            which on any real scan is the whole frame.
        min_area (float): Smallest feature kept, as a percentage of the
            frame.
        dx, dy (float): Pixel size. Only their ratio matters, and it matters
            whenever the pixels are not square - a scan of 1024 x 512 pixels
            over a square frame needs a filter twice as wide in one
            direction as the other. Without them the frame is measured in
            pixels, which is right for a square-pixel scan and wrong by that
            ratio for any other.

    Returns:
        np.ndarray: A boolean mask, True on the foreground.
    """
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape
    x_real = nx * float(dx) if dx else float(nx)
    y_real = ny * float(dy) if dy else float(ny)
    found = gs.segment(
        data, x_real, y_real, methods=("outline",),
        detail=float(detail), edge_level=float(edge_level),
        smoothness=float(smoothness),
        min_size=_square_side(min_area, x_real, y_real),
        separate=0.0,
    )
    return found.mask()


def segment_foreground(data, detect=DEFAULTS["detect"],
                       threshold=DEFAULTS["threshold"],
                       neighbourhood=DEFAULTS["neighbourhood"],
                       sensitivity=DEFAULTS["sensitivity"],
                       expand=DEFAULTS["expand"], edge=DEFAULTS["edge"],
                       grow=DEFAULTS["grow"],
                       min_area=DEFAULTS["min_area"],
                       feature_size=DEFAULTS["feature_size"],
                       detail=DEFAULTS["detail"],
                       edge_level=DEFAULTS["edge_level"],
                       smoothness=DEFAULTS["smoothness"],
                       dx=None, dy=None):
    """
    Mark everything that is sample rather than background.

    Find the features, drop the specks, push each outline out to the foot of
    its feature, fill what is enclosed, and add a margin. Holes are filled
    after the expansion as well as before it: a dip in the middle of a cell
    is still cell, and letting the fit see it would put a piece of the
    sample back into the background.

    The expansion is worth having whichever route found the features, and
    for slightly different reasons. A threshold cuts a bump partway up its
    flank and stops short of the foot. The shape route stops on the crest of
    the rim, which is the steepest point of the flank rather than its end.
    Both leave the foot of the feature outside the mask, and the foot is the
    worst thing a background fit can be given.

    Args:
        data (np.ndarray): A 2D image.
        detect (str): `convex`, `concave` or `both`. Ignored by the `shape`
            route, which finds a pit by the same rim that finds a bump.
        threshold (str): `shape` (patches walled off by the image's own
            edges - see `outline_mask` and the module docstring),
            `adaptive` (local median, see `adaptive_threshold`) or `otsu`
            (one threshold for the whole image, on a heavily smoothed copy -
            the split that works on images whose features are large, such as
            cells).
        detail, edge_level, smoothness: Passed to `outline_mask`; unused
            unless `threshold` is `shape`.
        neighbourhood, sensitivity: Passed to `adaptive_threshold`; unused
            unless `threshold` is `adaptive`.
        expand, edge: Passed to `expand_contour`.
        grow (int): Plain dilation applied at the end, in pixels.
        min_area (float): Smallest feature kept, as a percentage of the
            frame.
        feature_size (float): How much the image is blurred before the
            single `otsu` threshold, as a percentage of the shorter side;
            unused unless `threshold` is `otsu`. One threshold for the
            whole image has to be taken on a smoothed copy, or the texture
            of the sample splits instead of the sample, and this sets that
            scale. It also sets the finest shape the mask can take, because
            the outline follows that smoothed copy: measured on a cross with
            square corners, a blur of 8 % of the frame gave an outline
            agreeing with the true shape to 0.78 and one of 0.5 % agreed to
            1.00, the difference being background swallowed at the corners.
            Lower it to follow real edges and catch thin structure, raise it
            to ignore texture and keep only the large features.
        dx, dy (float): Pixel size, for the `shape` route; see
            `outline_mask`.

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

    if threshold == "shape":
        # One mask rather than a convex one and a concave one, and `detect`
        # left out of it: the outlines cannot tell a bump from a pit, so
        # they return both and there is nothing to choose between.
        parts = [outline_mask(data, detail, edge_level, smoothness, min_area,
                              dx, dy)]
    else:
        if threshold == "otsu":
            blur = max(1e-6, feature_size / 100.0)
            above = gb.segment_cells(data, cell_fraction=blur,
                                     min_area=min_area / 100.0)
            below = gb.segment_cells(-data, cell_fraction=blur,
                                     min_area=min_area / 100.0)
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
    # `both` and `auto` are resolved by `fit_background` and never arrive here
    raise ValueError(f"unknown fit {fit!r}, expected one of "
                     f"{list(LINES) + ['surface']}")


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


def _global_background(data, weight, fit, order, report=None):
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

    That ladder is also the one thing here that behaves discontinuously, and
    it is worth knowing about, so `report` is filled with the share of lines
    that had to come down. A line at order 3 and its neighbour at order 1 are
    levelled by visibly different curves, and on a frame two thirds covered
    by sample a single pixel added to the mask can flip a line from one to
    the other and move it by tens of nanometres. The alternative - carrying
    on with a cubic that the data does not support - is worse, so the ladder
    stays; but a scan where most lines are coming down is a scan asking to be
    fitted at a lower order in the first place.
    """
    ny, nx = data.shape
    present = weight > 0
    level = float(np.median(data[present])) if present.any() else 0.0

    if fit == "columns":
        return _global_background(data.T, weight.T, "rows", order, report).T

    if fit == "rows":
        x = np.linspace(-1.0, 1.0, nx)
        design = np.vander(x, order + 1, increasing=True)
        zero = np.zeros(nx)
        limits = [SLACK * _ideal_reach(_terms("rows", p), (x, zero))
                  for p in range(order + 1)]
        coeffs = np.full((ny, order + 1), np.nan)
        used = np.full(ny, -1)
        for y in range(ny):
            for p in range(order, -1, -1):
                fitted = _solve(design[:, :p + 1], weight[y], data[y],
                                _terms("rows", p), EPS, limits[p])
                if fitted is not None:
                    coeffs[y, :p + 1] = fitted
                    coeffs[y, p + 1:] = 0.0
                    used[y] = p
                    break
        if report is not None:
            report["reduced"] = float(np.mean(used < order))
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
            if report is not None:
                report["reduced"] = 0.0 if p == order else 1.0
            return (design @ fitted).reshape(ny, nx)
    if report is not None:
        report["reduced"] = 1.0
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


def _line_jitter(data, mask=None):
    """
    How far each line sits from the one before it, beyond what the noise in
    the lines themselves would explain. Robust: a straight tilt gives zero,
    because the step from line to line is then the same every time.
    """
    values = np.asarray(data, dtype=float)
    if mask is not None:
        values = np.where(np.asarray(mask, dtype=bool), np.nan, values)
    counts = np.isfinite(values).sum(axis=1)
    ok = counts > 0
    if int(ok.sum()) < 3:
        return 0.0
    level = np.nanmedian(values[ok], axis=1)
    seen = _robust_sigma(np.diff(level))

    # A median of n points has a standard error of sqrt(pi/2) * sigma / sqrt(n)
    # and this is the difference of two of them, so even lines that are
    # perfectly aligned show this much jitter. Subtracting it in quadrature is
    # what makes the two directions comparable when the image is not square,
    # or when one direction is noisier than the other. `sigma` comes from the
    # pixel-to-pixel difference along the line, whose spread is sqrt(2) sigma.
    noise = _robust_sigma(np.diff(values[ok], axis=1)) / np.sqrt(2.0)
    floor = np.sqrt(np.pi / 2.0) * noise * np.sqrt(2.0 / max(1.0,
                                                            float(np.median(counts[ok]))))
    return float(np.sqrt(max(0.0, seen ** 2 - floor ** 2)))


def choose_direction(data, mask=None):
    """
    Which way the scan lines run, measured rather than assumed.

    Zhang et al. pick the fitting direction from "the dominant slope
    direction". Taken literally that is the wrong statistic: a smooth tilt is
    removed just as well by a fit along rows as by one down columns, so the
    slope says nothing about which to choose. What only a line-by-line fit
    can remove is the part of the background that is *incoherent* between
    neighbouring lines - the offset the z scanner has drifted to by the time
    it starts the next line, and which no smooth surface can follow. That
    lands between lines along the slow axis, and it is removed by fitting
    along the fast one.

    So the two directions are compared on exactly that quantity: the spread
    of the step from one line to the next, with the part of it that the noise
    inside the lines already accounts for taken back out (`_line_jitter`).
    Whichever direction shows more of it is the direction the scan lines run,
    and the one to fit along. A tie means neither is drifting relative to the
    other, in which case the background is smooth and either fit will follow
    it - the choice does not matter, and it goes to `rows`.

    Args:
        data (np.ndarray): A 2D image.
        mask (np.ndarray): Optional foreground mask, whose pixels are left
            out of the measurement.

    Returns:
        str: `rows` or `columns`.
    """
    data = np.asarray(data, dtype=float)
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    across = _line_jitter(data, mask)
    down = _line_jitter(data.T, None if mask is None else mask.T)
    return "rows" if across >= down else "columns"


def _fit_once(data, weight, fit, order, window):
    """One fit in one direction: the whole-line (or whole-image) polynomial,
    with the sliding window laid over it wherever the window reached."""
    note = {"reduced": 0.0}
    background = _global_background(data, weight, fit, order, note)
    covered = 1.0
    if window and int(window) > 1:
        local, reach = _sliding_background(data, weight, fit, order,
                                           int(window))
        covered = float(reach.mean())
        background = np.where(reach, local, background)
    return background, covered, note["reduced"]


def fit_background(data, mask, fit=DEFAULTS["fit"], order=DEFAULTS["order"],
                   window=DEFAULTS["window"], report=None):
    """
    Fit the background of an image, using only the pixels outside `mask`.

    Args:
        data (np.ndarray): A 2D image.
        mask (np.ndarray): Boolean; True where the image is foreground and
            must be kept out of the fit.
        fit (str): `rows` (a polynomial along each scan line, Wang's curve
            fitting), `columns` (the same down each column), `both` (rows and
            then columns, Zhang's two-step), `surface` (one polynomial over
            the whole image) or `auto` (rows or columns, whichever
            `choose_direction` measures).
        order (int): Polynomial order.
        window (int): Sliding window in pixels; 0 fits each line, or the
            image, in one piece.
        report (dict): Optional; filled with `covered`, the fraction of the
            image the sliding fit could reach, `fit`, the direction actually
            used once `auto` has been resolved, and `reduced`, the share of
            scan lines that could not support `order` and were fitted lower.

    Returns:
        np.ndarray: The fitted background, the same shape as `data`.
    """
    data = np.asarray(data, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    weight = (~mask).astype(float)
    if fit not in FITS:
        raise ValueError(f"unknown fit {fit!r}, expected one of {list(FITS)}")

    if fit == "auto":
        fit = choose_direction(data, mask)

    if fit == "both":
        # Every row levelled, then every column of what that leaves. The
        # second fit sees no row-to-row drift, so what it removes is the
        # column-to-column part - the drift a single line fit cannot reach,
        # at the price of the sample being fitted around twice.
        first, covered, reduced = _fit_once(data, weight, "rows", order,
                                            window)
        second, again, more = _fit_once(data - first, weight, "columns",
                                        order, window)
        background = first + second
        covered = min(covered, again)
        reduced = max(reduced, more)
    else:
        background, covered, reduced = _fit_once(data, weight, fit, order,
                                                 window)

    if report is not None:
        report["covered"] = covered
        report["fit"] = fit
        report["reduced"] = reduced
    return background


def seed_background(data, fit=DEFAULTS["fit"], mask=None):
    """
    The crude flattening the features are found on.

    A plane subtracted and then a low-order polynomial taken off every scan
    line: the standard way of making a raw scan readable, and the thing to
    reach for when the question is *where are the features* rather than *how
    tall are they*. Without a mask it dents and trenches the features, which
    is exactly why it is never returned - but it puts the whole frame on one
    level, and that is what a threshold needs.

    Three decisions worth stating. The order is fixed at `SEED_ORDER`
    rather than following the order asked for the real fit, and 2 is not a
    guess: scored against `gwy_balance.segment_cells` on a properly
    flattened image - an independent segmentation, already checked against
    these same scans - a second-order seed agreed with it to a mean
    intersection-over-union of 0.91 across 18 scans from six sessions, and
    was the best of the three on every single one. First order managed 0.48
    and third order 0.57: too little leaves the drift in, too much starts
    following the features. The direction follows
    the scan lines rather than the real fit's, unless the real fit is down
    columns - a scan rotated 90 degrees being the one case where the lines
    genuinely run the other way. And it is a line fit even when the real fit
    is a surface, because drift lands *between* scan lines and no surface,
    of any order, can take it out: a surface-flattened image still has the
    line-to-line steps in it, and segmenting that finds the bright scan
    lines instead of the sample.

    Together those mean the mask comes out the same whether the background
    is afterwards fitted along rows or as a surface. That is the point.
    Which features are on the sample is a fact about the sample, and it
    should not change because of how the background is going to be removed.

    Subtracting a plane first, as one does by hand, is left out because here
    it would do nothing: a plane is linear along every scan line, so a
    per-line polynomial of order one or more absorbs it exactly. The line
    fit alone is the same surface, to the last digit.

    Args:
        data (np.ndarray): A 2D image.
        fit (str): The fit the caller intends to use afterwards; only its
            direction is taken, and only when it is `columns`.
        mask (np.ndarray): Optional; features already known, left out of
            this fit too, so a second look is taken at an image whose
            features are no longer dented.

    Returns:
        np.ndarray: The background to subtract for segmentation purposes.
    """
    data = np.asarray(data, dtype=float)
    weight = (np.ones(data.shape) if mask is None
              else (~np.asarray(mask, dtype=bool)).astype(float))
    lines = "columns" if fit == "columns" else "rows"
    return _global_background(data, weight, lines, SEED_ORDER)


def flatten(data, detect=DEFAULTS["detect"], threshold=DEFAULTS["threshold"],
            neighbourhood=DEFAULTS["neighbourhood"],
            sensitivity=DEFAULTS["sensitivity"], expand=DEFAULTS["expand"],
            edge=DEFAULTS["edge"], grow=DEFAULTS["grow"],
            min_area=DEFAULTS["min_area"], fit=DEFAULTS["fit"],
            order=DEFAULTS["order"], window=DEFAULTS["window"],
            passes=DEFAULTS["passes"], exclude=None,
            feature_size=DEFAULTS["feature_size"],
            detail=DEFAULTS["detail"], edge_level=DEFAULTS["edge_level"],
            smoothness=DEFAULTS["smoothness"], dx=None, dy=None):
    """
    Segment the foreground, fit the background to what is left, subtract it.

    There is a chicken and egg here and it has to be handled or nothing else
    works. Segmenting needs a reasonably flat image: on a raw scan the drift
    between one line and the next is routinely larger than the cells sitting
    on it, and a segmentation of that image marks the bright scan lines, not
    the sample. The mask then covers whole rows, those rows have no
    background left to fit, and the result is worse than doing nothing.

    So finding the features and removing the background are kept apart. The
    features are always looked for on a `seed_background` copy - a plane off
    and a low-order polynomial off every scan line - whatever fit is going
    to be used afterwards. Only then is the background fitted, once, the way
    the caller asked.

    That single look is already the paper's two-step segmentation: its last
    section segments, flattens with the resulting mask and segments the
    flattened image, and the seed here *is* that flattening. `passes` above
    1 repeats it again, each time with the previous mask excluded from the
    seed as well, and on this data it makes the mask steadily worse - the
    same 18 scans score 0.91 at one pass, 0.47 at two, 0.32 at three. The
    reason is specific and worth knowing before raising it: once the
    features are restored to their full height they have a wide spread of
    their own, and Otsu's threshold, which splits a histogram wherever that
    separates it best, starts splitting *inside* the features instead of
    between them and the substrate. So it is left at 1.

    The alternative - segmenting the properly flattened result of the pass
    before, which is what the paper's last section does - was tried and is
    worse here, because it makes the mask depend on the fit. With
    `fit="surface"` the flattened image still has every line-to-line step in
    it (no surface of any order can remove drift that lands between lines),
    so segmenting it marks the scan lines, and the mask that comes back
    disagrees with the one from `fit="rows"` on three quarters of its area.
    Seeding every pass instead brings that disagreement to nothing: the same
    image gives the same mask whether the background is then fitted along
    rows or as a surface, which is the only defensible answer, since which
    features are on the sample is a fact about the sample.

    `exclude` is the other paper's contribution: an area the caller already
    knows does not belong in the background - a step edge, a piece of debris,
    a structure the threshold has no reason to recognise - marked by hand and
    added to the mask. It is kept out of the seed as well as out of the final
    fit, so it cannot bend the image the features are looked for on either.

    Args:
        data (np.ndarray): A 2D image.
        detect, threshold, neighbourhood, sensitivity, expand, edge, grow,
        min_area, feature_size, detail, edge_level, smoothness, dx, dy:
            Passed to `segment_foreground`.
        fit, order, window: Passed to `fit_background`.
        passes (int): How many times to look for the features.
        exclude (np.ndarray): Optional boolean mask, True on pixels to keep
            out of the fit whatever the segmentation decides about them.

    Returns:
        dict: With `data` (the flattened image), `background` (what was
        subtracted), `mask` (the foreground of the last pass, `exclude`
        included), `coverage` (the fraction of the frame it holds), `covered`
        (the fraction the sliding fit reached; 1.0 when no window is used),
        `starved` (the fraction of scan lines the mask left no background
        on), `reduced` (the share of lines that could not support `order`)
        and `fit` (the direction used, which is the answer when `auto` was
        asked for).
    """
    data = np.asarray(data, dtype=float)
    mask = np.zeros(data.shape, dtype=bool)
    report = {"covered": 1.0, "reduced": 0.0}

    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != data.shape:
            raise ValueError(f"exclude is {exclude.shape}, image is "
                             f"{data.shape}")
        if not exclude.any():
            exclude = None

    # Resolved here rather than inside the fit so that the seed, the fit and
    # the report all speak of the same direction.
    if fit == "auto":
        fit = choose_direction(data, exclude)

    for step in range(max(1, int(passes))):
        known = mask if step else np.zeros(data.shape, dtype=bool)
        if exclude is not None:
            known = known | exclude
        # only ever used to find the features, never returned
        seed = seed_background(data, fit, known if known.any() else None)
        mask = segment_foreground(
            data - seed, detect=detect, threshold=threshold,
            neighbourhood=neighbourhood, sensitivity=sensitivity,
            expand=expand, edge=edge, grow=grow, min_area=min_area,
            feature_size=feature_size, detail=detail, edge_level=edge_level,
            smoothness=smoothness, dx=dx, dy=dy,
        )
        if exclude is not None:
            mask = mask | exclude

    background = fit_background(data, mask, fit=fit, order=order,
                                window=window, report=report)

    free = ~mask
    starved = max(float(np.mean(free.sum(axis=1) == 0))
                  if fit in ("rows", "both") else 0.0,
                  float(np.mean(free.sum(axis=0) == 0))
                  if fit in ("columns", "both") else 0.0)
    return {
        "data": data - background,
        "background": background,
        "mask": mask,
        "coverage": float(mask.mean()),
        "covered": report["covered"],
        "starved": starved,
        "reduced": report["reduced"],
        "fit": fit,
    }
