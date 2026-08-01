"""
Give a folder of scans one shared colour range.

The problem this solves: a stack of scans of the same sample - successive
layers of a cell, or a time series - almost never shares a false-colour
range. Each image is levelled on its own, each has a different fraction of
its frame covered by cells, and each has a different amount of drift left in
it. Set the range from the whole image (percentiles, or min to max) and the
answer moves from image to image: the same physical height comes out a
different colour, and a layer whose cells happen to be taller washes out
while its neighbour stays flat. Comparing layers by eye then means comparing
two different colour scales, which is not a comparison at all.

The fix here is to stop measuring the whole image and measure two *places*
instead, both of which mean the same thing in every image:

  * the substrate between the cells, and
  * the inside of the cells.

Those two anchors are what the range is built from, so how much of the frame
each one covers no longer matters. Finding them needs a segmentation, which
is what `segment_cells` does.

Segmenting on the height histogram alone does not work on levelled AFM data.
Row alignment - the polynomial fit that makes a raw scan readable - gives
every row its own baseline, so a row that is mostly cell is pushed down
towards the substrate and the two populations smear into one broad peak. But
the *shape* survives: cells are large, smooth and connected, while the
substrate texture is fine. So the split is made on a heavily smoothed copy
of the image, where the fine texture is gone and only the large-scale "cell
or not" component is left. On that copy the histogram is bimodal again and
Otsu's threshold lands in the valley between the two classes.

Statistics are then read from a shrunken mask (`margin`), which keeps the
bright rim at the edge of each cell out of the numbers. The rim is a steep
wall, it is a small part of the area, and if it is included it drags the top
of the range up and flattens exactly the structure inside the cell that the
range is supposed to reveal.

Three ways to use the anchors, in increasing order of how much they change
the data (`balance`):

  per_image  Each image keeps its own anchors. Nothing is shared; this is
             the "before" picture, and the honest baseline to compare
             against.
  common     Every image is shifted so its substrate reads zero, and then
             all of them are drawn with one range, taken as the median of
             the individual anchors. Only an offset is applied, so heights
             stay true heights and the colour bar still means nanometres.
             A layer with genuinely taller cells will clip - that is real
             information, not a defect.
  matched    As `common`, plus a per-image gain that stretches each image's
             substrate-to-cell-top span onto the folder's median span. The
             cells and the structures in them then read the same colour
             everywhere, whatever the layer. This is a display
             normalisation: the gain rescales z, so the colour bar is no
             longer a height in nanometres and images treated this way must
             not be used to read heights off. Every gain is reported so the
             rescaling is never silent.

Nothing here is applied to the processing tab's data; it is a way of looking
at a folder.
"""

import numpy as np
from scipy import ndimage

# Defaults, shared with the GUI so both agree on what "auto" means.
CELL_FRACTION = 0.03     # smoothing width, as a fraction of the short side
MARGIN = 0.012           # mask shrink before reading statistics, likewise
MIN_AREA = 0.002         # smallest region kept, as a fraction of the frame
MIN_SAMPLE = 0.001       # smallest class an anchor may be read from, likewise
MIN_CLASS = 0.02         # smallest share of the frame a real class can have
P_LO = 2.0               # low percentile of the cell interior
P_HI = 98.0              # high percentile of the cell interior
CLIP = (0.5, 99.5)       # outlier clip used before smoothing, percent

MODES = {
    "per_image": "Per image (no sharing)",
    "common": "Common range (heights kept)",
    "matched": "Matched contrast (rescaled)",
}
DEFAULT_MODE = "matched"


def otsu_threshold(values, bins=256):
    """
    Otsu's threshold: the value that splits `values` into two classes with
    the largest possible variance between them.

    Args:
        values (np.ndarray): Values to split; shape is ignored.
        bins (int): Histogram bins used for the search.

    Returns:
        float: The threshold.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    below = np.cumsum(counts).astype(float)          # weight of class 0
    above = below[-1] - below
    moment = np.cumsum(counts * centres)
    total = moment[-1]
    ok = (below > 0) & (above > 0)
    if not ok.any():
        return float(np.median(values))
    between = np.zeros_like(centres)
    between[ok] = (total - moment[ok]) / above[ok] - moment[ok] / below[ok]
    between = between ** 2 * below * above
    return float(centres[int(np.argmax(between))])


def segment_cells(data, cell_fraction=CELL_FRACTION, min_area=MIN_AREA,
                  clip=CLIP):
    """
    Mark the cells: the large, raised, connected parts of the image.

    The image is clipped to `clip` so a single spike cannot set the scale,
    smoothed until only structure of about a cell's size is left, and split
    at Otsu's threshold. Holes are filled - a dark patch inside a cell is
    still cell - and regions smaller than `min_area` are dropped.

    Args:
        data (np.ndarray): A 2D levelled image.
        cell_fraction (float): Smoothing width as a fraction of the shorter
            side of the image. Larger values merge neighbouring cells;
            smaller ones start following the texture.
        min_area (float): Regions covering less than this fraction of the
            frame are discarded.
        clip (tuple): Percentiles the data is clipped to before smoothing.

    Returns:
        np.ndarray: A boolean mask, True on the cells.
    """
    data = np.asarray(data, dtype=float)
    low, high = np.percentile(data, clip)
    sigma = max(1.0, cell_fraction * min(data.shape))
    smooth = ndimage.gaussian_filter(np.clip(data, low, high), sigma)

    mask = smooth > otsu_threshold(smooth)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count:
        areas = ndimage.sum(mask, labels, np.arange(1, count + 1))
        keep = 1 + np.flatnonzero(areas >= min_area * data.size)
        mask = np.isin(labels, keep) if keep.size else np.zeros_like(mask)
    return mask


def measure(data, cell_fraction=CELL_FRACTION, margin=MARGIN,
            min_area=MIN_AREA, p_lo=P_LO, p_hi=P_HI, clip=CLIP):
    """
    Read one image's two anchors: the substrate level and the spread of the
    cell interiors.

    Both are read from eroded masks, so the rim of a cell counts as neither
    substrate nor interior; `margin` is how far in from the boundary the
    reading starts.

    The split is only believed when both classes are a real part of the
    frame - at least `MIN_CLASS` of it, and still big enough to take a median
    of after erosion. That test matters: in a frame packed edge to edge with
    cells, the only thing left below the threshold is often a scan artifact,
    a hole a hundred nanometres deep covering one percent of the image, and
    measuring "the substrate" there puts the anchor far below anything real
    and squashes the picture.

    When the split fails there is no substrate to measure. The image is then
    treated as all cell and its lower quartile stands in for the substrate
    level - a stand-in, and flagged as one: `degenerate` is True and the
    caller is expected to say so.

    Args:
        data (np.ndarray): A 2D levelled image, in display units.
        cell_fraction, min_area, clip: Passed to `segment_cells`.
        margin (float): Mask erosion before reading statistics, as a
            fraction of the shorter side.
        p_lo, p_hi (float): Percentiles of the cell interior used as the
            bottom and top of the range.

    Returns:
        dict: With keys `mask`, `coverage` (cell fraction of the frame),
        `background`, `low`, `median`, `high` (all in the units of `data`)
        and `degenerate`.
    """
    data = np.asarray(data, dtype=float)
    mask = segment_cells(data, cell_fraction, min_area, clip)
    coverage = float(mask.mean())

    steps = max(1, int(round(margin * min(data.shape))))
    inside = ndimage.binary_erosion(mask, iterations=steps, border_value=0)
    outside = ndimage.binary_erosion(~mask, iterations=steps, border_value=0)

    enough = max(64, MIN_SAMPLE * data.size)
    degenerate = (not MIN_CLASS <= coverage <= 1.0 - MIN_CLASS
                  or inside.sum() < enough or outside.sum() < enough)
    if degenerate:
        background = float(np.percentile(data, 25.0))
        interior = data
    else:
        background = float(np.median(data[outside]))
        interior = data[inside]
    low, middle, high = np.percentile(interior, [p_lo, 50.0, p_hi])
    return {
        "mask": mask,
        "coverage": coverage,
        "background": background,
        "low": float(low),
        "median": float(middle),
        "high": float(high),
        "degenerate": degenerate,
    }


def balance(measures, mode=DEFAULT_MODE):
    """
    Turn a folder's worth of anchors into one range and a transform per
    image.

    The shared range runs from the median of the images' low anchors to the
    median of their high anchors, both taken relative to each image's own
    substrate. A median rather than a mean, so one bad scan cannot pull the
    range off.

    Args:
        measures (list): Dicts from `measure`, in display order.
        mode (str): One of `MODES`.

    Returns:
        dict: With `mode`, `shared` (whether one range covers every image),
        `vmin`/`vmax` (the shared range, None when not shared), `offsets`,
        `gains` and `ranges` (a (vmin, vmax) pair per image).

    Raises:
        ValueError: If `mode` is not one of `MODES`, or `measures` is empty.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of "
                         f"{sorted(MODES)}")
    if not measures:
        raise ValueError("no measurements to balance")

    lows = np.array([m["low"] - m["background"] for m in measures])
    highs = np.array([m["high"] - m["background"] for m in measures])
    vmin, vmax = float(np.median(lows)), float(np.median(highs))
    if not vmax > vmin:                      # a flat folder; keep it drawable
        vmax = vmin + max(abs(vmin), 1e-9)

    if mode == "per_image":
        return {
            "mode": mode, "shared": False, "vmin": None, "vmax": None,
            "offsets": [0.0] * len(measures),
            "gains": [1.0] * len(measures),
            "ranges": [(m["low"], m["high"]) for m in measures],
        }

    offsets = [-m["background"] for m in measures]
    if mode == "common":
        gains = [1.0] * len(measures)
    else:                                    # matched
        spans = highs.copy()
        spans[spans <= 0] = np.nan           # no measurable step: leave it be
        gains = [float(vmax / s) if np.isfinite(s) else 1.0 for s in spans]
    return {
        "mode": mode, "shared": True, "vmin": vmin, "vmax": vmax,
        "offsets": offsets, "gains": gains,
        "ranges": [(vmin, vmax)] * len(measures),
    }


def apply_levels(data, offset, gain):
    """
    Put an image on the balanced scale: shift it, then stretch it.

    Args:
        data (np.ndarray): The image.
        offset (float): Added first; `-background` puts the substrate at 0.
        gain (float): Multiplied after. 1.0 leaves heights untouched.

    Returns:
        np.ndarray: The transformed image.
    """
    return (np.asarray(data, dtype=float) + offset) * gain
