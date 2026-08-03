"""
Split an AFM height map into the things on it, and throw the rest away.

The job is to mark the objects, let the user overrule the marking by hand,
and then either flatten everything else down to the background or make it
transparent so only the marked objects are left standing.

Nothing here knows about windows or renderers. It takes an array and gives
back an array and a label map, so the 3D viewer, the processing GUI and a
script all get the same answer.


Regions have no type
--------------------

Every region that comes out of here is just a region. There is no notion of
a cell or a particle or a fibre, because that is a fact about the specimen
and this module has never seen the specimen. What a region *is* is the
user's to say; what this module offers is a proposal and the means to
correct it.

The three detectors below are therefore three *ways of looking*, not three
kinds of thing. Any mixture of them can run, and everything they find lands
in one undifferentiated list.


Outlines: shape, not height
---------------------------

This is the detector to reach for first, and the reason is that height is a
bad way to find an object.

The obvious approach - threshold the heights, keep what is high - fails on
anything whose parts sit at different levels, which is most real samples. An
object tilted in the scan, an object with a dip in the middle, two objects
of the same kind at different heights on an uneven substrate: a single
threshold takes a bite out of every one of them. Levelling makes it worse
rather than better, because row alignment gives every scan line its own
baseline and a row that is mostly object gets pushed down towards the
background.

So this detector never looks at a height. It looks at where the height
*changes*. The gradient magnitude is large along the rim of an object and
small anywhere the surface is merely smooth, whatever level that smooth
surface happens to sit at, and thresholding it gives a set of walls. What
falls out is the picture a person sees: the frame is divided into patches by
those walls, and each patch is a candidate object.

Measured on a yeast scan, with the top half of one object shifted upwards by
up to four times the object's own height range: this detector kept 94-98 %
of it every time, while an Otsu threshold on the blurred heights kept 49 % -
it lost precisely the half that had been moved. That is the failure this
exists to remove.

Two filters then decide which patches are objects.

*Size* is the obvious one - `min_size` and `max_size`, each as the side of a
square.

*Smoothness* is the useful one. A patch that is smooth inside is a thing;
the field it sits on is textured, and its texture is what broke it into
patches in the first place. The test is the median edge strength inside a
patch against the median over the whole frame, so it is a comparison of the
image with itself and carries no absolute number. On the yeast scan the
objects came out at 0.5-0.6 and the field at 1.1-1.3, which is a wide enough
gap that the setting is not delicate. `smoothness = 0` turns the test off
and keeps every patch that passes the size filter.

The one thing to know about it: the comparison is against the median over
the whole frame, so where the field covers most of the frame the field
*is* the median and reads about 1.0. That is why the default sits at 0.8
rather than at 1.0 - checked both ways round, on a frame where the objects
cover 55 % and on one where they cover 7 %, 0.7-0.8 was the setting that
worked for both.

Finally the walls themselves are shared out. A wall is the rim of the object
it bounds and belongs to it, so the patches are grown back into the wall by
a watershed run on the edge strength, with the rejected space seeded as
background so the flood stops instead of eating the frame. Where two objects
share a rim, the flood meets in the middle of it and each gets its half.


Ridges: for the things that have no inside
------------------------------------------

A thin line - a filament, a fold, the edge of a film - has no interior for
the outline detector to find, and it cannot be found by height either: it is
often no taller than the texture around it, so any threshold that catches it
catches the texture too. What separates them is curvature. A line is curved
sharply across itself and not at all along itself; a grain is curved both
ways. That is the difference the Hessian sees, and this detector thresholds
the more negative of its two eigenvalues instead of the height.

`ridge_width` matters more than the threshold does: at half the true width
the response fills with texture, at twice it neighbouring lines smear
together. Then `ridge_length` finishes the job, measuring every marked
region along its own principal axis and dropping the ones that do not run
far enough. A grain that survived the curvature test is a few tens of
nanometres across and goes; a line runs for a micron and stays.


Raised areas: the local comparison
----------------------------------

The one detector that does look at height, but only ever at a *local*
difference: heights minus a blurred copy of themselves, thresholded in
robust sigmas of that residual. Because the comparison is with the
immediate surroundings and not with a level, it survives a tilted or uneven
field, which a plain threshold does not. It is the right tool for small
things scattered on a surface.


Scales are fractions of the frame, not pixels
---------------------------------------------

Every size here is a percentage of the frame width, and the conversion to
pixels is done per axis. Two reasons. A scan is often not square in pixels -
the yeast scans are 1024 x 512 over 7 x 7 um, so a pixel is twice as tall as
it is wide - and a filter given one radius in pixels would then be measuring
something different along x than along y. And defaults expressed in pixels
stop meaning anything when the scan size changes, while "one percent of the
frame" survives it. The GUI shows what each percentage currently works out
to in nanometres, so nothing is hidden.


What "keep" means
-----------------

`Segmentation` holds the label map and two flags per region: whether it
still exists, and whether it is being kept. The mask it gives back is the
union of the kept regions, plus whatever was painted in by hand, minus
whatever was painted out. Painting is held separately from the regions on
purpose: re-running the detector with a different setting replaces the
labels and leaves the hand corrections in place, which is the order the work
actually happens in - detect, look, fix, adjust, look again.

Dropping a region and erasing it are different, and both are wanted.
Dropping says "found it, do not want it" and leaves it on the screen greyed
out, so a second look can change its mind. Erasing says "that is not a
thing" and takes it out of the map, which is what the box tools do over an
area that came out as rubbish.

`segment` can also be pointed at one rectangle, which is how the editor
searches inside a box the user has drawn. Everything is measured inside that
rectangle, so the thresholds adapt to it - a search over a corner of the
frame is not held to the whole frame's statistics.

`flatten` then replaces every pixel outside the mask with a background
surface estimated from those same outside pixels, so the field keeps its
large-scale shape and loses its texture. `alpha` instead returns an opacity
per pixel and touches no heights at all. Neither one modifies the input.
"""

import numpy as np
from scipy import ndimage


#: Every length is a percentage of the frame width; every level is in robust
#: standard deviations of whatever it is thresholding.
DEFAULTS = {
    # Outlines - shape, no height. See the module docstring.
    'detail': 0.4,        # scale the edges are measured at: the finest wall
    'edge_level': 1.0,    # how strong an edge has to be to count as a wall
    'close_gaps': 0.5,    # how wide a break in a wall may be and still bridge
    'smoothness': 0.8,    # how smooth a patch must be, against the frame's own

    # Ridges - thin continuous crests.
    'ridge_width': 0.9,   # the setting that matters most here
    'ridge_level': 3.0,   # threshold on the curvature response
    'ridge_length': 6.0,  # shortest run that still counts

    # Raised areas - above the immediate surroundings.
    'rise_window': 1.7,   # the window the local background is measured over
    'rise_level': 2.5,    # how far above it a region has to stand

    # Applied to everything found, whichever detector found it.
    'min_size': 0.3,      # smallest region kept, as the side of a square
    'max_size': 0.0,      # largest; 0 means no limit
    'separate': 14.0,     # cut regions apart at a neck; 0 turns it off
}

#: The detectors, in the order they resolve. Later ones win the pixels they
#: share, which is why ridges come last: a filament running over the shoulder
#: of a larger object should read as the filament.
METHODS = ('outline', 'rise', 'ridge')

#: Only used by the GUI, but kept here so the two never drift apart.
METHOD_TITLES = {
    'outline': 'Outlines',
    'rise': 'Raised areas',
    'ridge': 'Ridges',
}

#: Regions are coloured by number, not by type - there are no types. Cycling
#: a short palette is what makes two regions that touch read as two. The
#: colours are chosen to survive being drawn over an AFM colour map, which is
#: warm and light nearly everywhere: red and orange overlays vanish into it.
REGION_COLOURS = (
    (60, 230, 120),
    (255, 60, 220),
    (40, 210, 255),
    (255, 205, 40),
    (150, 130, 255),
    (255, 130, 95),
)
DROPPED_COLOUR = (150, 150, 160)
ADDED_COLOUR = (255, 255, 255)
REMOVED_COLOUR = (30, 30, 40)


# ------------------------------------------------------------------ helpers

def robust_sigma(values):
    """Median absolute deviation scaled to a standard deviation.

    Used instead of `std` everywhere a threshold is set, because the thing
    being thresholded for is by definition an outlier and would otherwise
    help set the threshold that is supposed to catch it.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def otsu_threshold(values, bins=256):
    """The split of `values` into two classes with the largest variance
    between them. Same as `gwy_balance.otsu_threshold`, repeated here so this
    module stands on its own."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    below = np.cumsum(counts).astype(float)
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


class Scale(object):
    """Turns a percentage of the frame into pixels, per axis.

    Everything in this module measures in fractions of the frame; the filters
    underneath measure in pixels, and on a scan whose pixels are not square
    the two axes need different numbers. This is the one place that
    conversion happens.
    """

    def __init__(self, shape, x_real=None, y_real=None, frame=None):
        ny, nx = int(shape[0]), int(shape[1])
        self.ny, self.nx = ny, nx
        self.x_real = float(x_real) if x_real else float(nx)
        self.y_real = float(y_real) if y_real else float(ny)
        self.dx = self.x_real / max(1, nx)
        self.dy = self.y_real / max(1, ny)
        # `frame` is what percentages are percentages *of*. A crop keeps its
        # parent's, so that "2 % of the frame" means the same length whether
        # the search is over the whole scan or over a box drawn on it.
        self.frame = float(frame) if frame else max(self.x_real, self.y_real)

    def crop(self, rows, cols):
        """A Scale for a window of this frame, still measured by this frame."""
        return Scale((rows.stop - rows.start, cols.stop - cols.start),
                     (cols.stop - cols.start) * self.dx,
                     (rows.stop - rows.start) * self.dy,
                     frame=self.frame)

    def length(self, percent):
        """A percentage of the frame width, as a physical length."""
        return max(0.0, float(percent)) / 100.0 * self.frame

    def sigma(self, percent):
        """... as a `(row, column)` Gaussian width in pixels."""
        size = self.length(percent)
        return (max(0.5, size / self.dy), max(0.5, size / self.dx))

    def pixels(self, percent):
        """... as a whole number of pixels, at least one."""
        return max(1, int(round(self.length(percent) / min(self.dx, self.dy))))

    def area(self, percent):
        """... as the area of a square with that side."""
        size = self.length(percent)
        return size * size

    @property
    def pixel_area(self):
        return self.dx * self.dy

    def describe(self, percent, unit='m'):
        """`'62 nm'` - what a control is currently asking for, in real units.

        The GUI puts this next to every size control. A percentage of the
        frame is the right thing to store and the wrong thing to read.
        """
        value = self.length(percent)
        if unit != 'm':
            return '%.3g %s' % (value, unit) if unit else '%.3g px' % value
        for size, prefix in ((1e-9, 'nm'), (1e-6, 'um'), (1e-3, 'mm')):
            if value < size * 1000.0:
                return '%.3g %s' % (value / size, prefix)
        return '%.3g m' % value


def _drop_small_area(mask, min_area, pixel_area, structure=None):
    """Discard connected regions smaller than `min_area`."""
    if not mask.any():
        return mask
    labels, count = ndimage.label(mask, structure=structure)
    if not count:
        return mask
    areas = ndimage.sum(mask, labels, np.arange(1, count + 1)) * pixel_area
    keep = 1 + np.flatnonzero(areas >= min_area)
    return np.isin(labels, keep) if keep.size else np.zeros_like(mask)


def _finite(z):
    """A copy with the gaps filled from their nearest neighbour.

    The filters below all smear a NaN across their whole kernel, so one dead
    pixel would blank a region the size of the window. The gaps are filled
    for the detection and put back as gaps afterwards; nothing that comes out
    of here is ever reported as a height.
    """
    z = np.asarray(z, dtype=float)
    bad = ~np.isfinite(z)
    if not bad.any():
        return z, bad
    if bad.all():
        return np.zeros_like(z), bad
    index = ndimage.distance_transform_edt(
        bad, return_distances=False, return_indices=True)
    return z[tuple(index)], bad


# ------------------------------------------------------------------ detectors

def watershed_source():
    """Which flood is in use: `'scikit-image'` or `'scipy'`.

    Worth asking, because the two do not give the same answer - see
    `_watershed`. On the scan this was developed against, scikit-image
    separated eleven objects and the scipy fallback left the largest six of
    them fused into one region of 26.6 um2. The editor says so rather than
    quietly handing over the worse result.
    """
    try:
        from skimage.segmentation import watershed          # noqa: F401
    except ImportError:
        return 'scipy'
    return 'scikit-image'


def _watershed(relief, markers, mask=None):
    """Flood outwards from `markers`, uphill over `relief`, inside `mask`.

    Both places this module cuts one thing away from another are watersheds:
    the outline detector shares a wall out between the objects on either side
    of it, and the separation cuts a neck between two things grown together.

    `scikit-image` does it if it is installed and `scipy` does it if not,
    because the fallback is not as good and the difference is worth having.
    `ndimage.watershed_ift` quantises the relief to 256 grey levels and
    settles ties in raster order; on a map whose deepest basin is far deeper
    than the necks between the shallow ones, every one of those necks
    collapses into a single level and one marker takes the lot. Measured on a
    real scan: ten markers in, one region out covering 33 um2 of a 49 um2
    frame. `skimage.segmentation.watershed` works on the values themselves
    and returned the ten regions the markers asked for, in 0.05 s.

    Either way the answer is `int32` labels, 0 where nothing reached.
    """
    try:
        from skimage.segmentation import watershed
    except ImportError:
        pass
    else:
        return np.asarray(watershed(relief, markers, mask=mask),
                          dtype=np.int32)

    where = np.ones(relief.shape, dtype=bool) if mask is None else mask
    low = float(relief[where].min()) if where.any() else 0.0
    span = float(np.ptp(relief[where])) if where.any() else 0.0
    grey = np.clip(255.0 * (relief - low) / (span or 1.0),
                   0, 255).astype(np.uint8)
    seeds = np.asarray(markers, dtype=np.int16).copy()
    if mask is not None:
        seeds[~mask] = -1
    grown = ndimage.watershed_ift(grey, seeds)
    return np.where(where & (grown > 0), grown, 0).astype(np.int32)


def edge_strength(z, sigma):
    """How fast the surface is changing at every pixel.

    The gradient magnitude of a Gaussian-blurred copy. `sigma` sets the scale
    the edges are looked for at, and the two derivatives are multiplied by
    their own sigmas so that the answer means the same thing at any setting -
    a first derivative of a blurred image shrinks as the scale grows, and
    without the correction a coarser setting would simply always read lower.

    No absolute height enters this. Two identical objects at different levels
    give identical answers, which is the entire point.
    """
    sr, sc = sigma
    gy = ndimage.gaussian_filter(z, sigma, order=(1, 0)) * sr
    gx = ndimage.gaussian_filter(z, sigma, order=(0, 1)) * sc
    return np.hypot(gy, gx)


def ridge_response(z, sigma):
    """How much the surface looks like a ridge at every pixel.

    The Hessian - the matrix of second derivatives - has two eigenvalues at
    each point, the curvature along the two principal directions. On the
    crest of a ridge one of them is strongly negative (the surface falls away
    to both sides) and the other is near zero (it does not fall away along
    the crest). On a grain both are negative; on a slope or a flat both are
    near zero. So the more negative eigenvalue, sign-flipped, is large only
    on ridges, and that is what this returns.

    The derivatives are taken of a Gaussian-blurred copy, which is what makes
    the answer scale-selective: `sigma` should be about the half-width of the
    lines being looked for. The result is multiplied by the two sigmas so
    that responses computed at different widths are comparable.

    Args:
        z (np.ndarray): A 2D image, finite everywhere.
        sigma (tuple): Gaussian width as `(rows, columns)` in pixels.

    Returns:
        np.ndarray: The response, in units of z. Positive on a raised ridge,
        negative in a groove.
    """
    sr, sc = sigma
    zyy = ndimage.gaussian_filter(z, (sr, sc), order=(2, 0))
    zxx = ndimage.gaussian_filter(z, (sr, sc), order=(0, 2))
    zxy = ndimage.gaussian_filter(z, (sr, sc), order=(1, 1))
    half = 0.5 * (zyy - zxx)
    spread = np.sqrt(half * half + zxy * zxy)
    smaller = 0.5 * (zyy + zxx) - spread
    return -smaller * (sr * sc)


def region_spans(labels, count, dx, dy):
    """How far each region runs along its own axis, and across it.

    A region's long direction is the first eigenvector of the covariance of
    its pixel coordinates; the two numbers returned are the extent of the
    region projected onto that direction and onto the one at right angles to
    it, both in physical units.

    An area or a bounding box would be easier and would not answer the
    question. A line and a blob of the same area have the same area; a
    diagonal line and a compact patch have nearly the same bounding box. The
    span along the region's own axis is what "how far does this run" means,
    and it is what the ridge detector filters on.

    Returns:
        np.ndarray: Shape `(count + 1, 2)` of `(length, width)`, indexed by
        label so row 0 is unused.
    """
    out = np.zeros((count + 1, 2))
    if count <= 0:
        return out
    floor = min(dx, dy)
    for label, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        rows, cols = np.nonzero(labels[window] == label)
        if rows.size < 2:
            out[label] = (floor, floor)
            continue
        y = (rows + window[0].start) * dy
        x = (cols + window[1].start) * dx
        y = y - y.mean()
        x = x - x.mean()
        _, vectors = np.linalg.eigh(np.cov(np.vstack([x, y])))
        axis = vectors[:, -1]
        along = x * axis[0] + y * axis[1]
        across = -x * axis[1] + y * axis[0]
        out[label] = (max(along.max() - along.min(), floor),
                      max(across.max() - across.min(), floor))
    return out


def find_outlines(z, scale, detail=DEFAULTS['detail'],
                  edge_level=DEFAULTS['edge_level'],
                  close_gaps=DEFAULTS['close_gaps'],
                  smoothness=DEFAULTS['smoothness'],
                  min_size=DEFAULTS['min_size'],
                  max_size=DEFAULTS['max_size']):
    """Regions walled off by their own outline. Height plays no part.

    Walls are where the surface changes fastest; the patches between them are
    the candidates; size and smoothness say which patches are objects; and a
    watershed then hands each object the half of its wall that faces it. The
    module docstring explains why this is the detector to reach for when an
    object's parts sit at different heights.

    Args:
        z (np.ndarray): A 2D image with no gaps.
        scale (Scale): The frame this image covers.
        detail (float): The scale edges are measured at, as a percentage of
            the frame. Roughly the width of the thinnest wall worth seeing.
            Larger ignores fine texture and rounds off corners.
        edge_level (float): How far above the frame's typical edge strength a
            wall has to be, in robust sigmas. Lower walls everything off.
        close_gaps (float): Breaks in a wall narrower than this are bridged,
            so an outline that the threshold nicked still encloses. As a
            percentage of the frame.
        smoothness (float): A patch is an object if its own edge strength is
            below this multiple of the frame's median. 0 turns the test off.
        min_size, max_size (float): The size window, each as the side of a
            square in percent of the frame. `max_size = 0` means no limit.

    Returns `labels`, not a mask, and that matters: the walls have already
    told each object apart from the one next to it, so numbering them is free
    and asking a later step to rediscover the boundary would be both slower
    and worse. Two objects sharing a rim come back as two regions here even
    though their pixels touch.

    Returns:
        tuple: `(labels, count, strength)` - the numbered regions, how many,
        and the edge strength they came from, which the caller can show when
        a result needs explaining.
    """
    nothing = np.zeros(z.shape, dtype=np.int32)
    strength = edge_strength(z, scale.sigma(detail))
    floor = float(np.median(strength))
    spread = robust_sigma(strength)
    if spread <= 0.0:
        return nothing, 0, strength

    walls = strength > floor + edge_level * spread
    if close_gaps > 0.0:
        walls = ndimage.binary_closing(walls, disc(scale, close_gaps))

    # 4-connectivity, so a wall one pixel thick stays a wall: with diagonals
    # allowed the patches on either side would leak through its corners.
    patches, count = ndimage.label(~walls)
    if not count:
        return nothing, 0, strength

    index = np.arange(1, count + 1)
    areas = ndimage.sum(~walls, patches, index) * scale.pixel_area
    ok = areas >= scale.area(min_size)
    if max_size > 0.0:
        ok &= areas <= scale.area(max_size)
    if smoothness > 0.0 and floor > 0.0:
        inside = np.asarray(ndimage.median(strength, patches, index))
        ok &= inside <= floor * smoothness
    chosen = 1 + np.flatnonzero(ok)
    if not chosen.size:
        return nothing, 0, strength

    # Give each object back the wall that is its own rim. Flooding uphill
    # from the patches over the edge strength puts the boundary on the crest
    # of the wall, so two objects that share a rim get half each. The
    # rejected space is seeded too, or the flood would run off across the
    # whole frame instead of stopping at the far side of the wall.
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[chosen] = np.arange(1, chosen.size + 1, dtype=np.int32)
    seeds = lookup[patches]
    rest = (~walls) & (seeds == 0)
    if rest.any():
        seeds[ndimage.binary_erosion(rest, np.ones((3, 3)))] = chosen.size + 1
    grown = _watershed(strength, seeds)
    labels = np.where((grown > 0) & (grown <= chosen.size), grown, 0)
    return labels.astype(np.int32), int(chosen.size), strength


def find_ridges(z, scale, outside, ridge_width=DEFAULTS['ridge_width'],
                ridge_level=DEFAULTS['ridge_level'],
                ridge_length=DEFAULTS['ridge_length']):
    """The thin continuous crests: shape first, then continuity.

    Args:
        z (np.ndarray): A 2D image with no gaps - the residual after the
            local background is taken out, not the raw heights.
        scale (Scale): The frame this image covers.
        outside (np.ndarray): Where the search is allowed.
        ridge_width (float): The width of a line, as a percentage of the
            frame. The single most important setting here.
        ridge_level (float): Threshold on the response, in robust sigmas of
            that response measured over `outside`.
        ridge_length (float): How far a region has to run to count, as a
            percentage of the frame.

    Returns:
        tuple: `(mask, response)`.
    """
    response = ridge_response(z, scale.sigma(ridge_width))
    reference = response[outside] if outside.any() else response
    level = robust_sigma(reference)
    if level <= 0.0:
        return np.zeros(z.shape, dtype=bool), response

    mask = (response > ridge_level * level) & outside
    # Close first: a line that the threshold nicked in two would be measured
    # as two short pieces and thrown away as texture.
    mask = ndimage.binary_closing(mask, np.ones((3, 3)))
    labels, count = ndimage.label(mask, np.ones((3, 3)))
    if not count:
        return np.zeros(z.shape, dtype=bool), response

    spans = region_spans(labels, count, scale.dx, scale.dy)
    keep = 1 + np.flatnonzero(spans[1:, 0] >= scale.length(ridge_length))
    mask = np.isin(labels, keep) if keep.size else np.zeros(z.shape, dtype=bool)
    return mask, response


def find_raised(residual, scale, outside,
                rise_level=DEFAULTS['rise_level'],
                min_size=DEFAULTS['min_size'],
                max_size=DEFAULTS['max_size']):
    """Whatever stands above its own immediate surroundings.

    Args:
        residual (np.ndarray): Heights with the local background taken out.
        scale (Scale): The frame this image covers.
        outside (np.ndarray): Where the search is allowed.
        rise_level (float): How far above the background a region has to
            stand, in robust sigmas of the residual. Lower catches more.
        min_size, max_size (float): The size window, each as the side of a
            square in percent of the frame. `max_size = 0` means no limit.

    Returns:
        np.ndarray: A boolean mask.
    """
    spread = robust_sigma(residual[outside] if outside.any() else residual)
    if spread <= 0.0:
        return np.zeros(residual.shape, dtype=bool)

    mask = (residual > rise_level * spread) & outside
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask, np.ones((3, 3)))
    if not count:
        return np.zeros(residual.shape, dtype=bool)

    areas = ndimage.sum(mask, labels, np.arange(1, count + 1)) * scale.pixel_area
    inside = areas >= scale.area(min_size)
    if max_size > 0.0:
        inside &= areas <= scale.area(max_size)
    keep = 1 + np.flatnonzero(inside)
    return np.isin(labels, keep) if keep.size else np.zeros(residual.shape, bool)


def separate_regions(labels, count, scale, separation):
    """Cut apart any region that is really two things grown together.

    Done one region at a time rather than over the whole map at once, and
    that is not an implementation detail. The split runs a watershed on a
    distance transform quantised to 256 levels; over a whole frame the
    deepest basin anywhere sets that scale, so on a map holding one large
    region and a hundred small ones every neck between the small ones
    collapses into a single level and the flood hands one basin everything.
    Measured on a real scan: ten seeds found, one region returned. Per region
    the depth scale is the region's own, and the necks stay resolved.

    Skipping the regions too small to hold two seeds also makes it cheap -
    on a map of a thousand specks only a handful are looked at.

    Returns:
        tuple: `(labels, count)`, with any new pieces numbered after the
        existing ones so nothing already on the map is renumbered.
    """
    if separation <= 0.0 or count <= 0:
        return labels, count
    out = labels.copy()
    total = count
    floor = 2.0 * scale.area(separation)
    for label, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        piece = out[window] == label
        if int(piece.sum()) * scale.pixel_area < floor:
            continue
        sub, pieces = split_touching(piece, scale, separation)
        if pieces <= 1:
            continue
        block = out[window]
        for k in range(2, pieces + 1):
            total += 1
            block[piece & (sub == k)] = total
    return out, total


def split_touching(mask, scale, separation):
    """Number a mask, cutting it where two round things have grown together.

    Objects packed on a surface touch, and no threshold can see the seam:
    plain connected-component labelling then returns the whole raft as one
    region, and a user who wants to keep one object has to keep all of them.
    On the yeast scans this was the difference between one region and ten.

    The split is made on shape. The distance to the nearest edge peaks in the
    middle of each object and collapses at the neck between two, so the seeds
    are the local maxima of that distance and a watershed run downhill from
    them puts the boundary exactly at the neck.

    `separation` is how close two seeds may be, as a percentage of the frame
    - roughly the width of the smallest thing that should come out on its
    own. A plain threshold on the distance was tried first and was not
    usable: on the real scans the count it gave went 4, 3, 3, 7, 2, 1 as the
    threshold rose, because an object's seed disappears entirely once the
    threshold passes its own centre. Local maxima have no such cliff - the
    same scans gave a stable count for every `separation` from 8 % of the
    frame to 25 %.

    Falls back to plain labelling whenever the split finds one seed or none,
    which is the right answer for a mask that is already one object.

    Returns:
        tuple: `(labels, count)`, as `ndimage.label` would give.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0
    distance = ndimage.distance_transform_edt(mask, sampling=(scale.dy, scale.dx))
    peak = distance.max()
    if separation <= 0.0 or peak <= 0.0:
        return ndimage.label(mask, np.ones((3, 3)))

    # Smooth first: a raw distance transform is faceted, and a facet edge is
    # a local maximum that means nothing.
    #
    # The neighbourhood is a rectangle rather than a disc because a rectangle
    # is separable and a disc is not: on a 512 x 1024 scan the round
    # footprint took 2.2 s and the rectangle 0.00 s, and both found the same
    # seeds. Seed spacing is not a shape that needs to be round.
    window = (max(1, int(round(scale.length(separation) / scale.dy))),
              max(1, int(round(scale.length(separation) / scale.dx))))
    smooth = ndimage.gaussian_filter(distance, scale.sigma(separation / 8.0))
    seeds = ((smooth >= ndimage.maximum_filter(smooth, size=window))
             & (smooth > 0.25 * scale.length(separation)))
    markers, count = ndimage.label(seeds)
    if count <= 1:
        return ndimage.label(mask, np.ones((3, 3)))

    # The flood runs outwards from the markers, so the distance is negated:
    # the middle of an object is the bottom of its own basin, and the neck
    # between two is the ridge the two floods meet on. It is the *smoothed*
    # distance, the one the seeds were found on, so the boundary lands where
    # the seeds said the neck was rather than on a facet of the raw one.
    grown = _watershed(-smooth, markers, mask)
    if not grown.any():
        return ndimage.label(mask, np.ones((3, 3)))
    # Renumber, because a marker can lose every pixel it started with.
    used = np.unique(grown)
    used = used[used > 0]
    lookup = np.zeros(int(grown.max()) + 1, dtype=np.int32)
    lookup[used] = np.arange(1, used.size + 1, dtype=np.int32)
    out, count = lookup[grown], int(used.size)

    # The flood leaves its own watershed lines unclaimed, and a piece too
    # small to hold a seed reaches no basin at all. Both would otherwise
    # disappear here - the mask would come back with less in it than it went
    # in with, which is not something a *numbering* step may do.
    #
    # What was left over next to a basin belongs to it, and is given to
    # whichever basin is nearest, so a watershed line is shared out the same
    # way a rim is. What was left over on its own is a small object that had
    # no seed, and gets a number of its own instead of being swallowed by a
    # region it does not touch.
    missed = mask & (out == 0)
    if missed.any():
        bits, pieces = ndimage.label(missed, np.ones((3, 3)))
        beside = ndimage.grey_dilation(out, size=(3, 3))
        adjoins = np.asarray(ndimage.maximum(beside, bits,
                                             np.arange(1, pieces + 1)))
        joined = np.isin(bits, 1 + np.flatnonzero(adjoins > 0))
        if joined.any():
            index = ndimage.distance_transform_edt(
                out == 0, return_distances=False, return_indices=True)
            out = np.where(joined, out[tuple(index)], out)
        alone = missed & ~joined
        if alone.any():
            extra, more = ndimage.label(alone, np.ones((3, 3)))
            out = np.where(alone, extra + count, out)
            count += more
    return out, count


# ------------------------------------------------------------------ the result

class Segmentation(object):
    """A label map, and per region whether it exists and whether it is kept.

    Region numbering starts at 1; 0 is everything no detector claimed, and is
    never kept by a flag - only by the brush.

    Two flags per region, because "I do not want this one" and "that is not a
    region at all" are different statements. `keep` is the first: the region
    stays on the map, drawn greyed out, and one more click brings it back.
    `alive` is the second: erasing takes the region out of the map for good,
    which is what the box tools do over an area that came out as rubbish.

    The brush lives in `added` and `removed` rather than in the label map.
    That way re-running the detector - which is the normal thing to do after
    looking at the first attempt - replaces the regions and leaves the hand
    corrections alone.
    """

    def __init__(self, labels, scale, params=None, count=None):
        self.labels = np.asarray(labels, dtype=np.int32)
        self.scale = scale
        self.params = dict(params or {})
        self.count = int(self.labels.max()) if count is None else int(count)

        self.keep = np.zeros(self.count + 1, dtype=bool)
        self.keep[1:] = True
        self.alive = np.zeros(self.count + 1, dtype=bool)
        self.alive[1:] = True
        self.added = np.zeros(self.labels.shape, dtype=bool)
        self.removed = np.zeros(self.labels.shape, dtype=bool)
        self._measured = None

    # ---- what is where ----

    @property
    def shape(self):
        return self.labels.shape

    @property
    def region_count(self):
        """How many regions are still on the map."""
        return int(self.alive[1:].sum())

    @property
    def kept_count(self):
        return int((self.alive[1:] & self.keep[1:]).sum())

    @property
    def painted(self):
        return bool(self.added.any() or self.removed.any())

    def label_at(self, row, col):
        """The region under a pixel, or 0. Out of range gives 0."""
        row, col = int(row), int(col)
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            return 0
        return int(self.labels[row, col])

    def labels_in(self, window):
        """Every region with a pixel inside a `(rows, cols)` slice pair."""
        inside = np.unique(self.labels[window])
        return inside[inside > 0]

    # ---- what is kept ----

    def mask(self):
        """The pixels that survive: kept regions, plus the brush."""
        out = self.keep[self.labels]
        if self.added.any():
            out = out | self.added
        if self.removed.any():
            out = out & ~self.removed
        return out

    def set_kept(self, label, kept):
        label = int(label)
        if 1 <= label <= self.count and self.alive[label]:
            self.keep[label] = bool(kept)

    def toggle(self, label):
        """Flip one region. Returns its new state, or None if there is none."""
        label = int(label)
        if not (1 <= label <= self.count) or not self.alive[label]:
            return None
        self.keep[label] = not self.keep[label]
        return bool(self.keep[label])

    def keep_all(self, kept=True):
        self.keep[1:] = np.where(self.alive[1:], bool(kept), False)

    def invert(self):
        self.keep[1:] = ~self.keep[1:] & self.alive[1:]

    # ---- taking regions off the map ----

    def erase(self, labels):
        """Remove regions entirely. Returns how many went.

        Their pixels go back to belonging to nothing, so the detector's
        opinion about them is gone rather than merely overruled.
        """
        labels = np.atleast_1d(np.asarray(labels, dtype=np.int64)).ravel()
        labels = labels[(labels > 0) & (labels <= self.count)]
        labels = labels[self.alive[labels]]
        if not labels.size:
            return 0
        self.labels[np.isin(self.labels, labels)] = 0
        self.keep[labels] = False
        self.alive[labels] = False
        self._measured = None
        return int(labels.size)

    def erase_where(self, where):
        """Take a patch of pixels off the map, and unpaint it.

        Regions that lose every pixel die; one that only partly overlaps is
        trimmed and lives on. This is what an erase over an area means: the
        area is now unclaimed, whatever it was claimed by.
        """
        where = np.asarray(where, dtype=bool)
        if not where.any():
            return 0
        touched = np.unique(self.labels[where])
        touched = touched[touched > 0]
        self.labels[where] = 0
        self.added[where] = False
        self.removed[where] = False
        if touched.size:
            left = np.unique(self.labels)
            gone = touched[~np.isin(touched, left)]
            if gone.size:
                self.keep[gone] = False
                self.alive[gone] = False
        self._measured = None
        return int(touched.size)

    def absorb(self, other, window=None):
        """Replace the regions inside `window` with the ones `other` found.

        This is what searching inside a drawn box does. Whatever the previous
        pass thought was in the box is dropped first, so a second search over
        the same place replaces its own last answer instead of piling a new
        set of regions on top of it. A region that only reaches into the box
        is trimmed rather than removed - the box is a statement about the
        area, not about everything the area touches.
        """
        box = np.zeros(self.shape, dtype=bool)
        if window is None:
            box[:] = True
        else:
            box[window] = True
        self.erase_where(box)

        new = np.where(box, other.labels, 0)
        ids = np.unique(new)
        ids = ids[ids > 0]
        if not ids.size:
            return 0
        lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
        lookup[ids] = self.count + 1 + np.arange(ids.size, dtype=np.int32)
        self.labels = np.where(new > 0, lookup[new], self.labels)
        self.keep = np.concatenate([self.keep, np.ones(ids.size, dtype=bool)])
        self.alive = np.concatenate([self.alive, np.ones(ids.size, dtype=bool)])
        self.count += int(ids.size)
        self._measured = None
        return int(ids.size)

    def clear_painting(self):
        self.added[:] = False
        self.removed[:] = False

    def paint(self, where, adding=True):
        """Force pixels in or out by hand.

        The two arrays are kept exclusive, so painting something in and then
        out again leaves no trace rather than leaving it in both.
        """
        where = np.asarray(where, dtype=bool)
        if adding:
            self.added |= where
            self.removed &= ~where
        else:
            self.removed |= where
            self.added &= ~where

    # ---- numbers about each region ----

    def measure(self, z=None):
        """Per-region `(area, length, width, height)`, indexed by label.

        Cached, because the spans need a pass over each region and the status
        line asks for them on every mouse move. `z` is only needed for the
        height and only on the first call.
        """
        if self._measured is not None:
            return self._measured
        scale = self.scale
        spans = region_spans(self.labels, self.count, scale.dx, scale.dy)
        if self.count:
            areas = ndimage.sum(self.labels > 0, self.labels,
                                np.arange(1, self.count + 1)) * scale.pixel_area
        else:
            areas = np.zeros(0)
        heights = np.full(self.count + 1, np.nan)
        if z is not None and self.count:
            finite, _ = _finite(z)
            floor = np.median(finite[self.labels == 0]) if (self.labels == 0).any() \
                else np.median(finite)
            peaks = ndimage.maximum(finite, self.labels,
                                    np.arange(1, self.count + 1))
            heights[1:] = np.asarray(peaks) - floor

        rows = [None]
        for label in range(1, self.count + 1):
            rows.append(dict(label=label,
                             area=float(areas[label - 1]),
                             length=float(spans[label, 0]),
                             width=float(spans[label, 1]),
                             height=float(heights[label])))
        self._measured = rows
        return rows


def segment(z, x_real=None, y_real=None, methods=METHODS, window=None,
            **params):
    """Find the objects in one height map.

    Args:
        z (np.ndarray): A 2D height map. NaNs are allowed and are filled from
            their neighbours for the detection only.
        x_real, y_real (float): The physical size of the frame. Without them
            everything is measured in pixels, which still works - the
            percentages are then percentages of the pixel count.
        methods (tuple): Which detectors to run, from `METHODS`. Dropping one
            is cheaper than running it and erasing the result.
        window (tuple): `(row0, row1, col0, col1)` to search only inside a
            rectangle. Every threshold is then measured inside it, so a
            search over a corner is judged by that corner and not by the
            whole frame. The result still has the full image's shape, with
            nothing outside the rectangle.
        **params: Anything in `DEFAULTS`.

    Returns:
        Segmentation: The label map, with every region kept to start with.
    """
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError('unknown segmentation setting %s - have %s'
                        % (', '.join(sorted(unknown)), ', '.join(sorted(DEFAULTS))))
    settings = dict(DEFAULTS)
    settings.update(params)

    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError('segmentation needs a 2D image, got shape %r'
                         % (z.shape,))
    scale = Scale(z.shape, x_real, y_real)
    if window is None:
        return _segment_frame(z, scale, methods, settings)

    rows, cols, inner = _window_slices(z.shape, scale, settings, window)
    part = _segment_frame(z[rows, cols], scale.crop(rows, cols),
                          methods, settings)
    # The filters were given a margin of context beyond the box so they would
    # not see an artificial edge at its border; the answer is only claimed
    # inside the box itself. Renumber afterwards, or the regions that lived
    # entirely in the margin would still be counted while owning no pixels.
    kept = np.where(inner, part.labels, 0)
    ids = np.unique(kept)
    ids = ids[ids > 0]
    labels = np.zeros(z.shape, dtype=np.int32)
    if ids.size:
        lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
        lookup[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
        labels[rows, cols] = lookup[kept]
    return Segmentation(labels, scale, settings, count=int(ids.size))


def _window_slices(shape, scale, settings, window):
    """The slices to work on, and which of their pixels the box really owns."""
    ny, nx = shape
    r0, r1, c0, c1 = (int(round(v)) for v in window)
    r0, r1 = max(0, min(r0, r1)), min(ny, max(r0, r1) + 1)
    c0, c1 = max(0, min(c0, c1)), min(nx, max(c0, c1) + 1)
    r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)

    widest = max(settings['detail'], settings['ridge_width'],
                 settings['rise_window'], settings['close_gaps'])
    margin = 3 * scale.pixels(widest)
    rows = slice(max(0, r0 - margin), min(ny, r1 + margin))
    cols = slice(max(0, c0 - margin), min(nx, c1 + margin))

    inner = np.zeros((rows.stop - rows.start, cols.stop - cols.start), dtype=bool)
    inner[r0 - rows.start:r1 - rows.start, c0 - cols.start:c1 - cols.start] = True
    return rows, cols, inner


def _segment_frame(z, scale, methods, settings):
    """The pipeline, over whatever array it is handed."""
    filled, gaps = _finite(z)
    shape = z.shape
    blank = (np.zeros(shape, dtype=np.int32), 0)
    nothing = np.zeros(shape, dtype=bool)
    separation = settings['separate']

    if 'outline' in methods:
        found, count, _ = find_outlines(
            filled, scale, settings['detail'], settings['edge_level'],
            settings['close_gaps'], settings['smoothness'],
            settings['min_size'], settings['max_size'])
        # The walls have already numbered these; the separation is only asked
        # to look for two things inside one patch.
        outlines = separate_regions(found, count, scale, separation)
    else:
        outlines = blank

    # The rim of an object is a steep wall and the ridge filter loves it, so
    # the search for the thin things starts a little way clear of the objects
    # already found.
    margin = scale.pixels(0.4)
    outside = (~ndimage.binary_dilation(outlines[0] > 0, iterations=margin)
               & ~gaps)

    residual = filled - ndimage.gaussian_filter(
        filled, scale.sigma(settings['rise_window']))

    ridge_mask = (find_ridges(residual, scale, outside, settings['ridge_width'],
                              settings['ridge_level'],
                              settings['ridge_length'])[0]
                  if 'ridge' in methods else nothing)
    # No separation for ridges: a line's distance transform peaks all the way
    # along it, so the split would shatter one line into a row of fragments.
    ridges = ndimage.label(ridge_mask, np.ones((3, 3)))

    if 'rise' in methods:
        raised = find_raised(residual, scale, outside & ~ridge_mask,
                             settings['rise_level'], settings['min_size'],
                             settings['max_size'])
        rises = separate_regions(*ndimage.label(raised, np.ones((3, 3))),
                                 scale=scale, separation=separation)
    else:
        rises = blank

    seg = _label_up({'outline': outlines, 'rise': rises, 'ridge': ridges},
                    scale, settings)
    # A pixel with no measurement belongs to nothing. The regions were found
    # on the filled copy so their outlines are still whole; they simply do
    # not claim the holes.
    if gaps.any():
        seg.labels[gaps] = 0
        _prune(seg)
    return seg


def _label_up(found, scale, settings):
    """Stack the detectors' answers into one map, later detectors winning."""
    shape = next(iter(found.values()))[0].shape
    labels = np.zeros(shape, dtype=np.int32)
    total = 0
    for method in METHODS:
        piece, count = found.get(method, (None, 0))
        if not count:
            continue
        piece = np.where(piece > 0, piece + total, 0)
        labels = np.where(piece > 0, piece, labels)
        total += count
    seg = Segmentation(labels, scale, settings, count=total)
    _prune(seg)
    return seg


def _prune(seg):
    """Renumber so every region on the map owns at least one pixel.

    A later detector can bury an earlier one completely, and a hole in the
    data can swallow a small region whole. Either way what is left is a
    number in the table with nothing under it, which would be counted in the
    summary and would never light up on the screen.
    """
    ids = np.unique(seg.labels)
    ids = ids[ids > 0]
    if ids.size == seg.count:
        return seg
    lookup = np.zeros(max(int(seg.labels.max()), seg.count) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    seg.labels = lookup[seg.labels]
    seg.count = int(ids.size)
    seg.keep = np.zeros(seg.count + 1, dtype=bool)
    seg.keep[1:] = True
    seg.alive = np.zeros(seg.count + 1, dtype=bool)
    seg.alive[1:] = True
    seg._measured = None
    return seg


def empty(z, x_real=None, y_real=None):
    """A segmentation with no regions - the starting point for painting."""
    z = np.asarray(z)
    scale = Scale(z.shape, x_real, y_real)
    return Segmentation(np.zeros(z.shape, dtype=np.int32), scale, {}, count=0)


# ------------------------------------------------------------------ applying

def background(z, keep, scale, smooth=DEFAULTS['rise_window'] * 2.0):
    """The surface the discarded parts get flattened onto.

    Estimated from the pixels that are *not* being kept, which is the whole
    point: a background fitted through the objects would be pulled up by
    them. The kept regions are filled in from their nearest outside
    neighbour before the blur, so the estimate does not dip where an object
    used to be, and the blur then removes the field's own texture while
    leaving its large-scale shape alone.

    `smooth = 0` gives a single level - the median of everything outside -
    which is the right answer when the field is genuinely flat and the honest
    one when it is not obvious that it is.
    """
    z = np.asarray(z, dtype=float)
    filled, gaps = _finite(z)
    outside = ~np.asarray(keep, dtype=bool) & ~gaps
    if not outside.any():
        return np.full(z.shape, float(np.nanmedian(filled)))
    if smooth <= 0.0:
        return np.full(z.shape, float(np.median(filled[outside])))

    index = ndimage.distance_transform_edt(
        ~outside, sampling=(scale.dy, scale.dx),
        return_distances=False, return_indices=True)
    return ndimage.gaussian_filter(filled[tuple(index)], scale.sigma(smooth))


def flatten(z, keep, scale, smooth=DEFAULTS['rise_window'] * 2.0):
    """`z` with everything outside `keep` replaced by the background.

    The kept pixels are untouched - not smoothed, not re-levelled, not
    shifted. What was measured on an object is still what is drawn on it.

    A pixel that was never measured stays unmeasured. Filling a NaN with the
    background would turn a hole in the data into a plausible-looking flat
    patch, which is the one thing a tool that removes things must not do.
    """
    z = np.asarray(z, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    out = np.where(keep, z, background(z, keep, scale, smooth))
    gaps = ~np.isfinite(z)
    if gaps.any():
        out[gaps] = np.nan
    return out


def alpha(keep, rest=0.0, edge=0.0, scale=None):
    """An opacity per pixel: 1 on the kept parts, `rest` elsewhere.

    `edge` softens the boundary over that percentage of the frame, which
    stops the cut looking like a cookie cutter when the discarded part is
    only faded rather than removed. It costs a blur, so it is off by default.
    """
    keep = np.asarray(keep, dtype=bool)
    out = np.where(keep, 1.0, float(rest))
    if edge > 0.0 and scale is not None:
        out = ndimage.gaussian_filter(out, scale.sigma(edge))
        out = np.maximum(out, np.where(keep, 1.0, 0.0))
    return out


def cut_out(z, keep):
    """`z` with everything outside `keep` set to NaN.

    For export rather than for the screen: a NaN is a hole in the data, and
    every reader downstream knows what to do with it, whereas a zero or a
    background level is a made-up height that looks measured.
    """
    z = np.asarray(z, dtype=float)
    out = z.copy()
    out[~np.asarray(keep, dtype=bool)] = np.nan
    return out


def disc(scale, diameter_percent):
    """A round brush of a given size, as a boolean footprint.

    Round in *physical* units, so on a scan whose pixels are not square the
    brush is a circle on the sample rather than a circle in the array.
    """
    radius = 0.5 * scale.length(diameter_percent)
    ry = max(0, int(round(radius / scale.dy)))
    rx = max(0, int(round(radius / scale.dx)))
    yy = np.arange(-ry, ry + 1)[:, None] * scale.dy
    xx = np.arange(-rx, rx + 1)[None, :] * scale.dx
    return (yy * yy + xx * xx) <= radius * radius + 1e-30


def stamp(target, footprint, row, col, value=True):
    """Paint `footprint` into `target` centred on a pixel, clipped at the edges."""
    fh, fw = footprint.shape
    r0, c0 = int(row) - fh // 2, int(col) - fw // 2
    r1, c1 = r0 + fh, c0 + fw
    tr0, tc0 = max(0, r0), max(0, c0)
    tr1, tc1 = min(target.shape[0], r1), min(target.shape[1], c1)
    if tr0 >= tr1 or tc0 >= tc1:
        return target
    piece = footprint[tr0 - r0:tr1 - r0, tc0 - c0:tc1 - c0]
    view = target[tr0:tr1, tc0:tc1]
    if value:
        view |= piece
    else:
        view &= ~piece
    return target
