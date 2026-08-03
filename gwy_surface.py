"""
The height map a 3D view draws, and the numbers it needs to draw it honestly.

This module knows nothing about windows, widgets or renderers. It holds one
AFM channel - the Z value of every pixel, plus how wide and how tall the
frame really is - and turns it into a mesh. Both the interactive viewer
(`gwy_3d_viewer`) and the Blender exporter (`gwy_blender_render`) start from
the same object here, so what is seen on screen and what comes out of a
final render are the same surface and not two hand-built approximations of
it.

Two things about AFM data force decisions that a general 3D library will not
make for you:

*Scale.* An AFM frame is microns across and nanometres tall. Drawn to true
proportion the surface is a flat sheet: a 10 um scan with 20 nm of relief has
an aspect of 1:500, and the topography that the whole measurement is about
becomes invisible. Every 3D SPM view therefore exaggerates Z, and the only
question is whether it says so. Here the exaggeration is an explicit number
the user sets, `natural_exaggeration` picks a first value that makes the
relief a quarter of the frame width, and 1.0 always means true physical
proportion - so "how tall does this look" is a choice on the record and not
an accident of the default.

*Precision.* Those same physical numbers - 1e-5 m across, 1e-8 m tall - are
poor coordinates to render with. Depth buffers, near/far planes and the
picking code all work in float32, and a scene whose whole extent is 1e-5
spends its depth precision on nothing. So the mesh is built in *frame units*:
the longer side of the scan is 1.0, and everything else follows. The physical
values are not thrown away, they travel along as the `height` scalar in
whatever SI prefix reads best, which is what the colour map and the scale bar
show. Geometry is normalised, numbers stay real.

The colour range is kept apart from the geometry for the same reason. The
histogram limits change which heights are dark and which are bright; the
exaggeration changes how tall the surface stands. They are independent, they
are stored separately, and moving one never silently moves the other.

`Surface` is deliberately a single 2D field. Stacking several of them into a
volume is a later job, and the shape of this class - one field, its own
extent, its own offset in Z - is what a stack would be built from.
"""

import os

import numpy as np

import gwy_loader


# ---------------------------------------------------------------- units

_PREFIXES = [
    (1e-12, 'p'),
    (1e-9, 'n'),
    (1e-6, 'µ'),
    (1e-3, 'm'),
    (1.0, ''),
    (1e3, 'k'),
]


def nice_units(span, unit):
    """Pick the SI prefix that shows `span` with the fewest zeros.

    Returns ``(factor, label)``: multiply a value in base units by `factor`
    to get a number to print, and `label` is the unit it is then in. A blank
    or unrecognised unit is left alone, so a channel measured in volts or
    degrees is not quietly relabelled.
    """
    if not unit or len(unit) > 3:
        return 1.0, unit or ''
    span = abs(float(span))
    if not np.isfinite(span) or span <= 0.0:
        return 1.0, unit
    chosen = _PREFIXES[0]
    for size, prefix in _PREFIXES:
        if span >= size:
            chosen = (size, prefix)
    size, prefix = chosen
    return 1.0 / size, prefix + unit


# ---------------------------------------------------------------- surface

class Surface(object):
    """One AFM channel: a height map and the physical size of its frame.

    `z` is indexed ``[row, column]`` - row 0 is the first scan line - and is
    held in the units the file used, normally metres. `x_real` and `y_real`
    are the width and height of the frame in the lateral unit. Non-finite
    pixels are allowed and are carried through as NaN; they are drawn flat
    and in the colour map's NaN colour rather than tearing a hole in the
    mesh.
    """

    def __init__(self, z, x_real, y_real,
                 xy_unit='m', z_unit='m', name='', source=None):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError('a surface needs a 2D height map, got shape %r'
                             % (z.shape,))
        if z.shape[0] < 2 or z.shape[1] < 2:
            raise ValueError('a surface needs at least 2x2 pixels, got %r'
                             % (z.shape,))
        self.z = z
        self.x_real = float(x_real) if x_real else float(z.shape[1])
        self.y_real = float(y_real) if y_real else float(z.shape[0])
        self.xy_unit = xy_unit or ''
        self.z_unit = z_unit or ''
        self.name = name or 'surface'
        self.source = source

    # ---- shape ----

    @property
    def shape(self):
        return self.z.shape

    @property
    def ny(self):
        return self.z.shape[0]

    @property
    def nx(self):
        return self.z.shape[1]

    @property
    def frame(self):
        """The longer side of the scan - the length that becomes 1.0."""
        return max(self.x_real, self.y_real)

    def __repr__(self):
        return '<Surface %r %dx%d %.3g x %.3g %s>' % (
            self.name, self.nx, self.ny, self.x_real, self.y_real,
            self.xy_unit)

    # ---- ranges ----

    def full_range(self):
        """Lowest and highest finite value."""
        finite = self.z[np.isfinite(self.z)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def percentile_range(self, low=0.5, high=99.5):
        """The range with the outermost `low`/`high` percent left out.

        A handful of spikes - a piece of debris, a parachuting artifact -
        can own the whole colour range and the whole height of the render.
        Trimming a fraction of a percent off each end costs nothing real and
        keeps them from doing that.
        """
        finite = self.z[np.isfinite(self.z)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(finite, [low, high])
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            return self.full_range()
        return lo, hi

    def histogram(self, bins=256, low=0.0, high=100.0):
        """Counts and bin edges over the finite pixels, for the range widget."""
        finite = self.z[np.isfinite(self.z)]
        if finite.size == 0:
            return np.zeros(bins), np.linspace(0.0, 1.0, bins + 1)
        if low > 0.0 or high < 100.0:
            lo, hi = np.percentile(finite, [low, high])
        else:
            lo, hi = finite.min(), finite.max()
        if not (hi > lo):
            hi = lo + 1.0
        counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
        return counts, edges

    def z_units(self):
        """`(factor, label)` for printing heights - see `nice_units`."""
        lo, hi = self.percentile_range(0.1, 99.9)
        return nice_units(hi - lo, self.z_unit)

    def xy_units(self):
        """`(factor, label)` for printing lateral distances."""
        return nice_units(self.frame, self.xy_unit)

    # ---- size ----

    def subsampled(self, max_points=4_000_000):
        """A coarser copy if the map is too large to render comfortably.

        Returns `self` when there is nothing to do, so it is safe to call
        unconditionally. The step is a plain stride rather than an average:
        an averaged copy would be a different measurement, and this is only
        meant to keep the viewport interactive.
        """
        total = self.nx * self.ny
        if total <= max_points:
            return self
        step = int(np.ceil(np.sqrt(total / float(max_points))))
        z = self.z[::step, ::step]
        return Surface(z, self.x_real, self.y_real,
                       xy_unit=self.xy_unit, z_unit=self.z_unit,
                       name=self.name, source=self.source)


# ---------------------------------------------------------------- loading

def channels(path):
    """The channel titles in a .gwy file, in file order."""
    return list(gwy_loader.get_channels(path))


def _unit_of(field, key):
    unit = field.get(key, None)
    if unit is None:
        return ''
    try:
        return unit.unitstr or ''
    except AttributeError:
        return str(unit)


def from_gwy(path, channel=None):
    """Read one channel of a Gwyddion file as a `Surface`.

    With no `channel` the first one is taken, which for most files is the
    forward height scan.
    """
    fields = gwy_loader.load_gwy(path)
    if not fields:
        raise ValueError('no data channels in %s' % os.path.basename(path))
    if channel is None:
        channel = next(iter(fields))
    if channel not in fields:
        raise KeyError('no channel %r in %s - have %s'
                       % (channel, os.path.basename(path),
                          ', '.join(repr(k) for k in fields)))
    field = fields[channel]
    return Surface(
        np.array(field.data, dtype=np.float64),
        field.xreal, field.yreal,
        xy_unit=_unit_of(field, 'si_unit_xy'),
        z_unit=_unit_of(field, 'si_unit_z'),
        name=channel,
        source=path,
    )


def from_npy(path, x_real=None, y_real=None):
    """Read a bare .npy height map - what the main GUI writes when it exports.

    A .npy carries no physical size, so without `x_real`/`y_real` the frame
    is taken to be the pixel count and the scale bar will read in pixels.
    """
    z = np.load(path)
    return Surface(z, x_real, y_real,
                   xy_unit='' if x_real is None else 'm',
                   z_unit='',
                   name=os.path.splitext(os.path.basename(path))[0],
                   source=path)


def load(path, channel=None):
    """Read a .gwy or .npy file, whichever it is."""
    if path.lower().endswith('.npy'):
        return from_npy(path)
    return from_gwy(path, channel)


# ---------------------------------------------------------------- geometry

DEFAULT_RELIEF = 0.25


def natural_exaggeration(surface, relief=DEFAULT_RELIEF):
    """The Z exaggeration that makes the relief `relief` x the frame width.

    This is the number the viewer opens with. It is not physical - 1.0 is
    physical - it is only a starting point that puts a readable amount of
    structure on screen whatever the scan size, so that a 500 nm scan and a
    50 um scan both arrive looking like a surface instead of like a plate or
    a mountain range.
    """
    lo, hi = surface.percentile_range(0.1, 99.9)
    span = hi - lo
    if span <= 0.0:
        return 1.0
    return float(relief * surface.frame / span)


class SurfaceMesh(object):
    """A `Surface` as a renderable mesh, with the exaggeration left loose.

    The geometry is built once. Changing the exaggeration afterwards moves
    the Z coordinate of the existing points instead of rebuilding, which is
    what makes dragging the height slider smooth on a megapixel scan.

    Coordinates are in frame units: the longer side of the scan is 1.0 and
    the surface is centred on Z = 0. The physical heights ride along as the
    ``height`` point array, in the SI prefix `Surface.z_units` chose, and
    that array - never the geometry - is what the colour map reads.
    """

    def __init__(self, surface, exaggeration=None):
        import pyvista as pv

        self.surface = surface
        self.z_factor, self.z_label = surface.z_units()
        self.xy_factor, self.xy_label = surface.xy_units()

        frame = surface.frame
        nx, ny = surface.nx, surface.ny
        dx = (surface.x_real / (nx - 1)) / frame
        dy = (surface.y_real / (ny - 1)) / frame

        z = surface.z
        self._frame = frame
        self.edited = None          # a height map drawn instead of the real one
        # Centre on the middle of the robust range, not on the mean: one
        # deep pit should not push the whole surface up out of the frame.
        lo, hi = surface.percentile_range(0.1, 99.9)
        self.z_reference = 0.5 * (lo + hi)

        grid = pv.ImageData(dimensions=(nx, ny, 1),
                            spacing=(dx, dy, 1.0),
                            origin=(0.0, 0.0, 0.0))
        grid.point_data['relief'] = self._as_relief(z)
        grid.point_data['height'] = (z.ravel(order='C') * self.z_factor)

        # warp_by_scalar gives a StructuredGrid whose points we can then
        # move directly; the ImageData itself has no explicit points.
        self.mesh = grid.warp_by_scalar('relief', factor=1.0)
        self.mesh.set_active_scalars('height')

        self.exaggeration = 1.0
        self.set_exaggeration(
            natural_exaggeration(surface) if exaggeration is None
            else exaggeration)

    def _as_relief(self, z):
        """Heights as geometry: centred on the reference and in frame units.

        Geometry has to be finite everywhere or the mesh tears, so the gaps
        sit at the reference level. Colour reads the untouched values, where
        the gaps are still NaN and are drawn as such.
        """
        good = np.isfinite(z)
        relief = np.where(good, z - self.z_reference, 0.0) / self._frame
        self._relief = relief.ravel(order='C').astype(np.float64)
        return self._relief

    def set_heights(self, z=None):
        """Draw a different height map on the same grid, or restore the real one.

        This is how an edited surface - one with the background flattened, or
        with regions cut out - is shown without rebuilding the mesh or
        touching the `Surface` it came from.

        The reference level and the unit prefix are deliberately *not*
        recomputed. They were chosen from the measurement, and letting an
        edit move them would make the colour range and the Z axis jump every
        time the edit is switched on and off, which reads as the data
        changing when only the view has.

        Args:
            z (np.ndarray): A height map of the same shape, in the same units
                as the original. `None` puts the original back.
        """
        original = z is None
        z = self.surface.z if original else np.asarray(z, dtype=float)
        if z.shape != self.surface.shape:
            raise ValueError('height map is %r, the surface is %r'
                             % (z.shape, self.surface.shape))
        self.edited = None if original else z
        self.mesh.point_data['relief'] = self._as_relief(z)
        self.mesh.point_data['height'] = z.ravel(order='C') * self.z_factor
        self.mesh.set_active_scalars('height')
        self.set_exaggeration(self.exaggeration)

    @property
    def heights(self):
        """Whatever is currently drawn - the edit if there is one."""
        return self.surface.z if self.edited is None else self.edited

    # ---- the one thing that changes often ----

    def set_exaggeration(self, exaggeration):
        """Restretch Z in place. 1.0 is true physical proportion."""
        exaggeration = float(exaggeration)
        if exaggeration <= 0.0:
            exaggeration = 1e-6
        self.exaggeration = exaggeration
        points = self.mesh.points
        points[:, 2] = self._relief * exaggeration
        self.mesh.points = points
        return exaggeration

    # ---- what the rest of the scene needs to know ----

    @property
    def relief_height(self):
        """How tall the surface currently stands, in frame units."""
        return float(np.ptp(self.mesh.points[:, 2]))

    def height_range(self, low=0.5, high=99.5):
        """Colour limits in the printed unit - the histogram's starting span."""
        lo, hi = self.surface.percentile_range(low, high)
        return lo * self.z_factor, hi * self.z_factor

    def full_height_range(self):
        lo, hi = self.surface.full_range()
        return lo * self.z_factor, hi * self.z_factor

    @property
    def height_title(self):
        return '%s [%s]' % (self.surface.name, self.z_label or 'a.u.')
