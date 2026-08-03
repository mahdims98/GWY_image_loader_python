"""
Putting a processed image on disk.

Three ways out, and the differences between them are the point.
`render_annotated_figure` draws the whole figure - axes in real units, a
colourbar, a scale bar - which is what goes into a talk or a paper.
`save_pure_image` writes the pixels and nothing else, resampled to square
pixels first so that a 512x256 scan of a square region is not shown
stretched; that is what goes into further analysis or a figure assembled
elsewhere. `save_channel_to_gwy` writes back into a Gwyddion container,
appending rather than overwriting, so repeated saves collect the processed
channels of one measurement in one file, next to the untouched channels they
came from - and next to the metadata blocks they came with, since a channel
that has lost the setpoint and the scan rate it was taken at is a picture
rather than a measurement.

All of it colours through gwy_colormaps, so the two image routes and every
preview on screen agree about what a height means.

Matplotlib is used through the Figure API only, never pyplot, so none of this
needs an interactive backend or a running event loop.
"""

import os

import numpy as np
import matplotlib.image as mpimage
import matplotlib.patheffects as patheffects
from matplotlib.figure import Figure

import gwy_loader
import gwy_colormaps as gcm



def nice_scale_length(target):
    """Round `target` down to a 'nice' 1-2-5 style length for a scale bar."""
    if target <= 0:
        return 1.0
    exp = np.floor(np.log10(target))
    base = target / 10**exp
    for b in (5.0, 2.0, 1.0):
        if base >= b:
            return b * 10**exp
    return 10**exp


def add_scale_bar(ax, x_real, y_real, units):
    """Draw a scale bar in the lower-right corner of an image axes."""
    length = nice_scale_length(x_real / 5.0)
    margin = 0.05 * x_real
    x1 = x_real - margin
    x0 = x1 - length
    y = 0.07 * y_real
    ax.plot(
        [x0, x1], [y, y], color="white", linewidth=4, solid_capstyle="butt",
        path_effects=[patheffects.Stroke(linewidth=6, foreground="black"),
                      patheffects.Normal()],
    )
    ax.text(
        (x0 + x1) / 2, y + 0.03 * y_real, f"{length:g} {units}",
        color="white", ha="center", va="bottom", fontsize=11, fontweight="bold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
    )


def render_annotated_figure(data, x_real, y_real, title, spatial_units, z_units,
                            dpi=150, vmin=None, vmax=None):
    """Build a publication-style figure: image, axes, colorbar and scale bar.

    `vmin`/`vmax` fix the two ends of the colour map, so a set of images can
    be drawn on one scale; without them the image is stretched over its own
    full range."""
    fig = Figure(figsize=(7, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        data, origin="upper", cmap=gcm.current(),
        extent=(0, x_real, 0, y_real), aspect="equal", vmin=vmin, vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(f"x ({spatial_units})")
    ax.set_ylabel(f"y ({spatial_units})")
    fig.colorbar(im, ax=ax, pad=0.05, fraction=0.046).set_label(z_units)
    add_scale_bar(ax, x_real, y_real, spatial_units)
    fig.tight_layout()
    return fig


def _resample_to_square_pixels(data, x_real, y_real):
    """
    Bilinearly resample `data` so the output has square pixels, i.e. its
    pixel aspect matches the physical aspect ratio x_real:y_real. The finer
    of the two pixel pitches is kept, so resolution is never reduced.
    """
    ny, nx = data.shape
    pitch = min(x_real / nx, y_real / ny)
    out_nx = max(1, int(round(x_real / pitch)))
    out_ny = max(1, int(round(y_real / pitch)))
    if (out_nx, out_ny) == (nx, ny):
        return data

    # Interpolate along x, then along y (separable bilinear)
    xi = np.linspace(0, nx - 1, out_nx)
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    fx = xi - x0
    tmp = data[:, x0] * (1 - fx) + data[:, x1] * fx

    yi = np.linspace(0, ny - 1, out_ny)
    y0 = np.floor(yi).astype(int)
    y1 = np.minimum(y0 + 1, ny - 1)
    fy = (yi - y0)[:, None]
    return tmp[y0, :] * (1 - fy) + tmp[y1, :] * fy


def save_pure_image(data, path, x_real=None, y_real=None, vmin=None,
                    vmax=None):
    """Save the data as a bare colormapped image with no axes, labels,
    colorbar or scale bar.

    If the physical extents are given, the data is resampled to square
    pixels first, so the image always shows the true physical aspect ratio
    even when the scan has non-square pixels (e.g. 512x256 px over a
    square region).

    `vmin`/`vmax` fix the two ends of the colour map; without them each
    image is stretched over its own full range, which is what a single
    image wants and a set of images to be compared does not."""
    if x_real and y_real:
        data = _resample_to_square_pixels(data, x_real, y_real)
    mpimage.imsave(path, data, cmap=gcm.current(), origin="upper",
                   vmin=vmin, vmax=vmax)


def _gwy_channel_titles(container):
    """The titles of the channels already present in a .gwy container."""
    titles = []
    for k in container.keys():
        parts = k.split("/")
        if len(parts) == 4 and parts[1].isdigit() and parts[2:] == ["data",
                                                                    "title"]:
            titles.append(container[k])
    return titles


def meta_container(mapping):
    """A metadata block as the container Gwyddion stores it in.

    Everything goes in as a string, including the values that look like
    numbers: that is what the instrument wrote and what Gwyddion's metadata
    browser expects to read back. The typecode is forced rather than guessed,
    because a one-character value would otherwise be written as a char.
    """
    container = gwy_loader.GwyContainer()
    for key, value in mapping.items():
        text = str(value)
        container[key] = text
        container.typecodes[key] = 's'
    return container


def save_channel_to_gwy(path, title, data, xreal=None, yreal=None,
                        unit_xy="", unit_z="", extra_channels=(),
                        meta=None, extra_meta=None):
    """
    Save `data` (in SI units) as a channel of a Gwyddion .gwy file.

    If the file already exists, the channel is APPENDED with the next free
    channel number, so repeated saves collect all processed channels in
    one .gwy file.

    `extra_channels` is a sequence of (title, GwyDataField) written next to
    it - typically the untouched channels of the source measurement, so the
    saved file stands on its own. A channel whose title is already in the
    file is skipped, so saving repeatedly never duplicates them.

    `meta` is the metadata block for the processed channel and `extra_meta` a
    {title: block} for the others. What the microscope recorded about a scan
    is worth as much as the pixels - the setpoint, the scan rate, what the
    operator typed about the sample - and a processed channel saved without it
    is a picture of an experiment nobody can identify afterwards. See gwy_meta
    for putting the processing history into that block on the way through.

    Returns (channel number of `data`, its title - numbered if that title
    was taken -, titles of the extra channels written).
    """
    if os.path.exists(path):
        container = gwy_loader.GwyObject.fromfile(path)
    else:
        container = gwy_loader.GwyContainer()

    nums = []
    for k in container.keys():
        parts = k.split("/")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2] == "data":
            nums.append(int(parts[1]))
    n = max(nums) + 1 if nums else 0

    field = gwy_loader.GwyDataField(
        np.ascontiguousarray(data, dtype=np.float64),
        xreal=float(xreal) if xreal else float(data.shape[1]),
        yreal=float(yreal) if yreal else float(data.shape[0]),
        si_unit_xy=unit_xy or None,
        si_unit_z=unit_z or None,
    )
    # Gwyddion identifies channels by title, so a repeated save gets a
    # numbered one instead of a second channel with the same name.
    have = set(_gwy_channel_titles(container))
    unique, k = title, 2
    while unique in have:
        unique = f"{title} {k}"
        k += 1
    container[f"/{n}/data"] = field
    container[f"/{n}/data/title"] = unique
    if meta:
        container[f"/{n}/meta"] = meta_container(meta)
    have.add(unique)

    extra_meta = extra_meta or {}
    written = []
    for extra_title, extra_field in extra_channels:
        if extra_title in have:
            continue
        n += 1
        container[f"/{n}/data"] = extra_field
        container[f"/{n}/data/title"] = extra_title
        if extra_meta.get(extra_title):
            container[f"/{n}/meta"] = meta_container(extra_meta[extra_title])
        have.add(extra_title)
        written.append(extra_title)

    container.tofile(path)
    return n - len(written), unique, written
