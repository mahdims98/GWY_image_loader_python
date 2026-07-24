"""
GUI front-end for gwy_processing.py

Provides an interactive Tkinter application to:
  - Load Gwyddion (.gwy) files via gwy_loader and select a channel
  - Apply processing steps from gwy_processing, each in its own dialog
    window with a live preview of the result AND the removed component:
      * Plane leveling (level_by_plane_fit)
      * Polynomial background removal with separate x/y orders
        (level_by_polynomial_xy)
      * Align rows: median of differences / polynomial (align_rows)
      * Percentile range clipping (filter_by_percentile)
      * FFT lowpass/highpass filtering (filter_by_2d_fft) with the
        spectrum shown and click-to-set cutoff
      * Scar removal (remove_scars)
      * Set baseline to zero (set_baseline_to_zero)
  - Keep a log of every change applied to the image
  - Undo changes step by step (or reset to the original data)
  - Batch-process every .gwy file in a folder by replaying the
    current processing pipeline on the selected channel

Run with:  python gwy_processor_gui.py
"""

import os
import threading
import traceback
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.image as mpimage
import matplotlib.patheffects as patheffects
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.widgets import RectangleSelector, SpanSelector

import gwy_loader
import gwy_processing as gp


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------
# Each operation defines:
#   label         - button / dialog title
#   func          - callable(data, params, dx, dy) -> processed data
#   params        - list of parameter specs used to build the dialog widgets:
#                     {name, label, type: 'int'|'float'|'choice'|'bool',
#                      default, min, max, values}
#   removed_label - title for the "what was removed" preview panel
#   validate      - optional callable(params) -> error string or None

def _op_plane(data, params, dx, dy):
    return gp.level_by_plane_fit(data)


def _op_polynomial(data, params, dx, dy):
    return gp.level_by_polynomial_xy(
        data, x_order=params["x_order"], y_order=params["y_order"]
    )


def _op_align_rows(data, params, dx, dy):
    return gp.align_rows(data, method=params["method"], order=params.get("order", 1))


def _op_percentile(data, params, dx, dy):
    return gp.filter_by_percentile(
        data, min_percentile=params["min"], max_percentile=params["max"]
    )


def _op_fft(data, params, dx, dy):
    window = params.get("window", "hanning")
    if window in (True, False):  # backward compat with old bool pipelines
        window = "hanning" if window else "none"
    return gp.filter_by_2d_fft(
        data,
        cutoff_freq=params["cutoff"],
        mode=params["mode"],
        dx=dx,
        dy=dy,
        window=None if window == "none" else window,
        alpha=params.get("alpha", 0.5),
    )


def _op_fft_notch(data, params, dx, dy):
    notches = list(params.get("notches", []))
    if params.get("auto"):
        # Re-detect peaks on THIS image (batch-friendly: every image gets
        # its own detection instead of replaying fixed frequencies)
        window = params.get("window", "hanning")
        detected = gp.detect_fft_peaks(
            data, dx=dx, dy=dy,
            protect_radius=params.get("protect_radius", 3.0),
            threshold_db=params.get("threshold_db", 15.0),
            max_peaks=50,
            min_separation=params["radius"],
            window=None if window == "none" else window,
            alpha=params.get("alpha", 0.5),
        )
        notches += [list(p) for p in detected]
    mask = gp.build_notch_mask(
        data.shape, dx=dx, dy=dy,
        notches=notches,
        radius=params["radius"],
    )
    x_bands = params.get("x_bands", [])
    y_bands = params.get("y_bands", [])
    if x_bands or y_bands:
        mask &= gp.build_band_mask(
            data.shape, dx=dx, dy=dy,
            x_bands=x_bands, y_bands=y_bands,
            half_width=params["radius"],
        )
    return gp.filter_by_2d_fft_mask(data, mask)


def _op_scars(data, params, dx, dy):
    return gp.remove_scars(
        data, threshold=params["threshold"], min_length=params["min_length"]
    )


def _op_zero(data, params, dx, dy):
    return gp.set_baseline_to_zero(data)


def _op_crop(data, params, dx, dy):
    return gp.crop(
        data, params["x0"], params["x1"], params["y0"], params["y1"],
        dx=dx, dy=dy,
    )


def _validate_crop(params):
    if params["x1"] <= params["x0"] or params["y1"] <= params["y0"]:
        return "Crop range must have x1 > x0 and y1 > y0"
    if params["x0"] < 0 or params["y0"] < 0:
        return "Crop range cannot be negative"
    return None


def _validate_percentile(params):
    if not (0 <= params["min"] < params["max"] <= 100):
        return "Percentiles must satisfy 0 <= min < max <= 100"
    return None


def _validate_alpha(params):
    if not (0.0 <= params.get("alpha", 0.5) <= 1.0):
        return "Tukey taper must be between 0 and 1"
    return None


def _validate_fft(params):
    if params["cutoff"] <= 0:
        return "Cutoff frequency must be positive"
    return _validate_alpha(params)


def _validate_notch(params):
    if params["radius"] <= 0:
        return "Notch radius must be positive"
    if params["protect_radius"] < 0:
        return "Protect radius cannot be negative"
    return _validate_alpha(params)


def _describe_notch(params):
    parts = [f"{len(params.get('notches', []))} notches"]
    if params.get("auto"):
        parts.append(f"auto-detect@{params.get('threshold_db')}dB")
    if params.get("x_bands"):
        parts.append(f"{len(params['x_bands'])} v-bands")
    if params.get("y_bands"):
        parts.append(f"{len(params['y_bands'])} h-bands")
    parts.append(f"radius={params['radius']}")
    return ", ".join(parts)


OPERATIONS = {
    "crop": {
        "label": "Crop",
        "func": _op_crop,
        "params": [
            {"name": "x0", "label": "x0", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1e9},
            {"name": "x1", "label": "x1", "type": "float",
             "default": 1.0, "min": 0.0, "max": 1e9},
            {"name": "y0", "label": "y0", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1e9},
            {"name": "y1", "label": "y1", "type": "float",
             "default": 1.0, "min": 0.0, "max": 1e9},
        ],
        "removed_label": "",  # not used; CropDialog draws its own panels
        "validate": _validate_crop,
    },
    "plane_level": {
        "label": "Plane level",
        "func": _op_plane,
        "params": [],
        "removed_label": "Removed plane",
    },
    "polynomial": {
        "label": "Polynomial background",
        "func": _op_polynomial,
        "params": [
            {"name": "x_order", "label": "X order", "type": "int",
             "default": 2, "min": 0, "max": 10},
            {"name": "y_order", "label": "Y order", "type": "int",
             "default": 2, "min": 0, "max": 10},
        ],
        "removed_label": "Removed background",
    },
    "align_rows": {
        "label": "Align rows",
        "func": _op_align_rows,
        "params": [
            {"name": "method", "label": "Method", "type": "choice",
             "default": "median_diff", "values": ["median_diff", "polynomial"]},
            {"name": "order", "label": "Poly order", "type": "int",
             "default": 1, "min": 0, "max": 5},
        ],
        "removed_label": "Removed row offsets",
    },
    "percentile": {
        "label": "Percentile range clip",
        "func": _op_percentile,
        "params": [
            {"name": "min", "label": "Min %", "type": "float",
             "default": 0.5, "min": 0.0, "max": 100.0},
            {"name": "max", "label": "Max %", "type": "float",
             "default": 99.5, "min": 0.0, "max": 100.0},
        ],
        "removed_label": "Clipped values (difference)",
        "validate": _validate_percentile,
    },
    "fft_filter": {
        "label": "FFT filter",
        "func": _op_fft,
        "params": [
            {"name": "mode", "label": "Mode", "type": "choice",
             "default": "lowpass", "values": ["lowpass", "highpass"]},
            {"name": "cutoff", "label": "Cutoff (1/spatial unit)", "type": "float",
             "default": 10.0, "min": 0.0, "max": 1e9},
            {"name": "window", "label": "Window", "type": "choice",
             "default": "hanning", "values": ["hanning", "tukey", "none"]},
            {"name": "alpha", "label": "Tukey taper (0-1)", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1.0},
        ],
        "removed_label": "Removed component (noise)",
        "validate": _validate_fft,
    },
    "fft_notch": {
        "label": "FFT notch filter",
        "func": _op_fft_notch,
        "params": [
            {"name": "radius", "label": "Notch radius", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1e9},
            {"name": "threshold_db", "label": "Detect threshold (dB)", "type": "float",
             "default": 15.0, "min": 0.0, "max": 200.0},
            {"name": "protect_radius", "label": "Protect center radius", "type": "float",
             "default": 3.0, "min": 0.0, "max": 1e9},
            {"name": "window", "label": "Spectrum window", "type": "choice",
             "default": "hanning", "values": ["hanning", "tukey", "none"]},
            {"name": "alpha", "label": "Tukey taper (0-1)", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1.0},
            {"name": "auto", "label": "Auto re-detect (per image)", "type": "bool",
             "default": False},
        ],
        "removed_label": "Removed periodic noise",
        "validate": _validate_notch,
        "describe": _describe_notch,
    },
    "remove_scars": {
        "label": "Remove scars",
        "func": _op_scars,
        "params": [
            {"name": "threshold", "label": "Threshold (x RMS)", "type": "float",
             "default": 3.0, "min": 0.1, "max": 100.0},
            {"name": "min_length", "label": "Min length (px)", "type": "int",
             "default": 5, "min": 1, "max": 10000},
        ],
        "removed_label": "Removed scars (difference)",
    },
    "zero_baseline": {
        "label": "Zero baseline",
        "func": _op_zero,
        "params": [],
        "removed_label": "Subtracted offset",
        "instant": True,  # applied directly, no preview dialog
    },
}

# Order in which operation buttons appear in the main window
OPERATION_ORDER = [
    "crop",
    "plane_level",
    "polynomial",
    "align_rows",
    "percentile",
    "fft_filter",
    "fft_notch",
    "remove_scars",
    "zero_baseline",
]


def describe_step(op_key, params):
    """Human-readable one-line description of a processing step."""
    spec = OPERATIONS[op_key]
    label = spec["label"]
    if "describe" in spec:
        return f"{label} ({spec['describe'](params)})"
    if params:
        p = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{label} ({p})"
    return label


def apply_pipeline(data, pipeline, dx, dy):
    """Apply a list of (op_key, params) steps to `data` and return the result."""
    for op_key, params in pipeline:
        func = OPERATIONS[op_key]["func"]
        data = func(data, params, dx, dy)
    return data


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _unit_of(field, attr):
    """Extract the unit string of a GwyDataField axis ('si_unit_xy'/'si_unit_z')."""
    try:
        unit = field.get(attr, None)
        if unit is not None:
            return unit.unitstr
    except Exception:
        pass
    return ""


def spatial_scale(unitstr):
    """Return (scale_factor, display_unit) for the lateral axes."""
    if unitstr == "m":
        return 1e6, "µm"  # meters -> micrometers
    return 1.0, unitstr or "px"


def z_scale(unitstr):
    """Return (scale_factor, display_unit) for the z (value) axis."""
    if unitstr == "m":
        return 1e9, "nm"  # meters -> nanometers
    return 1.0, unitstr or "a.u."


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

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
                            dpi=150):
    """Build a publication-style figure: image, axes, colorbar and scale bar."""
    fig = Figure(figsize=(7, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        data, origin="upper", cmap=gp.get_gwyddion_cmap(),
        extent=(0, x_real, 0, y_real), aspect="equal",
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


def save_pure_image(data, path, x_real=None, y_real=None):
    """Save the data as a bare colormapped image with no axes, labels,
    colorbar or scale bar.

    If the physical extents are given, the data is resampled to square
    pixels first, so the image always shows the true physical aspect ratio
    even when the scan has non-square pixels (e.g. 512x256 px over a
    square region)."""
    if x_real and y_real:
        data = _resample_to_square_pixels(data, x_real, y_real)
    mpimage.imsave(path, data, cmap=gp.get_gwyddion_cmap(), origin="upper")


def save_channel_to_gwy(path, title, data, xreal=None, yreal=None,
                        unit_xy="", unit_z=""):
    """
    Save `data` (in SI units) as a channel of a Gwyddion .gwy file.

    If the file already exists, the channel is APPENDED with the next free
    channel number, so repeated saves collect all processed channels in
    one .gwy file. Returns the channel number used.
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
    container[f"/{n}/data"] = field
    container[f"/{n}/data/title"] = title
    container.tofile(path)
    return n


# ---------------------------------------------------------------------------
# Operation dialog with live preview
# ---------------------------------------------------------------------------

class OperationDialog(tk.Toplevel):
    """
    A per-operation window: parameter widgets on top, a live preview of
    the processed result and of the removed component below, and
    Apply / Cancel buttons.

    The preview refreshes automatically (debounced) whenever a parameter
    changes.
    """

    PREVIEW_DEBOUNCE_MS = 400

    def __init__(self, app, op_key):
        super().__init__(app)
        self.app = app
        self.op_key = op_key
        self.spec = OPERATIONS[op_key]
        self.title(self.spec["label"])
        self.geometry("1050x560")

        self._after_id = None
        self.vars = {}

        self._build_params()
        self._build_figure()
        self._build_buttons()

        self.update_preview()

    # ---- UI construction ----

    def _build_params(self):
        frame = ttk.Frame(self, padding=8)
        frame.pack(side=tk.TOP, fill=tk.X)
        self.params_frame = frame
        self.param_widgets = {}

        if not self.spec["params"]:
            ttk.Label(frame, text="No parameters for this operation.").pack(side=tk.LEFT)

        for p in self.spec["params"]:
            ttk.Label(frame, text=p["label"] + ":").pack(side=tk.LEFT, padx=(8, 2))
            if p["type"] == "int":
                var = tk.IntVar(value=p["default"])
                widget = ttk.Spinbox(
                    frame, from_=p.get("min", 0), to=p.get("max", 100),
                    width=5, textvariable=var,
                )
            elif p["type"] == "float":
                var = tk.DoubleVar(value=p["default"])
                widget = ttk.Entry(frame, textvariable=var, width=8)
            elif p["type"] == "choice":
                var = tk.StringVar(value=p["default"])
                widget = ttk.Combobox(
                    frame, textvariable=var, values=p["values"],
                    state="readonly", width=max(len(v) for v in p["values"]) + 2,
                )
            elif p["type"] == "bool":
                var = tk.BooleanVar(value=p["default"])
                widget = ttk.Checkbutton(frame, variable=var)
            else:
                raise ValueError(f"Unknown param type: {p['type']}")
            widget.pack(side=tk.LEFT)

            var.trace_add("write", self._on_param_change)
            self.vars[p["name"]] = var
            self.param_widgets[p["name"]] = widget

        self.status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.status_var, foreground="red").pack(
            side=tk.RIGHT, padx=8
        )

    def _build_figure(self):
        self.figure = Figure(figsize=(10, 4.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

    def _build_buttons(self):
        frame = ttk.Frame(self, padding=8)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="Apply", command=self.apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="Update preview", command=self.update_preview).pack(
            side=tk.RIGHT, padx=4
        )

    # ---- Parameters ----

    def get_params(self):
        """Read current parameter values; returns None while an entry holds
        un-parseable text (e.g. mid-typing)."""
        params = {}
        for p in self.spec["params"]:
            try:
                params[p["name"]] = self.vars[p["name"]].get()
            except tk.TclError:
                return None
        return params

    def _on_param_change(self, *args):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(self.PREVIEW_DEBOUNCE_MS, self.update_preview)

    def _compute(self, params):
        """Run the operation on the app's current data."""
        return self.spec["func"](self.app.data, params, self.app.dx, self.app.dy)

    def _validated_params(self, show_error=False):
        params = self.get_params()
        if params is None:
            return None
        validate = self.spec.get("validate")
        if validate is not None:
            err = validate(params)
            if err:
                if show_error:
                    messagebox.showerror("Invalid parameters", err, parent=self)
                else:
                    self.status_var.set(err)
                return None
        self.status_var.set("")
        return params

    # ---- Preview ----

    def update_preview(self):
        self._after_id = None
        params = self._validated_params()
        if params is None:
            return
        try:
            result = self._compute(params)
        except Exception as e:
            self.status_var.set(str(e))
            return
        removed = self.app.data - result
        self._draw(result, removed)

    def _draw(self, result, removed):
        app = self.app
        extent = (0, app.x_real, 0, app.y_real)
        self.figure.clf()
        ax1, ax2 = self.figure.subplots(1, 2)

        im1 = ax1.imshow(
            result, origin="upper", cmap=gp.get_gwyddion_cmap(),
            extent=extent, aspect="equal",
        )
        ax1.set_title("Preview: result")
        ax1.set_xlabel(f"x ({app.spatial_units})")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        im2 = ax2.imshow(
            removed, origin="upper", cmap="viridis",
            extent=extent, aspect="equal",
        )
        ax2.set_title(self.spec["removed_label"])
        ax2.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im2, ax=ax2, fraction=0.046).set_label(app.z_units)

        self.figure.tight_layout()
        self.canvas.draw()

    # ---- Apply ----

    def apply(self):
        params = self._validated_params(show_error=True)
        if params is None:
            return
        self.app.apply_operation(self.op_key, params)
        self.destroy()


class PolynomialDialog(OperationDialog):
    """Polynomial background dialog with an option to sync the x and y orders."""

    def __init__(self, app, op_key="polynomial"):
        super().__init__(app, op_key)

    def _build_params(self):
        super()._build_params()
        self.sync_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_frame, text="Sync x/y", variable=self.sync_var,
            command=self._on_sync_toggle,
        ).pack(side=tk.LEFT, padx=(12, 0))
        self.vars["x_order"].trace_add("write", self._on_x_change)

    def _on_sync_toggle(self):
        if self.sync_var.get():
            try:
                self.vars["y_order"].set(self.vars["x_order"].get())
            except tk.TclError:
                pass
            self.param_widgets["y_order"].state(["disabled"])
        else:
            self.param_widgets["y_order"].state(["!disabled"])

    def _on_x_change(self, *args):
        if self.sync_var.get():
            try:
                self.vars["y_order"].set(self.vars["x_order"].get())
            except tk.TclError:
                pass


class FFTFilterDialog(OperationDialog):
    """
    FFT filter dialog with three panels: the FFT magnitude spectrum with
    the cutoff drawn as a circle (click on the spectrum to set the cutoff),
    the filtered result, and the removed component.
    """

    def __init__(self, app, op_key="fft_filter"):
        self._spectrum = None
        self._spectrum_key = None
        self._spec_ax = None
        super().__init__(app, op_key)
        self.geometry("1350x560")
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _window_settings(self):
        """Current (window, alpha) from the dialog, with safe fallbacks."""
        try:
            window = self.vars["window"].get()
        except (KeyError, tk.TclError):
            window = "hanning"
        try:
            alpha = self.vars["alpha"].get()
        except (KeyError, tk.TclError):
            alpha = 0.5
        return window, alpha

    def _ensure_spectrum(self):
        window, alpha = self._window_settings()
        key = (window, alpha)
        if self._spectrum is None or self._spectrum_key != key:
            self._spectrum = gp.get_2d_fft_magnitude(
                self.app.data, dx=self.app.dx, dy=self.app.dy,
                window=None if window == "none" else window, alpha=alpha,
            )
            self._spectrum_key = key

    def _on_click(self, event):
        if event.inaxes is not self._spec_ax or event.xdata is None:
            return
        cutoff = float(np.hypot(event.xdata, event.ydata))
        self.vars["cutoff"].set(round(cutoff, 3))
        # trace on the variable triggers the debounced preview update

    def _draw(self, result, removed):
        app = self.app
        self._ensure_spectrum()
        mag, freq_extent = self._spectrum
        extent = (0, app.x_real, 0, app.y_real)

        self.figure.clf()
        ax0, ax1, ax2 = self.figure.subplots(1, 3)

        im0 = ax0.imshow(
            mag, origin="upper", cmap="viridis",
            extent=freq_extent, aspect="equal",
        )
        ax0.set_title("FFT spectrum (click to set cutoff)")
        ax0.set_xlabel(f"fx (1/{app.spatial_units})")
        ax0.set_ylabel(f"fy (1/{app.spatial_units})")
        self.figure.colorbar(im0, ax=ax0, fraction=0.046).set_label("dB")
        try:
            cutoff = self.vars["cutoff"].get()
            ax0.add_patch(Circle((0, 0), cutoff, fill=False,
                                 color="red", linewidth=1.5))
        except tk.TclError:
            pass
        self._spec_ax = ax0

        im1 = ax1.imshow(
            result, origin="upper", cmap=gp.get_gwyddion_cmap(),
            extent=extent, aspect="equal",
        )
        ax1.set_title("Preview: result")
        ax1.set_xlabel(f"x ({app.spatial_units})")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        im2 = ax2.imshow(
            removed, origin="upper", cmap="viridis",
            extent=extent, aspect="equal",
        )
        ax2.set_title(self.spec["removed_label"])
        ax2.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im2, ax=ax2, fraction=0.046).set_label(app.z_units)

        self.figure.tight_layout()
        self.canvas.draw()


class NotchFilterDialog(OperationDialog):
    """
    FFT notch filter dialog for removing specific periodic signals.

    Three panels: the FFT magnitude spectrum where notches are placed by
    clicking (left-click adds a notch at the clicked frequency, right-click
    removes the nearest one), the filtered result, and the removed noise.

    'Auto-detect peaks' finds all sharp spectral peaks outside the protected
    center region and notches them in one go - this removes most periodic
    signals that are not part of the low-frequency image content.

    Every notch is applied symmetrically at +/-f, so clicking one peak of a
    conjugate pair is enough.
    """

    def __init__(self, app, op_key="fft_notch"):
        self.notches = []       # list of [fx, fy] circular notches
        self.x_bands = []       # list of fx centers (vertical band notches)
        self.y_bands = []       # list of fy centers (horizontal band notches)
        self._spectrum = None
        self._spectrum_key = None
        self._spec_ax = None
        super().__init__(app, op_key)
        self.geometry("1400x620")
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # ---- extra controls ----

    def _build_params(self):
        super()._build_params()
        btns = ttk.Frame(self, padding=(8, 0, 8, 4))
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="Auto-detect peaks", command=self.auto_detect).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btns, text="Clear all", command=self.clear_notches).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(btns, text="Click adds:").pack(side=tk.LEFT, padx=(12, 2))
        self.click_mode_var = tk.StringVar(value="circle")
        ttk.Combobox(
            btns, textvariable=self.click_mode_var,
            values=["circle", "vertical band", "horizontal band"],
            state="readonly", width=15,
        ).pack(side=tk.LEFT)
        ttk.Label(
            btns,
            text="Left-click spectrum: add  |  Right-click: remove nearest",
        ).pack(side=tk.LEFT, padx=12)

    def get_params(self):
        params = super().get_params()
        if params is not None:
            params["notches"] = [list(n) for n in self.notches]
            params["x_bands"] = list(self.x_bands)
            params["y_bands"] = list(self.y_bands)
        return params

    # ---- notch management ----

    def auto_detect(self):
        params = self._validated_params()
        if params is None:
            return
        window, alpha = self._window_settings()
        peaks = gp.detect_fft_peaks(
            self.app.data,
            dx=self.app.dx,
            dy=self.app.dy,
            protect_radius=params["protect_radius"],
            threshold_db=params["threshold_db"],
            max_peaks=50,
            min_separation=params["radius"],
            window=None if window == "none" else window,
            alpha=alpha,
        )
        self.notches = [list(p) for p in peaks]
        self.status_var.set(f"{len(peaks)} peaks detected")
        self.update_preview()

    def clear_notches(self):
        self.notches = []
        self.x_bands = []
        self.y_bands = []
        self.update_preview()

    def _on_click(self, event):
        if event.inaxes is not self._spec_ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.button == 1:
            mode = self.click_mode_var.get()
            if mode == "vertical band":
                self.x_bands.append(abs(x))
            elif mode == "horizontal band":
                self.y_bands.append(abs(y))
            else:
                self.notches.append([x, y])
            self.update_preview()
        elif event.button == 3:
            # find the globally nearest item (circle, v-band or h-band),
            # considering mirrored counterparts too
            best = None  # (distance, list, index)
            for i, (fx, fy) in enumerate(self.notches):
                d = min(np.hypot(x - fx, y - fy), np.hypot(x + fx, y + fy))
                if best is None or d < best[0]:
                    best = (d, self.notches, i)
            for i, c in enumerate(self.x_bands):
                d = min(abs(x - c), abs(x + c))
                if best is None or d < best[0]:
                    best = (d, self.x_bands, i)
            for i, c in enumerate(self.y_bands):
                d = min(abs(y - c), abs(y + c))
                if best is None or d < best[0]:
                    best = (d, self.y_bands, i)
            if best is not None:
                best[1].pop(best[2])
                self.update_preview()

    # ---- drawing ----

    _window_settings = FFTFilterDialog._window_settings
    _ensure_spectrum = FFTFilterDialog._ensure_spectrum

    def _draw(self, result, removed):
        app = self.app
        self._ensure_spectrum()
        mag, freq_extent = self._spectrum
        extent = (0, app.x_real, 0, app.y_real)

        self.figure.clf()
        ax0, ax1, ax2 = self.figure.subplots(1, 3)

        im0 = ax0.imshow(
            mag, origin="upper", cmap="viridis",
            extent=freq_extent, aspect="equal",
        )
        n_items = len(self.notches) + len(self.x_bands) + len(self.y_bands)
        ax0.set_title(f"Spectrum - {n_items} notches/bands")
        ax0.set_xlabel(f"fx (1/{app.spatial_units})")
        ax0.set_ylabel(f"fy (1/{app.spatial_units})")
        self.figure.colorbar(im0, ax=ax0, fraction=0.046).set_label("dB")
        self._spec_ax = ax0

        try:
            radius = self.vars["radius"].get()
            protect = self.vars["protect_radius"].get()
        except tk.TclError:
            radius, protect = None, None
        if protect:
            ax0.add_patch(Circle((0, 0), protect, fill=False, color="lime",
                                 linewidth=1.2, linestyle="--"))
        if radius:
            for fx, fy in self.notches:
                ax0.add_patch(Circle((fx, fy), radius, fill=False,
                                     color="red", linewidth=1.2))
                ax0.add_patch(Circle((-fx, -fy), radius, fill=False,
                                     color="red", linewidth=1.0, linestyle=":"))
            for c in self.x_bands:
                ax0.axvspan(c - radius, c + radius, color="red", alpha=0.25)
                ax0.axvspan(-c - radius, -c + radius, color="red", alpha=0.15)
            for c in self.y_bands:
                ax0.axhspan(c - radius, c + radius, color="red", alpha=0.25)
                ax0.axhspan(-c - radius, -c + radius, color="red", alpha=0.15)

        im1 = ax1.imshow(
            result, origin="upper", cmap=gp.get_gwyddion_cmap(),
            extent=extent, aspect="equal",
        )
        ax1.set_title("Preview: result")
        ax1.set_xlabel(f"x ({app.spatial_units})")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        im2 = ax2.imshow(
            removed, origin="upper", cmap="viridis",
            extent=extent, aspect="equal",
        )
        ax2.set_title(self.spec["removed_label"])
        ax2.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im2, ax=ax2, fraction=0.046).set_label(app.z_units)

        self.figure.tight_layout()
        self.canvas.draw()


class CropDialog(OperationDialog):
    """
    Crop dialog: drag a rectangle on the full image to select the region,
    with a live preview of the cropped result. The x0/x1/y0/y1 entries
    (in spatial units, y measured from the bottom) stay in sync with the
    dragged rectangle.
    """

    def __init__(self, app, op_key="crop"):
        self._rect_selector = None
        super().__init__(app, op_key)
        self.geometry("1150x560")

    def _build_params(self):
        super()._build_params()
        # Default crop region: the full image
        self.vars["x1"].set(round(self.app.x_real, 4))
        self.vars["y1"].set(round(self.app.y_real, 4))

    def _on_rect(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, x1 = sorted((float(eclick.xdata), float(erelease.xdata)))
        y0, y1 = sorted((float(eclick.ydata), float(erelease.ydata)))
        self.vars["x0"].set(round(max(0.0, x0), 4))
        self.vars["x1"].set(round(min(self.app.x_real, x1), 4))
        self.vars["y0"].set(round(max(0.0, y0), 4))
        self.vars["y1"].set(round(min(self.app.y_real, y1), 4))
        # variable traces trigger the debounced preview update

    def update_preview(self):
        # The base implementation computes `data - result`, which is
        # meaningless for crop (shapes differ) - draw directly instead.
        self._after_id = None
        params = self._validated_params()
        if params is None:
            return
        try:
            result = self._compute(params)
        except Exception as e:
            self.status_var.set(str(e))
            return
        self._draw(result, None)

    def _draw(self, result, removed):
        app = self.app
        self.figure.clf()
        ax1, ax2 = self.figure.subplots(1, 2)

        im1 = ax1.imshow(
            app.data, origin="upper", cmap=gp.get_gwyddion_cmap(),
            extent=(0, app.x_real, 0, app.y_real), aspect="equal",
        )
        ax1.set_title("Drag to select crop region")
        ax1.set_xlabel(f"x ({app.spatial_units})")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        # Show the currently selected region
        try:
            x0, x1 = self.vars["x0"].get(), self.vars["x1"].get()
            y0, y1 = self.vars["y0"].get(), self.vars["y1"].get()
            from matplotlib.patches import Rectangle
            ax1.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor="red", linewidth=1.5,
            ))
        except tk.TclError:
            pass

        # Re-attach the rectangle selector (figure was cleared)
        try:
            self._rect_selector = RectangleSelector(
                ax1, self._on_rect, useblit=True, button=[1],
                props=dict(fill=False, edgecolor="red", linestyle="--"),
            )
        except TypeError:
            # Older matplotlib uses `rectprops`
            self._rect_selector = RectangleSelector(
                ax1, self._on_rect, useblit=True, button=[1],
                rectprops=dict(fill=False, edgecolor="red", linestyle="--"),
            )

        cy, cx = result.shape
        im2 = ax2.imshow(
            result, origin="upper", cmap=gp.get_gwyddion_cmap(),
            extent=(0, cx * app.dx, 0, cy * app.dy), aspect="equal",
        )
        ax2.set_title(f"Cropped preview ({cx}x{cy} px)")
        ax2.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im2, ax=ax2, fraction=0.046).set_label(app.z_units)

        self.figure.tight_layout()
        self.canvas.draw()


class PercentileDialog(OperationDialog):
    """
    Percentile clip dialog with three panels: the data value distribution
    (histogram) where the clip range can be selected by dragging directly
    on the plot, the clipped result, and the difference.

    The min/max percentile entries stay in sync with the dragged range.
    """

    def __init__(self, app, op_key="percentile"):
        # Sorted copy of the data for fast value <-> percentile conversion
        self._sorted = np.sort(app.data.ravel())
        self._span = None
        super().__init__(app, op_key)
        self.geometry("1350x560")

    # ---- value <-> percentile conversion ----

    def _value_to_percentile(self, value):
        n = len(self._sorted)
        pct = np.searchsorted(self._sorted, value) / n * 100.0
        return float(np.clip(pct, 0.0, 100.0))

    def _on_span(self, vmin, vmax):
        """Called when a range is dragged on the histogram."""
        if vmax <= vmin:
            return
        pmin = self._value_to_percentile(vmin)
        pmax = self._value_to_percentile(vmax)
        if pmax <= pmin:
            return
        # Setting the vars triggers the debounced preview via their traces
        self.vars["min"].set(round(pmin, 2))
        self.vars["max"].set(round(pmax, 2))

    # ---- drawing ----

    def _draw(self, result, removed):
        app = self.app
        extent = (0, app.x_real, 0, app.y_real)

        self.figure.clf()
        ax0, ax1, ax2 = self.figure.subplots(1, 3)

        # Histogram of the current data distribution (log counts so the
        # outlier tails are visible)
        ax0.hist(app.data.ravel(), bins=200, color="steelblue")
        ax0.set_yscale("log")
        ax0.set_title("Distribution (drag to select range)")
        ax0.set_xlabel(f"value ({app.z_units})")
        ax0.set_ylabel("count")

        # Mark the current clip limits on the histogram
        try:
            lo = self.vars["min"].get()
            hi = self.vars["max"].get()
            vmin = np.percentile(app.data, lo)
            vmax = np.percentile(app.data, hi)
            ax0.axvline(vmin, color="red", linewidth=1.5)
            ax0.axvline(vmax, color="red", linewidth=1.5)
            ax0.axvspan(vmin, vmax, color="red", alpha=0.08)
        except tk.TclError:
            pass

        # Re-attach the span selector (the figure was cleared)
        try:
            self._span = SpanSelector(
                ax0, self._on_span, "horizontal", useblit=True,
                props=dict(alpha=0.25, facecolor="tab:red"),
            )
        except TypeError:
            # Older matplotlib uses `rectprops` instead of `props`
            self._span = SpanSelector(
                ax0, self._on_span, "horizontal", useblit=True,
                rectprops=dict(alpha=0.25, facecolor="tab:red"),
            )

        im1 = ax1.imshow(
            result, origin="upper", cmap=gp.get_gwyddion_cmap(),
            extent=extent, aspect="equal",
        )
        ax1.set_title("Preview: result")
        ax1.set_xlabel(f"x ({app.spatial_units})")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        im2 = ax2.imshow(
            removed, origin="upper", cmap="viridis",
            extent=extent, aspect="equal",
        )
        ax2.set_title(self.spec["removed_label"])
        ax2.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im2, ax=ax2, fraction=0.046).set_label(app.z_units)

        self.figure.tight_layout()
        self.canvas.draw()


# Dialog class to use per operation (default: OperationDialog)
DIALOG_CLASSES = {
    "crop": CropDialog,
    "polynomial": PolynomialDialog,
    "fft_filter": FFTFilterDialog,
    "fft_notch": NotchFilterDialog,
    "percentile": PercentileDialog,
}


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class GwyProcessorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GWY Processor")
        self.geometry("1250x780")

        # --- State ---
        self.filename = None
        self.channels = {}          # {title: GwyDataField}
        self.field = None           # currently selected GwyDataField
        self.original_data = None   # data as loaded (display units)
        self.data = None            # current processed data (display units)
        self.undo_stack = []        # list of np.ndarray snapshots
        self.pipeline = []          # list of (op_key, params) applied in order
        self.log_entries = []       # list of log strings (with timestamps)
        self.x_real = self.y_real = 1.0
        self.dx = self.dy = 1.0
        self.spatial_units = "px"
        self.z_units = "a.u."

        self._build_layout()
        self._draw_placeholder()

    # ------------------------------------------------------------------ UI --

    def _build_layout(self):
        # Left: controls, Right: plot
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---- File / channel section ----
        file_frame = ttk.LabelFrame(left, text="File", padding=6)
        file_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(file_frame, text="Open .gwy file...", command=self.open_file).pack(
            fill=tk.X
        )
        self.file_label = ttk.Label(file_frame, text="No file loaded", wraplength=260)
        self.file_label.pack(fill=tk.X, pady=(4, 2))

        ttk.Label(file_frame, text="Channel:").pack(anchor=tk.W)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(
            file_frame, textvariable=self.channel_var, state="readonly"
        )
        self.channel_combo.pack(fill=tk.X)
        self.channel_combo.bind("<<ComboboxSelected>>", lambda e: self.select_channel())

        # ---- Operations section: one button per dialog ----
        proc = ttk.LabelFrame(left, text="Operations", padding=6)
        proc.pack(fill=tk.X, pady=(0, 6))

        for key in OPERATION_ORDER:
            suffix = "" if OPERATIONS[key].get("instant") else "..."
            ttk.Button(
                proc,
                text=OPERATIONS[key]["label"] + suffix,
                command=lambda k=key: self.open_operation(k),
            ).pack(fill=tk.X, pady=1)

        ttk.Button(proc, text="View FFT spectrum", command=self.show_fft).pack(
            fill=tk.X, pady=(6, 1)
        )

        # ---- Undo / reset ----
        hist = ttk.Frame(left)
        hist.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(hist, text="Undo", command=self.undo).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(hist, text="Reset to original", command=self.reset).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # ---- Log section ----
        log_frame = ttk.LabelFrame(left, text="Processing log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.log_list = tk.Listbox(log_frame, height=8)
        self.log_list.pack(fill=tk.BOTH, expand=True)
        ttk.Button(log_frame, text="Save log...", command=self.save_log).pack(
            fill=tk.X, pady=(4, 0)
        )

        # ---- Save / batch ----
        out = ttk.LabelFrame(left, text="Output", padding=6)
        out.pack(fill=tk.X)
        ttk.Button(out, text="Save processed image...", command=self.save_image).pack(
            fill=tk.X, pady=1
        )
        ttk.Button(out, text="Save channel to .gwy...", command=self.save_to_gwy).pack(
            fill=tk.X, pady=1
        )
        ttk.Button(out, text="Batch process folder...", command=self.batch_dialog).pack(
            fill=tk.X, pady=1
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, wraplength=260).pack(
            fill=tk.X, pady=(6, 0)
        )

        # ---- Plot area ----
        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

    # ------------------------------------------------------------- Loading --

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open Gwyddion file",
            filetypes=[("Gwyddion files", "*.gwy"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            channels = gwy_loader.load_gwy(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not read file:\n{e}")
            return
        if not channels:
            messagebox.showerror("Load error", "No data channels found in file.")
            return

        self.filename = path
        self.channels = channels
        self.file_label.config(text=os.path.basename(path))
        names = list(channels.keys())
        self.channel_combo["values"] = names

        # Prefer a Height channel if present
        default = next((n for n in names if "Height" in n), names[0])
        self.channel_var.set(default)
        self.select_channel()

    def select_channel(self):
        name = self.channel_var.get()
        if name not in self.channels:
            return
        if self.pipeline:
            if not messagebox.askyesno(
                "Change channel",
                "Changing the channel discards the current processing history.\nContinue?",
            ):
                return

        self.field = self.channels[name]

        xy_unit = _unit_of(self.field, "si_unit_xy")
        z_unit = _unit_of(self.field, "si_unit_z")
        xy_factor, self.spatial_units = spatial_scale(xy_unit)
        z_factor, self.z_units = z_scale(z_unit)
        self.z_factor = z_factor          # display units -> SI conversion
        self.unit_xy_str = xy_unit        # original SI unit strings, kept
        self.unit_z_str = z_unit          # for .gwy export

        data = self.field.data.astype(np.float64) * z_factor
        ny, nx = data.shape
        self.x_real = (self.field.xreal or nx) * xy_factor
        self.y_real = (self.field.yreal or ny) * xy_factor
        self.dx = self.x_real / nx
        self.dy = self.y_real / ny

        self.original_data = data.copy()
        self._orig_extent = (self.x_real, self.y_real)
        self.data = data
        self.undo_stack = []
        self.pipeline = []
        self.log_entries = []
        self.log_list.delete(0, tk.END)
        self._log(f"Loaded channel '{name}' from {os.path.basename(self.filename)}")
        self.status_var.set(f"Channel '{name}' loaded ({nx}x{ny})")
        self.redraw()

    # ----------------------------------------------------------- Operations --

    def _require_data(self):
        if self.data is None:
            messagebox.showinfo("No data", "Open a .gwy file and select a channel first.")
            return False
        return True

    def open_operation(self, op_key):
        """Open the dialog window for one operation (or apply directly for
        parameter-less instant operations like zero baseline)."""
        if not self._require_data():
            return
        if OPERATIONS[op_key].get("instant"):
            self.apply_operation(op_key, {})
            return
        dialog_cls = DIALOG_CLASSES.get(op_key, OperationDialog)
        dialog_cls(self, op_key)

    def apply_operation(self, op_key, params):
        """Apply one operation, push undo state, record pipeline + log.
        Called by the operation dialogs on Apply."""
        func = OPERATIONS[op_key]["func"]
        try:
            new_data = func(self.data, params, self.dx, self.dy)
        except Exception as e:
            messagebox.showerror(
                "Processing error", f"{describe_step(op_key, params)} failed:\n{e}"
            )
            return
        self.undo_stack.append((self.data, self.x_real, self.y_real))
        self.data = new_data
        # Operations like crop change the image dimensions; keep the
        # physical extents consistent (pixel size dx/dy never changes).
        ny, nx = new_data.shape
        self.x_real = nx * self.dx
        self.y_real = ny * self.dy
        self.pipeline.append((op_key, params))
        self._log(describe_step(op_key, params))
        self.redraw()

    def show_fft(self):
        """Open a window showing the current FFT magnitude spectrum."""
        if not self._require_data():
            return
        mag, extent = gp.get_2d_fft_magnitude(
            self.data, dx=self.dx, dy=self.dy, window="hanning"
        )
        win = tk.Toplevel(self)
        win.title("2D FFT magnitude")
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(mag, extent=extent, cmap="viridis", origin="upper", aspect="equal")
        ax.set_xlabel(f"Frequency X (1/{self.spatial_units})")
        ax.set_ylabel(f"Frequency Y (1/{self.spatial_units})")
        ax.set_title("2D FFT magnitude (dB)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        canvas.draw()

    # ------------------------------------------------------- Undo / logging --

    def undo(self):
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        self.data, self.x_real, self.y_real = self.undo_stack.pop()
        undone = self.pipeline.pop()
        self._log(f"UNDO: {describe_step(*undone)}")
        self.redraw()

    def reset(self):
        if self.original_data is None:
            return
        if not self.pipeline and not self.undo_stack:
            return
        self.undo_stack.append((self.data, self.x_real, self.y_real))
        self.data = self.original_data.copy()
        self.x_real, self.y_real = self._orig_extent
        self.pipeline = []
        self._log("Reset to original data")
        self.redraw()

    def _log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{stamp}] {message}"
        self.log_entries.append(entry)
        self.log_list.insert(tk.END, entry)
        self.log_list.see(tk.END)

    def save_log(self):
        if not self.log_entries:
            messagebox.showinfo("Empty log", "There is nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="processing_log.txt",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# GWY processing log - {datetime.now().isoformat()}\n")
            f.write(f"# File: {self.filename}\n")
            f.write(f"# Channel: {self.channel_var.get()}\n\n")
            f.write("\n".join(self.log_entries) + "\n")
        self.status_var.set(f"Log saved to {os.path.basename(path)}")

    # ---------------------------------------------------------------- Plot --

    def _draw_placeholder(self):
        self.ax.text(
            0.5, 0.5, "Open a .gwy file to begin",
            ha="center", va="center", transform=self.ax.transAxes,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def redraw(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        im = self.ax.imshow(
            self.data,
            origin="upper",
            cmap=gp.get_gwyddion_cmap(),
            extent=(0, self.x_real, 0, self.y_real),
            aspect="equal",
        )
        title = self.channel_var.get() or "Image"
        self.ax.set_title(title)
        self.ax.set_xlabel(f"x ({self.spatial_units})")
        self.ax.set_ylabel(f"y ({self.spatial_units})")
        cbar = self.figure.colorbar(im, ax=self.ax, pad=0.05, fraction=0.046)
        cbar.set_label(self.z_units)
        self.figure.tight_layout()
        self.canvas.draw()

    # -------------------------------------------------------------- Saving --

    def save_image(self):
        if not self._require_data():
            return
        base = os.path.splitext(os.path.basename(self.filename or "processed"))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("NumPy data", "*.npy")],
            initialfile=f"{base}_processed.png",
        )
        if not path:
            return
        if path.lower().endswith(".npy"):
            np.save(path, self.data)
            self._log(f"Saved output to {os.path.basename(path)}")
            self.status_var.set(f"Saved {os.path.basename(path)}")
            return

        # Annotated image: axes, colorbar and scale bar
        fig = render_annotated_figure(
            self.data, self.x_real, self.y_real,
            self.channel_var.get() or "Image",
            self.spatial_units, self.z_units,
        )
        fig.savefig(path, dpi=200, bbox_inches="tight")

        # Pure image (no labels, one pixel per data point) in a 'pure' subfolder
        pure_dir = os.path.join(os.path.dirname(path) or ".", "pure")
        os.makedirs(pure_dir, exist_ok=True)
        pure_path = os.path.join(pure_dir, os.path.basename(path))
        save_pure_image(self.data, pure_path, self.x_real, self.y_real)

        self._log(f"Saved {os.path.basename(path)} (+ pure/{os.path.basename(path)})")
        self.status_var.set(f"Saved {os.path.basename(path)} and pure copy")

    def save_to_gwy(self):
        """Append the processed channel to a .gwy file (creating it if needed),
        so all processed channels can be collected in one Gwyddion file."""
        if not self._require_data():
            return
        base = os.path.splitext(os.path.basename(self.filename or "image"))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".gwy",
            filetypes=[("Gwyddion files", "*.gwy")],
            initialfile="processed.gwy",
            confirmoverwrite=False,  # existing files are appended to, not replaced
        )
        if not path:
            return
        title = f"{base} - {self.channel_var.get()} (processed)"
        try:
            n = save_channel_to_gwy(
                path, title,
                self.data / self.z_factor,  # back to SI units
                xreal=self.field.xreal, yreal=self.field.yreal,
                unit_xy=self.unit_xy_str, unit_z=self.unit_z_str,
            )
        except Exception as e:
            messagebox.showerror("Save error", f"Could not write .gwy file:\n{e}")
            return
        self._log(f"Saved channel {n} '{title}' to {os.path.basename(path)}")
        self.status_var.set(f"Appended channel {n} to {os.path.basename(path)}")

    # --------------------------------------------------------------- Batch --

    def batch_dialog(self):
        """Ask for input/output folders and replay the current pipeline."""
        if not self.pipeline:
            messagebox.showinfo(
                "No pipeline",
                "Apply at least one processing step to the current image first.\n"
                "The batch run replays those same steps on every file.",
            )
            return
        channel = self.channel_var.get()
        steps = "\n".join(f"  {i+1}. {describe_step(*s)}" for i, s in enumerate(self.pipeline))
        if not messagebox.askokcancel(
            "Batch process",
            f"Channel: {channel}\nPipeline to apply to every .gwy file:\n{steps}\n\n"
            "Choose the input folder next.",
        ):
            return
        in_dir = filedialog.askdirectory(title="Select folder with .gwy files")
        if not in_dir:
            return
        out_dir = filedialog.askdirectory(title="Select output folder")
        if not out_dir:
            return

        files = sorted(
            f for f in os.listdir(in_dir) if f.lower().endswith(".gwy")
        )
        if not files:
            messagebox.showinfo("No files", "No .gwy files found in the selected folder.")
            return

        # Run in a background thread so the UI stays responsive.
        pipeline = list(self.pipeline)
        thread = threading.Thread(
            target=self._batch_worker,
            args=(in_dir, out_dir, files, channel, pipeline),
            daemon=True,
        )
        thread.start()

    def _batch_worker(self, in_dir, out_dir, files, channel, pipeline):
        results = []
        pure_dir = os.path.join(out_dir, "pure")
        os.makedirs(pure_dir, exist_ok=True)
        gwy_out = gwy_loader.GwyContainer()
        gwy_count = 0
        for i, fname in enumerate(files, 1):
            self._set_status_async(f"Batch: {i}/{len(files)} - {fname}")
            path = os.path.join(in_dir, fname)
            try:
                channels = gwy_loader.load_gwy(path)
                field = channels.get(channel)
                if field is None:
                    # fall back to any channel containing the requested name
                    matches = [k for k in channels if channel.split(" ")[0] in k]
                    if matches:
                        field = channels[matches[0]]
                if field is None:
                    results.append(f"SKIP  {fname}: channel '{channel}' not found")
                    continue

                xy_factor, sp_units = spatial_scale(_unit_of(field, "si_unit_xy"))
                z_factor, z_units = z_scale(_unit_of(field, "si_unit_z"))
                data = field.data.astype(np.float64) * z_factor
                ny, nx = data.shape
                x_real = (field.xreal or nx) * xy_factor
                y_real = (field.yreal or ny) * xy_factor
                dx, dy = x_real / nx, y_real / ny

                processed = apply_pipeline(data, pipeline, dx, dy)

                # Recompute extents from the processed shape - operations
                # like crop change the image dimensions (pixel size is fixed)
                py, px = processed.shape
                x_real_out = px * dx
                y_real_out = py * dy

                base = os.path.splitext(fname)[0]
                np.save(os.path.join(out_dir, f"{base}_processed.npy"), processed)

                # Annotated image with axes, colorbar and scale bar
                fig = render_annotated_figure(
                    processed, x_real_out, y_real_out, f"{base} - {channel}",
                    sp_units, z_units,
                )
                fig.savefig(os.path.join(out_dir, f"{base}_processed.png"),
                            bbox_inches="tight")

                # Pure image (no labels) in the 'pure' subfolder
                save_pure_image(processed,
                                os.path.join(pure_dir, f"{base}_processed.png"),
                                x_real_out, y_real_out)

                # Collect the processed channel (back in SI units) into the
                # combined .gwy container
                field_out = gwy_loader.GwyDataField(
                    np.ascontiguousarray(processed / z_factor, dtype=np.float64),
                    xreal=px * float(field.xreal or nx) / nx,
                    yreal=py * float(field.yreal or ny) / ny,
                    si_unit_xy=_unit_of(field, "si_unit_xy") or None,
                    si_unit_z=_unit_of(field, "si_unit_z") or None,
                )
                gwy_out[f"/{gwy_count}/data"] = field_out
                gwy_out[f"/{gwy_count}/data/title"] = f"{base} - {channel}"
                gwy_count += 1

                results.append(f"OK    {fname}")
            except Exception as e:
                traceback.print_exc()
                results.append(f"ERROR {fname}: {e}")

        # Write all processed channels into one combined .gwy file
        if gwy_count:
            try:
                gwy_out.tofile(os.path.join(out_dir, "batch_processed.gwy"))
                results.append(
                    f"GWY   batch_processed.gwy ({gwy_count} channels)"
                )
            except Exception as e:
                traceback.print_exc()
                results.append(f"ERROR batch_processed.gwy: {e}")

        # Write a batch log
        log_path = os.path.join(out_dir, "batch_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Batch processing log - {datetime.now().isoformat()}\n")
            f.write(f"# Input folder: {in_dir}\n")
            f.write(f"# Channel: {channel}\n")
            f.write("# Pipeline:\n")
            for i, step in enumerate(pipeline, 1):
                f.write(f"#   {i}. {describe_step(*step)}\n")
            f.write("\n")
            f.write("\n".join(results) + "\n")

        ok = sum(1 for r in results if r.startswith("OK"))
        self._set_status_async(
            f"Batch done: {ok}/{len(files)} processed. Log: {log_path}"
        )
        self.after(0, lambda: self._log(
            f"Batch processed {ok}/{len(files)} files from {in_dir}"
        ))

    def _set_status_async(self, text):
        self.after(0, lambda: self.status_var.set(text))


if __name__ == "__main__":
    app = GwyProcessorGUI()
    app.mainloop()
