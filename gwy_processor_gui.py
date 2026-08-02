
"""
GUI front-end for gwy_processing.py

Provides an interactive Tkinter application to:
  - Load Gwyddion (.gwy) files via gwy_loader and select a channel
  - Apply processing steps from gwy_processing, each in its own dialog
    window with a live preview of the result AND the removed component:
      * Plane leveling (level_by_plane_fit)
      * Polynomial background removal with separate x/y orders
        (level_by_polynomial_xy)
      * Smart background (gwy_flatten): the same polynomial levelling,
        but fitted only to the pixels that are not sample. The cells,
        bubbles or pits are segmented first and excluded, so the fit
        cannot bend itself around them and leave the trenches and
        uneven surfaces that ordinary levelling does; after Wang et
        al., Beilstein J. Nanotechnol. 2018, 9, 975. The direction can
        be picked by the scan itself, rows and columns can be done in
        turn, and areas the threshold has no way of recognising can be
        dragged out by hand; after Zhang et al., arXiv:2602.04051
      * Align rows: median of differences / polynomial (align_rows)
      * Percentile range clipping (filter_by_percentile), re-editable:
        a clip opened right after another one edits that same step, so
        the range can be widened again and not only narrowed
      * FFT filtering (filter_by_2d_fft_mask): lowpass/highpass, circular
        notches, rectangles and straight bands combined in one dialog,
        with a large interactive spectrum (click to place cutoff/notches,
        drag to notch a rectangle), optional smooth mask edges and a
        zoom window comparing the image before and after the filter
      * Stripe removal (gwy_destripe): the multidirectional stripe
        remover of Liang et al. (2016), the variational general stripe
        remover of Rottmayer et al. (2025) or the spectrum denoiser
        DeStripe of Chen & Pellequer (2011), selected in the dialog,
        which shows only the chosen method's parameters, with a
        parameter sweep over any two of them
      * Scar removal (remove_scars)
      * Set baseline to zero (set_baseline_to_zero)
      * Two-way merge of the forward and backward scans (gwy_twoway):
        scanner lag / hysteresis alignment, parachuting-artifact
        detection and soft-min merging
  - Draw with any of Gwyddion's false-colour gradients (gwy_colormaps),
    chosen in the main window and followed by every preview and every
    saved image
  - Flip through a whole folder in the 'Quick view' tab: each .gwy file
    is shown with a plane subtracted and rows aligned (polynomial,
    order 2), one Next/Back step at a time
  - Put a whole folder on one colour scale in the 'Balanced view' tab
    (gwy_balance): every image is segmented into cells and substrate and
    measured at both, and the folder is reduced to a single range - so
    the same colour means the same thing in every image of a set - with
    a contact sheet, diagnostics, and an export of the whole folder as
    annotated PNGs, pure images and/or .gwy files
  - Keep a log of every change applied to the image
  - Undo and redo changes step by step (or reset to the original data)
  - Batch-process every .gwy file in a folder by replaying the
    current processing pipeline on the selected channel
  - Save the result as an image or back into a .gwy file, next to
    every other channel of the measurement

What is in this file is the front end only: windows, widgets, previews and
the state they are edited through. What the steps *are* - the operations,
their parameters, their validation and the sentences that describe them -
lives in gwy_ops, and getting a result back out lives in gwy_export. Neither
of those imports a GUI toolkit, so a batch script can replay a pipeline
without a screen and a front end written in something else would start from
them unchanged.

Run with:  python gwy_processor_gui.py
"""

import os
import re
import threading
import traceback
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector, SpanSelector

import gwy_loader
import gwy_processing as gp
import gwy_balance as gb
import gwy_colormaps as gcm
import gwy_flatten as gf
import gwy_destripe as gd
import gwy_twoway as gtw
from gwy_ops import (
    OPERATIONS, OPERATION_ROWS, apply_pipeline, describe_step,
    channel_view, pick_channel, spatial_scale, z_scale,
    aux_pairs_for, twoway_kwargs, twoway_param_relevant,
    _chen_kwargs, _fft_auto_items, _mdsr_kwargs, _natural_key,
    _smart_flatten, _unit_of,
)
from gwy_export import (
    render_annotated_figure, save_channel_to_gwy, save_pure_image,
)


# ---------------------------------------------------------------------------
# Zoom on a selected area (shared by the dialogs that preview images)
# ---------------------------------------------------------------------------

class ZoomWindow(tk.Toplevel):
    """A large side-by-side view of one region of the previewed images, so
    small features can be inspected close up. The region is picked by
    dragging a rectangle on one panel of the parent dialog; until one is
    picked the full images are shown. The views share one color scale so
    heights are directly comparable, and the window follows every preview
    update and display-leveling change."""

    def __init__(self, dialog, source="Forward"):
        super().__init__(dialog)
        self.title(f"Zoom - drag a rectangle on the {source} panel "
                   "to pick the area")
        self.geometry("1500x600")
        self.figure = Figure(figsize=(15, 5.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

    def show(self, panels, extent, subtitle, z_units):
        """panels: [(title, image), ...], all the same shape."""
        self.figure.clf()
        axes = np.atleast_1d(self.figure.subplots(1, len(panels),
                                                  sharex=True, sharey=True))
        allv = np.concatenate([img.ravel() for _, img in panels])
        v0, v1 = np.percentile(allv, [0.5, 99.5])
        if v1 <= v0:
            v1 = v0 + 1.0
        cmap = gcm.current()
        im = None
        for ax, (title, img) in zip(axes, panels):
            im = ax.imshow(img, origin="upper", cmap=cmap, extent=extent,
                           aspect="equal", vmin=v0, vmax=v1)
            ax.set_title(title, fontsize=10)
        self.figure.colorbar(im, ax=list(axes), fraction=0.03,
                             pad=0.02).set_label(z_units)
        self.figure.suptitle(subtitle, fontsize=10)
        self.canvas.draw()


class DestripeSweepWindow(tk.Toplevel):
    """
    A grid of stripe-removal results over two chosen parameters, for finding
    the setting that takes the stripes out without eating the structures.

    Each axis sweeps one parameter of the current method around the value
    set in the dialog. Gains are stepped by a factor (geometric: with factor
    2 and a 3x3 grid the rows are mu1/2, mu1, 2*mu1), counts, angles and
    thresholds by an increment. 'Same rate' keeps both axes on the same
    step, which for GSR's mu1/mu2 makes the diagonal the "scale both
    together" direction the paper describes; it is on by default only for
    two multiplied axes, since two incremented parameters need not share a
    unit.

    Nothing runs until 'Run' is pressed. The cells are then computed one at a
    time so the window fills in visibly instead of freezing. Every cell is
    computed on the WHOLE image and only then cropped for display, so the
    zoom area does not change the result. Clicking a cell copies its two
    values back into the dialog.
    """

    # Sweepable parameters: how to step them, and the default step.
    # "mul" multiplies (gains, widths - they span orders of magnitude),
    # "add" adds (angles and counts, where a factor makes no sense).
    SWEEP_STEPS = {
        "angle":      ("add", 5.0, "float"),
        "sigma":      ("mul", 2.0, "float"),
        "directions": ("mul", 2.0, "pow2"),
        "levels":     ("add", 1.0, "int"),
        "sigma_a":    ("mul", 1.5, "float"),
        "max_angle":  ("add", 15.0, "float"),
        "mu1":        ("mul", 2.0, "float"),
        "mu2":        ("mul", 2.0, "float"),
        "iterations": ("mul", 2.0, "int"),
        "cvar_k":     ("add", 0.5, "float"),
        "window":     ("add", 1.0, "int"),
        "density":    ("add", 0.05, "float"),
        "min_run":    ("add", 2.0, "int"),
    }
    DEFAULT_AXES = {"GSR": ("mu1", "mu2"), "MDSR": ("sigma", "levels"),
                    "DESTRIPE": ("cvar_k", "min_run")}

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.app = dialog.app
        self.title("Stripe removal parameter sweep - "
                   "click a cell to use its values")
        self.geometry("1250x900")

        self._after_id = None
        self._tasks = []
        self._cells = {}            # axes -> ((name, value), (name, value))
        self._vlim = None
        self._method = None
        self._syncing = False
        self._specs = {p["name"]: p
                       for p in OPERATIONS[dialog.op_key]["params"]}

        self._build_controls()
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.sync_method()
        self.status_var.set("Press Run to compute the grid")

    # ---- controls ----

    def _build_controls(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 0))
        top.pack(side=tk.TOP, fill=tk.X)
        bottom = ttk.Frame(self, padding=(8, 4, 8, 6))
        bottom.pack(side=tk.TOP, fill=tk.X)

        self.row_var = tk.StringVar()
        self.col_var = tk.StringVar()
        self.row_step_var = tk.DoubleVar(value=2.0)
        self.col_step_var = tk.DoubleVar(value=2.0)
        self.row_mode_var = tk.StringVar(value="x")
        self.col_mode_var = tk.StringVar(value="x")
        self.link_var = tk.BooleanVar(value=True)
        self.size_var = tk.IntVar(value=3)
        self.zoom_var = tk.BooleanVar(value=True)
        try:
            iters = int(self.dialog.vars["iterations"].get())
        except (tk.TclError, KeyError):
            iters = gd.GSR_DEFAULTS["iterations"]
        self.iter_var = tk.IntVar(value=iters)

        ttk.Label(top, text="Rows:").pack(side=tk.LEFT)
        self.row_combo = ttk.Combobox(top, textvariable=self.row_var, width=20,
                                      state="readonly")
        self.row_combo.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(top, text="step").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.row_mode_var,
                  width=2).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Entry(top, textvariable=self.row_step_var,
                  width=6).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(top, text="Columns:").pack(side=tk.LEFT)
        self.col_combo = ttk.Combobox(top, textvariable=self.col_var, width=20,
                                      state="readonly")
        self.col_combo.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(top, text="step").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.col_mode_var,
                  width=2).pack(side=tk.LEFT, padx=(4, 0))
        self.col_step_entry = ttk.Entry(top, textvariable=self.col_step_var,
                                        width=6)
        self.col_step_entry.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(top, text="Same rate", variable=self.link_var,
                        command=self._sync_link).pack(side=tk.LEFT)

        ttk.Label(bottom, text="Grid:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Spinbox(bottom, from_=2, to=5, width=3,
                    textvariable=self.size_var).pack(side=tk.LEFT)
        self._iter_frame = ttk.Frame(bottom)
        ttk.Label(self._iter_frame, text="Iterations:").pack(side=tk.LEFT,
                                                             padx=(12, 2))
        ttk.Entry(self._iter_frame, textvariable=self.iter_var,
                  width=7).pack(side=tk.LEFT)
        self._zoom_check = ttk.Checkbutton(bottom, text="Zoom area only",
                                           variable=self.zoom_var)
        self._zoom_check.pack(side=tk.LEFT, padx=12)
        ttk.Button(bottom, text="Run", command=self.run).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT,
                                                             padx=12)

        self.row_var.trace_add("write", lambda *a: self._on_axis_change("row"))
        self.col_var.trace_add("write", lambda *a: self._on_axis_change("col"))
        self.row_step_var.trace_add("write", lambda *a: self._mirror_step())

    def _label(self, name):
        return self._specs.get(name, {}).get("label", name)

    def sync_method(self):
        """Offer the parameters of the method the dialog is set to. Returns
        True when the method changed and the axes were reset."""
        method = self.dialog._method()
        if method == self._method:
            return False
        self._method = method
        names = [n for n in DestripeDialog.METHOD_PARAMS.get(method, ())
                 if n in self.SWEEP_STEPS]
        self._names = {self._label(n): n for n in names}
        row, col = self.DEFAULT_AXES.get(method, tuple(names[:2]))
        self._syncing = True
        try:
            self.row_combo["values"] = list(self._names)
            self.col_combo["values"] = list(self._names)
            self.row_var.set(self._label(row))
            self.col_var.set(self._label(col))
        finally:
            self._syncing = False
        self._on_axis_change("row")
        self._on_axis_change("col")

        # only GSR has an iteration count, and it is worth lowering for a
        # sweep without disturbing the dialog
        self._iter_frame.pack_forget()
        if method == "GSR":
            self._iter_frame.pack(side=tk.LEFT, before=self._zoom_check)
            try:                        # start from the dialog's count
                self.iter_var.set(int(self.dialog.vars["iterations"].get()))
            except (tk.TclError, KeyError):
                pass
        return True

    def _on_axis_change(self, which):
        """A different parameter on an axis brings its own kind of step."""
        if self._syncing:
            return
        var = self.row_var if which == "row" else self.col_var
        name = getattr(self, "_names", {}).get(var.get())
        if name is None:
            return
        mode, step, _ = self.SWEEP_STEPS[name]
        self._syncing = True
        try:
            if which == "row":
                self.row_mode_var.set("x" if mode == "mul" else "+")
                self.row_step_var.set(step)
            else:
                self.col_mode_var.set("x" if mode == "mul" else "+")
                self.col_step_var.set(step)
        finally:
            self._syncing = False
        # a new axis parameter, so the link goes back to its default
        self.link_var.set(self._axis_modes() == ("mul", "mul"))
        self._sync_link()

    def _axis_modes(self):
        """How each axis steps, or (None, None) before the axes are set."""
        row = getattr(self, "_names", {}).get(self.row_var.get())
        col = getattr(self, "_names", {}).get(self.col_var.get())
        if row is None or col is None:
            return None, None
        return self.SWEEP_STEPS[row][0], self.SWEEP_STEPS[col][0]

    def _modes_match(self):
        modes = self._axis_modes()
        return modes[0] is not None and modes[0] == modes[1]

    def _sync_link(self):
        """'Same rate' only means something when both axes step the same
        way; otherwise the column keeps its own step. It is on by default
        only for two multiplied axes - two gains scaled together is the
        comparison it exists for, whereas two incremented parameters (an
        angle and a pixel count, say) do not share a unit."""
        same = self._modes_match()
        if not same:
            self.link_var.set(False)
        linked = same and self.link_var.get()
        self.col_step_entry.state(["disabled"] if linked else ["!disabled"])
        self._mirror_step()

    def _mirror_step(self):
        if self._syncing or not self.link_var.get() or not self._modes_match():
            return
        self._syncing = True
        try:
            self.col_step_var.set(self.row_step_var.get())
        except tk.TclError:
            pass
        finally:
            self._syncing = False

    # ---- the sweep ----

    def _values(self, center, count, step, name):
        """`count` values around `center`, one step apart."""
        start = -(count - 1) / 2.0
        mode = self.SWEEP_STEPS[name][0]
        return [self._cast(center * step ** (start + k) if mode == "mul"
                           else center + step * (start + k), name)
                for k in range(count)]

    def _cast(self, value, name):
        """Round to what the parameter accepts and keep it in range."""
        kind = self.SWEEP_STEPS[name][2]
        if kind == "pow2":
            value = 2.0 ** max(0, int(round(np.log2(max(value, 1.0)))))
        if kind in ("int", "pow2"):
            value = int(round(value))
        spec = self._specs.get(name, {})
        if spec.get("min") is not None:
            value = max(value, spec["min"])
        if spec.get("max") is not None:
            value = min(value, spec["max"])
        return int(value) if kind in ("int", "pow2") else float(value)

    def run(self):
        """(Re)start the sweep with the current settings."""
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.sync_method()
        params = self.dialog.get_params()
        if params is None:
            self.status_var.set("Fix the dialog parameters first")
            return
        row = self._names.get(self.row_var.get())
        col = self._names.get(self.col_var.get())
        if row is None or col is None:
            self.status_var.set("Pick a parameter for each axis")
            return
        if row == col:
            self.status_var.set("Pick two different parameters")
            return
        try:
            n = max(2, min(5, int(self.size_var.get())))
            steps = {row: float(self.row_step_var.get()),
                     col: float(self.col_step_var.get())}
        except tk.TclError:
            return
        for name, step in steps.items():
            if self.SWEEP_STEPS[name][0] == "mul" and step <= 1.0:
                self.status_var.set(f"The step factor for {name} must be "
                                    f"larger than 1")
                return
            if step <= 0:
                self.status_var.set(f"The step for {name} must be positive")
                return
        if self._method == "GSR":
            try:
                params["iterations"] = max(1, int(self.iter_var.get()))
            except tk.TclError:
                pass

        self._row, self._col = row, col
        self._params = params
        self._row_values = self._values(float(params[row]), n, steps[row], row)
        self._col_values = self._values(float(params[col]), n, steps[col], col)

        data = self.dialog._base_data()
        self._slices = (self.dialog._zoom_slices(data.shape)
                        if self.zoom_var.get() else None)
        crop = self._crop(data)
        v0, v1 = np.percentile(crop, [0.5, 99.5])
        self._vlim = (v0, v1 if v1 > v0 else v0 + 1.0)

        self.figure.clf()
        self._cells = {}
        axes = np.atleast_2d(self.figure.subplots(n, n, sharex=True,
                                                  sharey=True))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                ax.set_title(f"{row}={self._row_values[i]:.4g}, "
                             f"{col}={self._col_values[j]:.4g}", fontsize=8)
                ax.tick_params(labelsize=7)
                self._cells[ax] = ((row, self._row_values[i]),
                                   (col, self._col_values[j]))
        where = "zoom area" if self._slices is not None else "full image"
        sym = {"mul": "x", "add": "+"}
        self.figure.suptitle(
            f"{self._method} sweep - {where}, "
            f"stripe angle {float(params.get('angle', 0.0)):g} deg"
            + (f", {params['iterations']} iterations"
               if self._method == "GSR" else "")
            + f"   (rows: {row} {sym[self.SWEEP_STEPS[row][0]]}{steps[row]:g}, "
              f"columns: {col} {sym[self.SWEEP_STEPS[col][0]]}{steps[col]:g})",
            fontsize=10)
        self.canvas.draw()

        self._axes = axes
        self._tasks = [(i, j) for i in range(n) for j in range(n)]
        self._data = data
        self._after_id = self.after(10, self._step)

    def _crop(self, image):
        if self._slices is None:
            return image
        rows, cols = self._slices
        return image[rows, cols]

    def _step(self):
        """Compute and draw one cell, then queue the next."""
        self._after_id = None
        if not self._tasks:
            self.status_var.set("Done - click a cell to use its values")
            return
        i, j = self._tasks.pop(0)
        total = len(self._row_values) * len(self._col_values)
        self.status_var.set(f"Computing {total - len(self._tasks)}/{total} ...")
        self.update_idletasks()

        op = OPERATIONS[self.dialog.op_key]
        params = dict(self._params)
        params[self._row] = self._row_values[i]
        params[self._col] = self._col_values[j]
        ax = self._axes[i, j]
        error = op["validate"](params) if op.get("validate") else None
        if error:
            ax.text(0.5, 0.5, error, ha="center", va="center", fontsize=8,
                    transform=ax.transAxes)
        else:
            # the whole image goes through the method; the crop is for
            # display only, so the zoom area cannot change the result
            result = op["func"](self._data, params, self.app.dx, self.app.dy)
            ax.imshow(self._crop(result), origin="upper",
                      cmap=gcm.current(), extent=self._extent(),
                      aspect="equal", vmin=self._vlim[0], vmax=self._vlim[1])
        self.canvas.draw()
        self._after_id = self.after(1, self._step)

    def _extent(self):
        app = self.app
        if self._slices is None:
            return (0, app.x_real, 0, app.y_real)
        rows, cols = self._slices
        ny = self._data.shape[0]
        return (cols.start * app.dx, cols.stop * app.dx,
                (ny - rows.stop) * app.dy, (ny - rows.start) * app.dy)

    # ---- picking a cell ----

    def _on_click(self, event):
        cell = self._cells.get(event.inaxes)
        if cell is None:
            return
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return                      # pan/zoom tool active
        try:
            for name, value in cell:
                var = self.dialog.vars.get(name)
                if var is None:
                    continue
                spec = self._specs.get(name, {})
                if spec.get("type") == "choice":
                    var.set(str(int(value)))
                elif spec.get("type") == "int":
                    var.set(int(value))
                else:
                    var.set(round(float(value), 6))
        except tk.TclError:
            return
        self.status_var.set("Using " + ", ".join(f"{n}={v:.4g}"
                                                 for n, v in cell))

    def destroy(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        super().destroy()


class ZoomAreaMixin:
    """Drag-to-pick an area on one preview panel and inspect it big in a
    `ZoomWindow`.

    The host dialog must provide `self.app` and `self.canvas`, call
    `_init_zoom()` in its constructor, and on every draw call
    `_attach_zoom_selector(ax)` on the panel to drag on, `_mark_zoom_rect()`
    on the panels that should outline the area, and `_update_zoom_window()`.
    It supplies the images through `_zoom_panels()` and re-renders itself
    through `_redraw_zoom_source()`.
    """

    ZOOM_SOURCE = "Forward"          # name of the panel carrying the selector

    def _init_zoom(self):
        self._zoom_rect = None       # (x0, x1, y0, y1) in physical units
        self._zoom_win = None
        self._zoom_selector = None

    # ---- to be provided by the dialog ----

    def _zoom_panels(self):
        """`([(title, image), ...], subtitle_tag)`, or None when there is
        nothing to show yet. All images must have the same shape."""
        raise NotImplementedError

    def _redraw_zoom_source(self):
        """Re-render the dialog itself, so the picked area gets outlined."""
        raise NotImplementedError

    # ---- area selection ----

    def open_zoom_window(self):
        """Open (or raise) the large zoom view of the selected area."""
        if self._zoom_win is None or not self._zoom_win.winfo_exists():
            self._zoom_win = ZoomWindow(self, self.ZOOM_SOURCE)
        else:
            self._zoom_win.lift()
        self._update_zoom_window()

    def _attach_zoom_selector(self, ax):
        """Drag-to-select on the source panel; the rectangle is shown big in
        the zoom window. Re-created on every draw (figure was cleared)."""
        try:
            self._zoom_selector = RectangleSelector(
                ax, self._on_zoom_select, useblit=True, button=[1],
                props=dict(fill=False, edgecolor="red", linestyle="--"),
            )
        except TypeError:
            # Older matplotlib uses `rectprops`
            self._zoom_selector = RectangleSelector(
                ax, self._on_zoom_select, useblit=True, button=[1],
                rectprops=dict(fill=False, edgecolor="red", linestyle="--"),
            )

    def _on_zoom_select(self, eclick, erelease):
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return              # pan/zoom tool active - not a selection
        coords = (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        if any(c is None for c in coords):
            return
        x0, x1 = sorted(coords[:2])
        y0, y1 = sorted(coords[2:])
        if (x1 - x0) < 2 * self.app.dx or (y1 - y0) < 2 * self.app.dy:
            return              # a click, not a drag
        self._zoom_rect = (x0, x1, y0, y1)
        self.open_zoom_window()
        self._redraw_zoom_source()

    def _zoom_slices(self, shape):
        """The pixel rows/columns of the current zoom rectangle, or None for
        the full image (images are drawn with origin='upper', so pixel row 0
        sits at the TOP of the physical extent)."""
        if self._zoom_rect is None:
            return None
        ny, nx = shape
        dx, dy = self.app.dx, self.app.dy
        x0, x1, y0, y1 = self._zoom_rect
        ix0 = max(0, int(np.floor(x0 / dx)))
        ix1 = min(nx, int(np.ceil(x1 / dx)))
        iy0 = max(0, int(np.floor(ny - y1 / dy)))
        iy1 = min(ny, int(np.ceil(ny - y0 / dy)))
        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            return None
        return slice(iy0, iy1), slice(ix0, ix1)

    def _mark_zoom_rect(self, *axes):
        """Outline the zoomed area on the dialog's own image panels."""
        if self._zoom_rect is None:
            return
        x0, x1, y0, y1 = self._zoom_rect
        for ax in axes:
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="red", lw=1.2))

    def _update_zoom_window(self):
        if self._zoom_win is None or not self._zoom_win.winfo_exists():
            return
        panels = self._zoom_panels()
        if panels is None:
            return
        panels, tag = panels
        images = [img for _, img in panels]
        dx, dy = self.app.dx, self.app.dy
        ny, nx = images[0].shape
        sl = self._zoom_slices(images[0].shape)
        if sl is None:
            extent = (0, nx * dx, 0, ny * dy)
            where = (f"full image (drag on the {self.ZOOM_SOURCE} panel "
                     f"to pick an area)")
        else:
            rows, cols = sl
            images = [img[rows, cols] for img in images]
            extent = (cols.start * dx, cols.stop * dx,
                      (ny - rows.stop) * dy, (ny - rows.start) * dy)
            where = (f"area {cols.stop - cols.start}x{rows.stop - rows.start}"
                     f" px at ({extent[0]:.3g}, {extent[2]:.3g}) "
                     f"{self.app.spatial_units}")
        self._zoom_win.show([(t, img) for (t, _), img in zip(panels, images)],
                            extent, f"{where}{tag}", self.app.z_units)

    def destroy(self):
        win = getattr(self, "_zoom_win", None)
        if win is not None and win.winfo_exists():
            win.destroy()
        super().destroy()


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
        self.param_labels = {}

        if not self.spec["params"]:
            ttk.Label(frame, text="No parameters for this operation.").pack(side=tk.LEFT)

        for p in self.spec["params"]:
            self._make_param(frame, p)

        self.status_var = tk.StringVar(value="")
        self._status_label = ttk.Label(frame, textvariable=self.status_var,
                                       foreground="red")
        self._status_label.pack(side=tk.RIGHT, padx=8)

    def _make_param(self, parent, p):
        """Build the label and the widget for one parameter inside `parent`.

        Kept separate from `_build_params` so a dialog with more parameters
        than fit on one line can put them in several frames; `_show_params`
        works either way, because packing a widget again puts it back in the
        frame it was built in."""
        label = ttk.Label(parent, text=p["label"] + ":")
        label.pack(side=tk.LEFT, padx=(8, 2))
        self.param_labels[p["name"]] = label
        if p["type"] == "int":
            var = tk.IntVar(value=p["default"])
            widget = ttk.Spinbox(
                parent, from_=p.get("min", 0), to=p.get("max", 100),
                width=5, textvariable=var,
            )
        elif p["type"] == "float":
            var = tk.DoubleVar(value=p["default"])
            widget = ttk.Entry(parent, textvariable=var, width=8)
        elif p["type"] == "choice":
            var = tk.StringVar(value=p["default"])
            widget = ttk.Combobox(
                parent, textvariable=var, values=p["values"],
                state="readonly", width=max(len(v) for v in p["values"]) + 2,
            )
        elif p["type"] == "bool":
            var = tk.BooleanVar(value=p["default"])
            widget = ttk.Checkbutton(parent, variable=var)
        else:
            raise ValueError(f"Unknown param type: {p['type']}")
        widget.pack(side=tk.LEFT)

        var.trace_add("write", self._on_param_change)
        self.vars[p["name"]] = var
        self.param_widgets[p["name"]] = widget
        return widget

    def _show_params(self, names):
        """Show only these parameter widgets, keeping the declared order.
        Used by dialogs whose parameters depend on a selected method."""
        for name, label in self.param_labels.items():
            label.pack_forget()
            self.param_widgets[name].pack_forget()
        self._status_label.pack_forget()
        for p in self.spec["params"]:
            if p["name"] in names:
                self.param_labels[p["name"]].pack(side=tk.LEFT, padx=(8, 2))
                self.param_widgets[p["name"]].pack(side=tk.LEFT)
        self._status_label.pack(side=tk.RIGHT, padx=8)

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

    def _base_data(self):
        """The image the operation is previewed on. Normally the current
        data; a dialog that re-edits its own last step overrides this with
        the image from before that step."""
        return self.app.data

    def _toolbar_busy(self):
        """True while the navigation toolbar's pan or zoom tool is armed, so
        a dialog that reads clicks on its own panels can stay out of the way
        of one that is only meant to move the view."""
        toolbar = getattr(self.canvas, "toolbar", None)
        return toolbar is not None and getattr(toolbar, "mode", "")

    def _compute(self, params):
        """Run the operation on the base image."""
        return self.spec["func"](self._base_data(), params,
                                 self.app.dx, self.app.dy)

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
        removed = self._base_data() - result
        self._draw(result, removed)

    def _draw(self, result, removed):
        app = self.app
        extent = (0, app.x_real, 0, app.y_real)
        self.figure.clf()
        ax1, ax2 = self.figure.subplots(1, 2)

        im1 = ax1.imshow(
            result, origin="upper", cmap=gcm.current(),
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


class SmartLevelDialog(OperationDialog):
    """
    Background subtraction that leaves the sample out of the fit
    (gwy_flatten, after Wang et al. 2018).

    The window answers the two questions this operation raises. What was
    called a feature? - the mask is drawn on the result as a contour. And
    what did masking actually change? - the third panel is the difference
    against the same fit run over every pixel, which is the ordinary
    background subtraction sitting two buttons up. Where that panel is flat
    the mask made no difference; where it is bright, that is the amount of
    the sample that the ordinary fit was subtracting from itself.

    The parameters are in three rows because there are three steps and they
    fail in different ways: too much or too little threshold shows up in the
    contour, an outline stopping short of the foot shows up as a trench
    beside each feature, and a bad fit shows up in the background panel.

    A fourth row is the manual override from Zhang et al.: DRAG on the result
    panel to exclude that rectangle from the fit whatever the threshold
    thinks of it, and right-click one to take it back. It is there for what a
    threshold has no way of recognising - a step edge, a piece of debris, the
    corner where the tip crashed - and it feeds the segmentation too, not
    only the fit, so the excluded area cannot bend the image the features are
    looked for on either.
    """

    GROUPS = (
        ("1. Find the features",
         ("detect", "threshold", "feature_size", "neighbourhood",
          "sensitivity")),
        ("2. Take the outline out to the foot of each one",
         ("min_area", "expand", "edge", "grow")),
        ("3. Fit the background to what is left",
         ("fit", "order", "window", "passes")),
    )
    ADAPTIVE_ONLY = ("neighbourhood", "sensitivity")
    OTSU_ONLY = ("feature_size",)

    def __init__(self, app, op_key="smart_level"):
        self.result = None
        self.excluded = []           # [x0, x1, y0, y1] in physical units
        self._area_selector = None
        super().__init__(app, op_key)
        self.geometry("1180x760")
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _build_params(self):
        outer = ttk.Frame(self, padding=(8, 8, 8, 0))
        outer.pack(side=tk.TOP, fill=tk.X)
        self.params_frame = outer
        self.param_widgets = {}
        self.param_labels = {}

        frames = {}
        for title, names in self.GROUPS:
            frame = ttk.LabelFrame(outer, text=title, padding=4)
            frame.pack(fill=tk.X, pady=(0, 4))
            for name in names:
                frames[name] = frame

        for p in self.spec["params"]:
            self._make_param(frames[p["name"]], p)

        hand = ttk.LabelFrame(outer, text="4. Anything the threshold missed",
                              padding=4)
        hand.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(hand, text="Drag on the result panel to keep an area out "
                             "of the fit; right-click one to take it back.").pack(
            side=tk.LEFT, padx=(8, 2))
        self.excluded_var = tk.StringVar(value="none")
        ttk.Label(hand, textvariable=self.excluded_var).pack(side=tk.LEFT,
                                                             padx=(8, 2))
        ttk.Button(hand, text="Clear", command=self.clear_excluded).pack(
            side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="")
        self._status_label = ttk.Label(outer, textvariable=self.status_var,
                                       foreground="red", wraplength=1100)
        self._status_label.pack(side=tk.TOP, fill=tk.X)

        self.vars["threshold"].trace_add("write", lambda *a: self._sync_threshold())
        self._sync_threshold()

    def _sync_threshold(self):
        """Otsu takes one threshold for the whole image, so the adaptive
        neighbourhood and offset have nothing to act on - and the other way
        round for the blur the single threshold is taken on."""
        adaptive = self.vars["threshold"].get() == "adaptive"
        for name in self.ADAPTIVE_ONLY:
            self.param_widgets[name].state(
                ["!disabled"] if adaptive else ["disabled"])
        for name in self.OTSU_ONLY:
            self.param_widgets[name].state(
                ["disabled"] if adaptive else ["!disabled"])

    # ---- areas excluded by hand ----

    def get_params(self):
        params = super().get_params()
        if params is not None:
            params["exclude"] = [list(r) for r in self.excluded]
        return params

    def clear_excluded(self):
        self.excluded = []
        self.update_preview()

    def _attach_area_selector(self, ax):
        """Drag-to-exclude on the result panel. Re-created on every draw
        (the figure was cleared)."""
        try:
            self._area_selector = RectangleSelector(
                ax, self._on_area_select, useblit=True, button=[1],
                props=dict(fill=False, edgecolor="#ffd000", linestyle="--"),
            )
        except TypeError:
            # Older matplotlib uses `rectprops`
            self._area_selector = RectangleSelector(
                ax, self._on_area_select, useblit=True, button=[1],
                rectprops=dict(fill=False, edgecolor="#ffd000",
                               linestyle="--"),
            )

    def _on_area_select(self, eclick, erelease):
        if self._toolbar_busy():
            return
        coords = (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        if any(c is None for c in coords):
            return
        x0, x1 = sorted(coords[:2])
        y0, y1 = sorted(coords[2:])
        if (x1 - x0) < 2 * self.app.dx or (y1 - y0) < 2 * self.app.dy:
            return              # a click, not a drag
        self.excluded.append([x0, x1, y0, y1])
        self.update_preview()

    def _on_click(self, event):
        """Right-click takes back the area under the pointer, or if there is
        none there, the nearest one."""
        if event.button != 3 or event.inaxes is None or self._toolbar_busy():
            return
        if not self.excluded or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        inside = [i for i, (x0, x1, y0, y1) in enumerate(self.excluded)
                  if x0 <= x <= x1 and y0 <= y <= y1]
        if inside:
            # the smallest one, so a rectangle drawn inside another is
            # reachable
            pick = min(inside, key=lambda i: ((self.excluded[i][1]
                                               - self.excluded[i][0])
                                              * (self.excluded[i][3]
                                                 - self.excluded[i][2])))
        else:
            pick = min(range(len(self.excluded)),
                       key=lambda i: np.hypot(
                           (self.excluded[i][0] + self.excluded[i][1]) / 2 - x,
                           (self.excluded[i][2] + self.excluded[i][3]) / 2 - y))
        del self.excluded[pick]
        self.update_preview()

    def _compute(self, params):
        data = self._base_data()
        self.result = _smart_flatten(data, params, self.app.dx, self.app.dy)
        # the same fit with nothing masked out: the ordinary levelling, kept
        # for the comparison panel. `auto` has settled on a direction by now,
        # so the comparison is between the same two fits.
        self.plain = data - gf.fit_background(
            data, np.zeros(data.shape, dtype=bool),
            fit=self.result["fit"], order=params["order"],
            window=params["window"])
        return self.result["data"]

    def _report(self):
        res = self.result
        notes = [f"mask {100 * res['coverage']:.0f}% of the frame"]
        if self.vars["fit"].get() == "auto":
            notes.append(f"the scan lines run {res['fit']}, so the fit went "
                         f"that way")
        if res["covered"] < 1.0:
            notes.append(f"sliding fit reached {100 * res['covered']:.0f}%, "
                         f"the rest from the whole-line fit")
        if res["starved"] > 0:
            notes.append(f"{100 * res['starved']:.0f}% of lines had no "
                         f"background left and were interpolated")
        if res["reduced"] > 0:
            notes.append(f"{100 * res['reduced']:.0f}% of lines could not "
                         f"support order {self.vars['order'].get()} and were "
                         f"fitted lower")
        self.excluded_var.set(
            "none" if not self.excluded
            else f"{len(self.excluded)} area"
                 f"{'s' if len(self.excluded) > 1 else ''} excluded by hand")
        warn = ""
        if res["coverage"] > 0.85:
            warn = ("  -- almost nothing is left to fit; loosen the "
                    "threshold or check the mask.")
        elif res["coverage"] < 0.005:
            warn = ("  -- nothing was masked, so this is the ordinary "
                    "background subtraction.")
        elif res["reduced"] > 0.25:
            warn = ("  -- most lines cannot carry this order, so neighbouring "
                    "lines are being levelled by different curves. Lower the "
                    "order.")
        self.status_var.set(", ".join(notes) + warn)
        self._status_label.configure(
            foreground="red" if warn else "#404040")

    def _draw(self, result, removed):
        app = self.app
        extent = (0, app.x_real, 0, app.y_real)
        self.figure.clf()
        ax1, ax2, ax3 = self.figure.subplots(1, 3)

        lo, hi = np.percentile(result, [0.5, 99.5])
        im1 = ax1.imshow(result, origin="upper", cmap=gcm.current(),
                         extent=extent, aspect="equal", vmin=lo, vmax=hi)
        if self.result["mask"].any():
            ax1.contour(self.result["mask"].astype(float), [0.5],
                        colors="#ff30c0", linewidths=0.8, extent=extent,
                        origin="upper")
        for x0, x1, y0, y1 in self.excluded:
            ax1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="#ffd000", lw=1.2))
        ax1.set_title("Result, with the excluded area outlined")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self._attach_area_selector(ax1)
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        im2 = ax2.imshow(removed, origin="upper", cmap="viridis",
                         extent=extent, aspect="equal")
        ax2.set_title(self.spec["removed_label"])
        self.figure.colorbar(im2, ax=ax2, fraction=0.046).set_label(app.z_units)

        gained = result - self.plain
        span = float(np.abs(gained).max()) or 1.0
        im3 = ax3.imshow(gained, origin="upper", cmap="coolwarm",
                         extent=extent, aspect="equal", vmin=-span, vmax=span)
        ax3.set_title("What the mask changed\n(vs the same fit on every pixel)")
        self.figure.colorbar(im3, ax=ax3, fraction=0.046).set_label(app.z_units)

        for ax in (ax1, ax2, ax3):
            ax.set_xlabel(f"x ({app.spatial_units})")
        self.figure.tight_layout()
        self.canvas.draw()
        self._report()


class FFTFilterDialog(ZoomAreaMixin, OperationDialog):
    """
    Combined FFT filter dialog: an optional radial lowpass/highpass and
    notch filtering of specific periodic signals, all applied as ONE
    frequency-domain mask in a single inverse transform.

    The FFT magnitude spectrum fills the left half of the window
    (interactive); the filtered result and the removed component stack on
    the right. What a left-click on the spectrum does is chosen with the
    "Click sets" selector:

      * cutoff - set the pass-filter cutoff to the clicked radius
      * circle notch - notch a circular patch at the clicked frequency
      * vertical / horizontal band - notch a straight stripe
        (single-frequency interference along one scan axis)

    DRAGGING on the spectrum notches the dragged rectangle - for noise
    that fills an extended patch of the spectrum rather than a point or
    a full line. Right-click removes the nearest notch/rectangle/band.
    Everything is applied symmetrically at +/-f, so marking one member
    of a conjugate pair is enough.

    'Auto-detect' finds all regions of excess spectral power outside the
    protected center: compact ones become circular notches, extended
    ones rectangles. 'Edge smoothing' softens the whole mask with a
    Gaussian roll-off so the filters do not ring.

    'Zoom window...' opens the before/after images side by side and large;
    dragging on the result panel picks the area to inspect there, so it can
    be checked that the filter removed the noise and not the topography.
    """

    ZOOM_SOURCE = "result"

    def __init__(self, app, op_key="fft_filter"):
        self.notches = []       # list of [fx, fy] circular notches
        self.rects = []         # list of [fx, fy, wx, wy] rectangle notches
        self.x_bands = []       # list of fx centers (vertical band notches)
        self.y_bands = []       # list of fy centers (horizontal band notches)
        self._spectrum = None
        self._spec_ax = None
        self._spec_selector = None
        self._press = None      # left-button press position (click vs drag)
        self._last_result = None
        self._init_zoom()
        super().__init__(app, op_key)
        self.geometry("1500x850")
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # ---- extra controls ----

    def _build_params(self):
        super()._build_params()
        self.vars["mode"].trace_add("write", lambda *a: self._sync_cutoff_state())
        self._sync_cutoff_state()
        btns = ttk.Frame(self, padding=(8, 0, 8, 4))
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="Auto-detect", command=self.auto_detect).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btns, text="Clear notches", command=self.clear_notches).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btns, text="Zoom window...",
                   command=self.open_zoom_window).pack(side=tk.LEFT, padx=2)
        ttk.Label(btns, text="Click sets:").pack(side=tk.LEFT, padx=(12, 2))
        self.click_mode_var = tk.StringVar(value="circle notch")
        ttk.Combobox(
            btns, textvariable=self.click_mode_var,
            values=["cutoff", "circle notch", "vertical band", "horizontal band"],
            state="readonly", width=15,
        ).pack(side=tk.LEFT)
        ttk.Label(
            btns,
            text="Left-click: set/add  |  Drag: notch rectangle  |  "
                 "Right-click: remove nearest",
        ).pack(side=tk.LEFT, padx=12)

    def _sync_cutoff_state(self):
        """The cutoff entry only matters while a pass filter is selected."""
        try:
            on = self.vars["mode"].get() in ("lowpass", "highpass")
        except tk.TclError:
            return
        self.param_widgets["cutoff"].state(["!disabled"] if on else ["disabled"])

    def get_params(self):
        params = super().get_params()
        if params is not None:
            params["notches"] = [list(n) for n in self.notches]
            params["rects"] = [list(r) for r in self.rects]
            params["x_bands"] = list(self.x_bands)
            params["y_bands"] = list(self.y_bands)
        return params

    # ---- notch management ----

    def auto_detect(self):
        params = self._validated_params()
        if params is None:
            return
        self.notches, self.rects = _fft_auto_items(
            self.app.data, params, self.app.dx, self.app.dy
        )
        self.status_var.set(
            f"{len(self.notches)} peaks + {len(self.rects)} rectangles detected"
        )
        self.update_preview()

    def clear_notches(self):
        self.notches = []
        self.rects = []
        self.x_bands = []
        self.y_bands = []
        self.update_preview()

    # ---- zoom on a selected area (see ZoomAreaMixin) ----

    def _zoom_panels(self):
        """Before/after the filter, so the zoom window shows directly what
        the mask took out of the image."""
        if self._last_result is None:
            return None
        params = self.get_params()
        tag = f" - {describe_step(self.op_key, params)}" if params else ""
        return ([("Before filtering", self._base_data()),
                 ("After filtering", self._last_result)], tag)

    def _redraw_zoom_source(self):
        self.update_preview()

    def _on_click(self, event):
        """Press handler: remember left presses (to tell clicks from
        rectangle drags on release) and do right-click removal."""
        if event.inaxes is not self._spec_ax or event.xdata is None:
            return
        if self._toolbar_busy():
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.button == 1:
            self._press = (event.x, event.y, x, y)
        elif event.button == 3:
            # find the globally nearest item (circle, rectangle, v-band or
            # h-band), considering mirrored counterparts too
            best = None  # (distance, list, index)
            for i, (fx, fy) in enumerate(self.notches):
                d = min(np.hypot(x - fx, y - fy), np.hypot(x + fx, y + fy))
                if best is None or d < best[0]:
                    best = (d, self.notches, i)
            for i, (fx, fy, _, _) in enumerate(self.rects):
                d = min(np.hypot(x - fx, y - fy), np.hypot(x + fx, y + fy))
                if best is None or d < best[0]:
                    best = (d, self.rects, i)
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

    def _on_release(self, event):
        """A left press+release that barely moved is a click; dispatch the
        selected click action. Real drags are handled by the rectangle
        selector instead."""
        if event.button != 1 or self._press is None:
            return
        px, py, x, y = self._press
        self._press = None
        if event.x is None or abs(event.x - px) > 3 or abs(event.y - py) > 3:
            return                                   # drag -> rectangle
        mode = self.click_mode_var.get()
        if mode == "cutoff":
            try:
                pass_on = self.vars["mode"].get() in ("lowpass", "highpass")
            except tk.TclError:
                pass_on = False
            if not pass_on:
                self.status_var.set(
                    "Pick lowpass/highpass first, then click to set the cutoff"
                )
                return
            self.vars["cutoff"].set(round(float(np.hypot(x, y)), 3))
            # trace on the variable triggers the debounced preview update
            return
        if mode == "vertical band":
            self.x_bands.append(abs(x))
        elif mode == "horizontal band":
            self.y_bands.append(abs(y))
        else:
            self.notches.append([x, y])
        self.update_preview()

    def _attach_spec_selector(self, ax):
        """Drag-to-notch-a-rectangle on the spectrum. Re-created on every
        draw (the figure was cleared)."""
        try:
            self._spec_selector = RectangleSelector(
                ax, self._on_spec_select, useblit=True, button=[1],
                props=dict(fill=False, edgecolor="red", linestyle="--"),
            )
        except TypeError:
            # Older matplotlib uses `rectprops`
            self._spec_selector = RectangleSelector(
                ax, self._on_spec_select, useblit=True, button=[1],
                rectprops=dict(fill=False, edgecolor="red", linestyle="--"),
            )

    def _on_spec_select(self, eclick, erelease):
        if self._toolbar_busy():
            return
        coords = (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        if any(c is None for c in coords):
            return
        x0, x1 = sorted(coords[:2])
        y0, y1 = sorted(coords[2:])
        ny, nx = self.app.data.shape
        dfx = 1.0 / (nx * self.app.dx)
        dfy = 1.0 / (ny * self.app.dy)
        if (x1 - x0) < 2 * dfx or (y1 - y0) < 2 * dfy:
            return              # a click, not a drag (release handles it)
        self.rects.append([(x0 + x1) / 2.0, (y0 + y1) / 2.0,
                           x1 - x0, y1 - y0])
        self.update_preview()

    # ---- drawing ----

    def _ensure_spectrum(self):
        # The spectrum depends only on the data (no windowing), so it is
        # computed once per dialog.
        if self._spectrum is None:
            self._spectrum = gp.get_2d_fft_magnitude(
                self.app.data, dx=self.app.dx, dy=self.app.dy
            )

    def _draw(self, result, removed):
        app = self.app
        self._last_result = result
        self._ensure_spectrum()
        mag, freq_extent = self._spectrum
        extent = (0, app.x_real, 0, app.y_real)

        self.figure.clf()
        # Large interactive spectrum on the left, result and removed
        # component stacked on the right.
        gs = self.figure.add_gridspec(2, 2, width_ratios=[1.6, 1.0])
        ax0 = self.figure.add_subplot(gs[:, 0])
        ax1 = self.figure.add_subplot(gs[0, 1])
        ax2 = self.figure.add_subplot(gs[1, 1])

        im0 = ax0.imshow(
            mag, origin="upper", cmap="viridis",
            extent=freq_extent, aspect="equal",
        )
        n_items = (len(self.notches) + len(self.rects)
                   + len(self.x_bands) + len(self.y_bands))
        ax0.set_title(f"FFT spectrum - {n_items} notches/rects/bands")
        ax0.set_xlabel(f"fx (1/{app.spatial_units})")
        ax0.set_ylabel(f"fy (1/{app.spatial_units})")
        self.figure.colorbar(im0, ax=ax0, fraction=0.046).set_label("dB")
        self._spec_ax = ax0

        try:
            pass_on = self.vars["mode"].get() in ("lowpass", "highpass")
            cutoff = self.vars["cutoff"].get()
            radius = self.vars["radius"].get()
            protect = self.vars["protect_radius"].get()
        except tk.TclError:
            pass_on, cutoff, radius, protect = False, None, None, None
        if pass_on and cutoff:
            ax0.add_patch(Circle((0, 0), cutoff, fill=False,
                                 color="red", linewidth=1.5))
        if protect:
            ax0.add_patch(Circle((0, 0), protect, fill=False, color="lime",
                                 linewidth=1.2, linestyle="--"))
        for fx, fy, wx, wy in self.rects:
            ax0.add_patch(Rectangle((fx - wx / 2, fy - wy / 2), wx, wy,
                                    fill=False, color="red", linewidth=1.2))
            ax0.add_patch(Rectangle((-fx - wx / 2, -fy - wy / 2), wx, wy,
                                    fill=False, color="red", linewidth=1.0,
                                    linestyle=":"))
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
            result, origin="upper", cmap=gcm.current(),
            extent=extent, aspect="equal",
        )
        ax1.set_title("Preview: result  (drag = area to zoom)")
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

        self._mark_zoom_rect(ax1, ax2)
        self.figure.tight_layout()
        self._attach_spec_selector(ax0)
        self._attach_zoom_selector(ax1)
        self.canvas.draw()
        self._update_zoom_window()


class DestripeDialog(ZoomAreaMixin, OperationDialog):
    """
    Stripe removal, with the method chosen from the 'Method' selector. Only
    the parameters of the selected method are shown.

    MDSR - the multidirectional stripe remover of Liang et al. (2016).
    Fourier filtering: the image is split into shift-invariant subbands of
    different scale and direction (a nonsubsampled contourlet transform),
    the frequencies carrying stripes of the given direction are damped in
    each of them, and the image is put back together. The bottom right
    panel shows the resulting composite frequency mask - black is removed,
    yellow is kept. The groove along the stripe frequencies is what takes
    the stripes out; its width is 'Damping width' and its waist at the
    center is the low-pass residual, which is never filtered (add scales to
    narrow that waist and reach coarser stripes).

    GSR - the general stripe remover of Rottmayer et al. (2025). It splits
    the image into a clean part and a stripe part by minimizing an energy
    that wants few strong edges in the clean image and wants the stripe
    part to be sparse and constant along the stripes. 'mu1' sets how hard
    the stripes are pushed out, 'mu2' how carefully real structure is kept;
    the result improves with 'Iterations' until it converges.

    DeStripe - the AFM method of Chen & Pellequer (2011). It looks for the
    stripes in the image's own log-amplitude spectrum: pixels that are both
    bright and abruptly brighter than their surroundings, and that lie in
    lines, are pulled down to the level of their neighbours. The bottom
    right panel shows the resulting filter Phi (the paper's F-image) - 1
    means the frequency is kept untouched, 0 that it is removed entirely.
    'CVAR threshold' is the knob: it is how many standard deviations above
    its neighbours a frequency must sit to count as noise, so lower removes
    more. This method takes no stripe direction - it finds whatever lines
    are in the spectrum.

    'Zoom window...' - or dragging on the result panel - opens the image
    before and after the filter side by side, which is the honest way to
    check that only stripes were removed. 'Parameter sweep...' runs a grid
    over two parameters of the current method and shows the results side by
    side.
    """

    ZOOM_SOURCE = "result"
    PREVIEW_DEBOUNCE_MS = 700          # GSR runs an iteration for each pixel

    # Which parameters belong to which method
    METHOD_PARAMS = {
        "MDSR": ["method", "angle", "sigma", "directions", "levels",
                 "sigma_a", "max_angle", "pad"],
        "GSR": ["method", "angle", "mu1", "mu2", "iterations"],
        # DeStripe finds the stripe direction itself, so no angle here
        "DESTRIPE": ["method", "cvar_k", "window", "density", "min_run",
                     "keep_mean"],
    }

    def __init__(self, app, op_key="destripe"):
        self._last_result = None
        self._sweep_win = None
        self._init_zoom()
        super().__init__(app, op_key)
        self.geometry("1500x850")

    def _build_params(self):
        super()._build_params()
        self.vars["method"].trace_add("write", lambda *a: self._sync_method())
        btns = ttk.Frame(self, padding=(8, 0, 8, 4))
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="Zoom window...",
                   command=self.open_zoom_window).pack(side=tk.LEFT, padx=2)
        self._sweep_btn = ttk.Button(btns, text="Parameter sweep...",
                                     command=self.open_sweep_window)
        self._sweep_btn.pack(side=tk.LEFT, padx=2)
        self.hint_var = tk.StringVar(value="")
        ttk.Label(btns, textvariable=self.hint_var).pack(side=tk.LEFT, padx=12)
        self._sync_method()

    def open_sweep_window(self):
        """Open (or raise) the parameter sweep. It does not compute anything
        until its 'Run' is pressed."""
        if self._sweep_win is None or not self._sweep_win.winfo_exists():
            self._sweep_win = DestripeSweepWindow(self)
        else:
            self._sweep_win.lift()
            self._sweep_win.sync_method()

    def _method(self):
        try:
            return str(self.vars["method"].get()).upper()
        except tk.TclError:
            return "MDSR"

    def _sync_method(self):
        """Show only the parameters of the selected method."""
        method = self._method()
        self._show_params(self.METHOD_PARAMS.get(method,
                                                 self.METHOD_PARAMS["MDSR"]))
        win = getattr(self, "_sweep_win", None)
        if win is not None and win.winfo_exists():
            win.sync_method()          # offer the new method's parameters
        if hasattr(self, "hint_var"):
            hints = {
                "GSR": "  |  GSR: more iterations = better converged, slower",
                "DESTRIPE": "  |  DeStripe: lower CVAR threshold = more "
                            "frequencies removed",
            }
            self.hint_var.set(
                ("DeStripe finds the stripe direction itself"
                 if method == "DESTRIPE" else
                 "Stripe angle: 0 = horizontal scan lines, 90 = vertical")
                + "  |  Drag on the result panel to pick the zoom area"
                + hints.get(method, "")
            )

    # ---- zoom on a selected area (see ZoomAreaMixin) ----

    def _zoom_panels(self):
        if self._last_result is None:
            return None
        params = self.get_params()
        tag = f" - {describe_step(self.op_key, params)}" if params else ""
        return ([("Before destriping", self._base_data()),
                 ("After destriping", self._last_result)], tag)

    def _redraw_zoom_source(self):
        self.update_preview()

    def destroy(self):
        win = getattr(self, "_sweep_win", None)
        if win is not None and win.winfo_exists():
            win.destroy()
        super().destroy()

    # ---- drawing ----

    def _draw(self, result, removed):
        app = self.app
        self._last_result = result
        data = self._base_data()
        extent = (0, app.x_real, 0, app.y_real)
        params = self.get_params() or {}

        self.figure.clf()
        # The result gets the large panel - it is what the parameters are
        # judged on; the removed stripes and a method-specific panel share
        # the right column.
        gs = self.figure.add_gridspec(2, 2, width_ratios=[1.5, 1.0])
        ax0 = self.figure.add_subplot(gs[:, 0])
        ax1 = self.figure.add_subplot(gs[0, 1])
        ax2 = self.figure.add_subplot(gs[1, 1])

        im0 = ax0.imshow(result, origin="upper", cmap=gcm.current(),
                         extent=extent, aspect="equal")
        ax0.set_title(f"Preview: {params.get('method', 'MDSR')} result  "
                      f"(drag = area to zoom)")
        ax0.set_xlabel(f"x ({app.spatial_units})")
        ax0.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im0, ax=ax0, fraction=0.046).set_label(app.z_units)

        im1 = ax1.imshow(removed, origin="upper", cmap="viridis",
                         extent=extent, aspect="equal")
        ax1.set_title(self.spec["removed_label"])
        ax1.set_xlabel(f"x ({app.spatial_units})")
        ax1.set_ylabel(f"y ({app.spatial_units})")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046).set_label(app.z_units)

        method = self._method()
        if method == "GSR":
            self._draw_input_panel(ax2, data, extent)
        elif method == "DESTRIPE":
            self._draw_phi_panel(ax2, data, params)
        else:
            self._draw_mask_panel(ax2, data, params)

        self._mark_zoom_rect(ax0, ax1)
        self.figure.tight_layout()
        self._attach_zoom_selector(ax0)
        self.canvas.draw()
        self._update_zoom_window()

    def _draw_input_panel(self, ax, data, extent):
        """GSR has no filter to show, so the input goes here for comparison
        (on the same color scale as the result would be hard to read, so it
        gets its own)."""
        app = self.app
        im = ax.imshow(data, origin="upper", cmap=gcm.current(),
                       extent=extent, aspect="equal")
        ax.set_title("Input (before)")
        ax.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im, ax=ax, fraction=0.046).set_label(app.z_units)

    def _freq_extent(self, shape):
        """Physical frequency axes of a shifted spectrum, as an imshow
        extent (the same convention as the FFT filter dialog)."""
        app = self.app
        ny, nx = shape
        freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=app.dx))
        freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=app.dy))
        hx, hy = 0.5 / (nx * app.dx), 0.5 / (ny * app.dy)
        return [freq_x[0] - hx, freq_x[-1] + hx,
                freq_y[-1] + hy, freq_y[0] - hy]

    def _draw_phi_panel(self, ax, data, params):
        """DeStripe's filter image: how much of each frequency survives,
        and how many spectral pixels were found noisy."""
        app = self.app
        phi, noisy, _ = gd.destripe_chen_filter(data, **_chen_kwargs(params))
        im = ax.imshow(phi, origin="upper", cmap="viridis",
                       extent=self._freq_extent(data.shape), aspect="equal",
                       vmin=0, vmax=1)
        ax.set_title(f"DeStripe filter Phi (1 = kept): {int(noisy.sum())} "
                     f"noisy frequencies\nof {noisy.size} "
                     f"({100 * noisy.mean():.3f}%)")
        ax.set_xlabel(f"fx (1/{app.spatial_units})")
        ax.set_ylabel(f"fy (1/{app.spatial_units})")
        self.figure.colorbar(im, ax=ax, fraction=0.046).set_label("kept")

    def _draw_mask_panel(self, ax, data, params):
        """The composite MDSR frequency mask, on the same physical frequency
        axes as the FFT filter dialog."""
        app = self.app
        ny, nx = data.shape
        mask = gd.mdsr_mask(data.shape, **_mdsr_kwargs(params))
        freq_extent = self._freq_extent(data.shape)
        im = ax.imshow(mask, origin="upper", cmap="viridis",
                       extent=freq_extent, aspect="equal", vmin=0, vmax=1)
        # the groove reaches about 2.5 sigma bins, so it takes out
        # stripe-parallel structure longer than this
        sigma = float(params.get("sigma", gd.DEFAULTS["sigma"])) or 1.0
        reach = nx * app.dx / (2.5 * sigma)
        ax.set_title(f"MDSR mask (0 = removed): takes out stripe-parallel\n"
                     f"structure longer than ~{reach:.2f} {app.spatial_units}")
        ax.set_xlabel(f"fx (1/{app.spatial_units})")
        ax.set_ylabel(f"fy (1/{app.spatial_units})")
        self.figure.colorbar(im, ax=ax, fraction=0.046).set_label("kept")


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
            app.data, origin="upper", cmap=gcm.current(),
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
            result, origin="upper", cmap=gcm.current(),
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

    Clipping is RE-EDITABLE: when the previous step was itself a clip, this
    dialog edits that step instead of clipping the already-clipped data. So
    the histogram keeps showing the full range of the unclipped image and
    the limits can be widened again - clipping on top of a clip could only
    ever narrow the range, and the values outside it are already gone.
    """

    def __init__(self, app, op_key="percentile"):
        self._reedit = bool(app.pipeline and app.undo_stack
                            and app.pipeline[-1][0] == op_key)
        self._base = app.undo_stack[-1][0] if self._reedit else app.data
        # Sorted copy of the data for fast value <-> percentile conversion
        self._sorted = np.sort(self._base.ravel())
        self._span = None
        super().__init__(app, op_key)
        self.geometry("1350x560")
        if self._reedit:
            self.status_var.set("Editing the previous clip - full range shown")

    def _base_data(self):
        return self._base

    def _build_params(self):
        super()._build_params()
        if self._reedit:
            # start from the limits of the clip being edited
            prev = self.app.pipeline[-1][1]
            for name in ("min", "max"):
                if name in prev:
                    self.vars[name].set(prev[name])

    def apply(self):
        params = self._validated_params(show_error=True)
        if params is None:
            return
        if self._reedit:
            self.app.reapply_last_operation(self.op_key, params)
        else:
            self.app.apply_operation(self.op_key, params)
        self.destroy()

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

        # Histogram of the data the clip is computed from - the unclipped
        # image when a previous clip is being re-edited (log counts so the
        # outlier tails are visible)
        base = self._base
        ax0.hist(base.ravel(), bins=200, color="steelblue")
        ax0.set_yscale("log")
        ax0.set_title("Distribution before the clip (drag to select range)"
                      if self._reedit else "Distribution (drag to select range)")
        ax0.set_xlabel(f"value ({app.z_units})")
        ax0.set_ylabel("count")

        # Mark the current clip limits on the histogram
        try:
            lo = self.vars["min"].get()
            hi = self.vars["max"].get()
            vmin = np.percentile(base, lo)
            vmax = np.percentile(base, hi)
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
            result, origin="upper", cmap=gcm.current(),
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
    "smart_level": SmartLevelDialog,
    "polynomial": PolynomialDialog,
    "fft_filter": FFTFilterDialog,
    "destripe": DestripeDialog,
    "percentile": PercentileDialog,
}


class CorrelationWindow(tk.Toplevel):
    """Diagnostics of the correlation-gated merge, in its own big window:
    the local forward/backward height correlation with the margin, the
    per-pixel decision, the referee score that picks the winning direction on
    the disputed pixels, the merged result, and the phase/error reference
    patterns the referee correlates against. All panels are linked for
    zooming and the window follows every preview update of the dialog."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.title("Correlation merge - details")
        self.geometry("1400x780")
        self.figure = Figure(figsize=(14, 7.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

    def show(self, res, margin, aux_names, extent, merged_d, tag, z_units):
        fig = self.figure
        fig.clf()
        if res is None or getattr(res, "corr_map", None) is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Set Combine = 'correlation' and update the\n"
                    "preview to see the correlation diagnostics.",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            self.canvas.draw()
            return
        axes = fig.subplots(2, 3, sharex=True, sharey=True)

        ax = axes[0, 0]
        im = ax.imshow(res.corr_map, origin="upper", cmap="viridis",
                       extent=extent, aspect="equal", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("correlation")
        shared = float(np.mean(res.corr_map >= margin))
        ax.set_title(f"Local fwd/bwd height correlation\n"
                     f"{100 * shared:.1f}% >= margin {margin:g} = shared",
                     fontsize=9)

        ax = axes[0, 1]
        dec = res.corr_decision
        im = ax.imshow(dec, origin="upper", extent=extent, aspect="equal",
                       vmin=0, vmax=2,
                       cmap=matplotlib.colors.ListedColormap(
                           ["#c8c8c8", "#d62728", "#1f77b4"]))
        fig.colorbar(im, ax=ax, fraction=0.046,
                     ticks=[0.33, 1.0, 1.67]).ax.set_yticklabels(
            ["combined", "fwd", "bwd"])
        ax.set_title(f"Decision: combined {100 * np.mean(dec == 0):.1f}%, "
                     f"fwd {100 * np.mean(dec == 1):.1f}%, "
                     f"bwd {100 * np.mean(dec == 2):.1f}%", fontsize=9)

        ax = axes[0, 2]
        diff = np.where(dec > 0,
                        res.corr_score_fwd - res.corr_score_bwd, np.nan)
        v = (float(np.nanpercentile(np.abs(diff), 99.0))
             if np.any(dec > 0) else 1.0) or 1.0
        cmap = matplotlib.colormaps["coolwarm"].copy()
        cmap.set_bad("#e8e8e8")
        im = ax.imshow(diff, origin="upper", cmap=cmap, extent=extent,
                       aspect="equal", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("score fwd - bwd")
        ax.set_title("Referee score on the disputed pixels\n"
                     "(red = forward wins, blue = backward wins)", fontsize=9)

        ax = axes[1, 0]
        v0, v1 = np.percentile(merged_d, [0.5, 99.5])
        im = ax.imshow(merged_d, origin="upper",
                       cmap=gcm.current(), extent=extent,
                       aspect="equal", vmin=v0, vmax=v1)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(z_units)
        ax.set_title(f"Merged result{tag}", fontsize=9)

        refs = res.corr_aux_refs or []
        for k in range(2):
            ax = axes[1, 1 + k]
            if k < len(refs):
                name = aux_names[k] if k < len(aux_names) else f"aux {k + 1}"
                v0, v1 = np.percentile(refs[k], [0.5, 99.5])
                im = ax.imshow(refs[k], origin="upper", cmap="gray",
                               extent=extent, aspect="equal",
                               vmin=v0, vmax=v1)
                fig.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"{name} referee pattern\n"
                             "(fwd/bwd mean, aligned like the heights)",
                             fontsize=9)
            else:
                ax.set_axis_off()

        fig.tight_layout()
        self.canvas.draw()


class StripeWindow(tk.Toplevel):
    """Diagnostics of the stripe-gated merge, in its own big window: the
    per-scan stripe evidence (same-sign vertical jump in robust-sigma units,
    which the threshold cuts), the detected artifact segments, the per-pixel
    decision, the merged result and what the merge changed. All panels are
    linked for zooming and the window follows every preview update."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.title("Stripe merge - details")
        self.geometry("1400x780")
        self.figure = Figure(figsize=(14, 7.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

    def show(self, res, thresh, pref, extent, merged_d, tag, z_units):
        fig = self.figure
        fig.clf()
        if res is None or getattr(res, "stripe_mask_fwd", None) is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Set Combine = 'stripes' and update the\n"
                    "preview to see the stripe diagnostics.",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            self.canvas.draw()
            return
        axes = fig.subplots(2, 3, sharex=True, sharey=True)
        vmax = 2.0 * thresh

        for k, (img, name) in enumerate(((res.fwd, "Forward"),
                                         (res.bwd, "Backward"))):
            ax = axes[0, k]
            im = ax.imshow(gtw.line_artifact_score(img), origin="upper",
                           cmap="inferno", extent=extent, aspect="equal",
                           vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046).set_label("jump (sigma)")
            ax.set_title(f"{name} stripe evidence\n"
                         f"(same-sign vertical jump; threshold "
                         f"{thresh:g} sigma)", fontsize=9)

        ax = axes[0, 2]
        overlay = (res.stripe_mask_fwd.astype(float)
                   + 2.0 * res.stripe_mask_bwd)
        im = ax.imshow(overlay, origin="upper", cmap="viridis",
                       extent=extent, aspect="equal", vmin=0, vmax=3)
        fig.colorbar(im, ax=ax, fraction=0.046,
                     ticks=[0, 1, 2, 3]).set_label(
            "0 none / 1 fwd / 2 bwd / 3 both")
        ax.set_title(f"Detected stripes (runs >= min length): "
                     f"fwd {100 * res.stripe_mask_fwd.mean():.2f}%, "
                     f"bwd {100 * res.stripe_mask_bwd.mean():.2f}%",
                     fontsize=9)

        ax = axes[1, 0]
        dec = res.corr_decision
        im = ax.imshow(dec, origin="upper", extent=extent, aspect="equal",
                       vmin=0, vmax=2,
                       cmap=matplotlib.colors.ListedColormap(
                           ["#c8c8c8", "#d62728", "#1f77b4"]))
        fig.colorbar(im, ax=ax, fraction=0.046,
                     ticks=[0.33, 1.0, 1.67]).ax.set_yticklabels(
            ["combined", "fwd", "bwd"])
        ax.set_title(f"Decision (clean-scan weight {pref:g}): "
                     f"combined {100 * np.mean(dec == 0):.1f}%, "
                     f"fwd {100 * np.mean(dec == 1):.1f}%, "
                     f"bwd {100 * np.mean(dec == 2):.1f}%", fontsize=9)

        ax = axes[1, 1]
        v0, v1 = np.percentile(merged_d, [0.5, 99.5])
        im = ax.imshow(merged_d, origin="upper",
                       cmap=gcm.current(), extent=extent,
                       aspect="equal", vmin=v0, vmax=v1)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(z_units)
        ax.set_title(f"Merged result{tag}", fontsize=9)

        ax = axes[1, 2]
        removed = res.fwd - res.merged
        v = np.percentile(np.abs(removed - np.mean(removed)), 99.0) or 1.0
        im = ax.imshow(removed - np.mean(removed), origin="upper",
                       cmap="coolwarm", extent=extent, aspect="equal",
                       vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(z_units)
        ax.set_title("Difference (forward - merged)", fontsize=9)

        fig.tight_layout()
        self.canvas.draw()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class TwoWayDialog(ZoomAreaMixin, tk.Toplevel):
    """Forward/backward scan merging: hysteresis-and-lag alignment and the
    per-pixel combination of the two scans, with every hyper-parameter
    exposed. Shows the two raw scans, their opacity/anaglyph overlay, the
    hysteresis (shift) curves in both directions, and the final merged image
    cropped to the doubly-imaged area.

    Parachuting removal lives in its own window (ParachuteDialog).

    Unlike the other dialogs this one does not start from the current image -
    it always reads the raw forward and backward channels of the loaded file,
    so it belongs at the very start of a processing pipeline.
    """

    PREVIEW_DEBOUNCE_MS = 500
    DETECT = False           # ParachuteDialog overrides this

    # parameter names grouped into the panels of the dialog
    GROUPS = [
        ("Preprocess (both scans)",
         ["pre_plane", "pre_rows", "pre_rows_order"]),
        ("Alignment (hysteresis + lag)",
         ["mapping", "poly_order", "n_blocks", "max_lag", "match_level",
          "match_poly_order", "warp", "flip_backward", "crop"]),
        ("Merge",
         ["combine", "corr_combine", "weight", "slope_gain",
          "consensus_size", "beta",
          "corr_margin", "corr_window", "corr_aux",
          "stripe_thresh", "stripe_min_len", "stripe_pref"]),
    ]

    def __init__(self, app, op_key="two_way"):
        super().__init__(app)
        self.app = app
        self.op_key = op_key
        self.spec = OPERATIONS[op_key]
        self.title(self.spec["label"])
        self.geometry("1350x900")

        self._after_id = None
        self._busy = False
        self.vars = {}
        self.result = None
        self._last_params = None
        self._init_zoom()
        self._corr_win = None
        self._stripe_win = None
        self._aux_names = []

        self.fwd_title, self.bwd_title = gtw.find_pair(
            app.channels, app.channel_var.get())
        if self.bwd_title is None:
            messagebox.showerror(
                "No backward channel",
                f"No backward channel matching '{self.fwd_title}' was found in "
                f"this file.\nAvailable channels:\n  "
                + "\n  ".join(app.channels),
                parent=app,
            )
            self.destroy()
            return

        z = app.z_factor
        self.fwd = app.channels[self.fwd_title].data.astype(np.float64) * z
        self.bwd = app.channels[self.bwd_title].data.astype(np.float64) * z

        # Buttons are packed before the figure: the canvas expands to fill
        # whatever is left, so anything packed after it can get squeezed out
        # of the window.
        self._build_params()
        self._build_buttons()
        self._build_figure()
        self.update_preview()

    # ---- UI construction ----

    def _build_params(self):
        outer = ttk.Frame(self, padding=6)
        outer.pack(side=tk.TOP, fill=tk.X)
        by_name = {p["name"]: p for p in self.spec["params"]}
        self._param_widgets = {}

        for title, names in self.GROUPS:
            frame = ttk.LabelFrame(outer, text=title, padding=6)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            for row, name in enumerate(names):
                p = by_name[name]
                label = ttk.Label(frame, text=p["label"] + ":")
                label.grid(row=row, column=0, sticky=tk.W, padx=(0, 6),
                           pady=1)
                if p["type"] == "int":
                    var = tk.IntVar(value=p["default"])
                    widget = ttk.Spinbox(frame, from_=p.get("min", 0),
                                         to=p.get("max", 100), width=8,
                                         textvariable=var)
                elif p["type"] == "float":
                    var = tk.DoubleVar(value=p["default"])
                    widget = ttk.Entry(frame, textvariable=var, width=10)
                elif p["type"] == "choice":
                    var = tk.StringVar(value=p["default"])
                    widget = ttk.Combobox(frame, textvariable=var,
                                          values=p["values"], state="readonly",
                                          width=max(len(v) for v in p["values"]) + 2)
                elif p["type"] == "bool":
                    var = tk.BooleanVar(value=p["default"])
                    widget = ttk.Checkbutton(frame, variable=var)
                else:
                    raise ValueError(f"Unknown param type: {p['type']}")
                widget.grid(row=row, column=1, sticky=tk.W, pady=1)
                var.trace_add("write", self._on_param_change)
                self.vars[name] = var
                self._param_widgets[name] = (label, widget)

        self._update_param_visibility()
        self._build_display_controls(outer)

        info = ttk.Frame(self, padding=(8, 0))
        info.pack(side=tk.TOP, fill=tk.X)
        self.info_var = tk.StringVar(
            value=f"{self.fwd_title}  +  {self.bwd_title}")
        ttk.Label(info, textvariable=self.info_var, font=("TkFixedFont", 9)).pack(
            side=tk.LEFT)
        self.status_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.status_var, foreground="red").pack(
            side=tk.RIGHT)

    def _build_display_controls(self, outer):
        """Display-only settings (they re-render the preview but are not part
        of the operation parameters and are not recorded in the pipeline)."""
        frame = ttk.LabelFrame(outer, text="Display", padding=6)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=4)

        self.overlay_style = tk.StringVar(value="blend")
        self.overlay_alpha = tk.DoubleVar(value=0.5)
        self.curve_view = tk.StringVar(value="mapping (0-1)")

        ttk.Label(frame, text="Overlay:").grid(row=0, column=0, sticky=tk.W,
                                               padx=(0, 6), pady=1)
        ttk.Combobox(frame, textvariable=self.overlay_style,
                     values=["blend", "anaglyph", "corr map", "decision",
                             "stripes"],
                     state="readonly",
                     width=12).grid(row=0, column=1, sticky=tk.W, pady=1)
        ttk.Label(frame, text="Bwd opacity:").grid(row=1, column=0, sticky=tk.W,
                                                   padx=(0, 6), pady=1)
        ttk.Scale(frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=110,
                  variable=self.overlay_alpha).grid(row=1, column=1,
                                                    sticky=tk.W, pady=1)
        ttk.Label(frame, text="Curves:").grid(row=2, column=0, sticky=tk.W,
                                              padx=(0, 6), pady=1)
        ttk.Combobox(frame, textvariable=self.curve_view,
                     values=["mapping (0-1)", "shift (px)"], state="readonly",
                     width=12).grid(row=2, column=1, sticky=tk.W, pady=1)
        self._build_level_controls(frame, 3)
        self._details_btn = ttk.Button(frame, text="Correlation details...",
                                       command=self.open_details_window)
        self._details_btn.grid(row=6, column=0, columnspan=2, sticky=tk.EW,
                               pady=1)
        self._update_details_button()
        for var in (self.overlay_style, self.overlay_alpha, self.curve_view):
            var.trace_add("write", self._on_display_change)

    def _build_level_controls(self, frame, row0):
        """Display-only leveling widgets (plane + polynomial row alignment),
        shared by the merge and parachuting dialogs."""
        self.display_plane = tk.BooleanVar(value=True)
        self.display_rows = tk.BooleanVar(value=False)
        self.display_rows_order = tk.IntVar(value=2)

        ttk.Label(frame, text="Plane level:").grid(
            row=row0, column=0, sticky=tk.W, padx=(0, 6), pady=1)
        ttk.Checkbutton(frame, variable=self.display_plane).grid(
            row=row0, column=1, sticky=tk.W, pady=1)
        ttk.Label(frame, text="Row align (poly):").grid(
            row=row0 + 1, column=0, sticky=tk.W, padx=(0, 6), pady=1)
        inner = ttk.Frame(frame)
        inner.grid(row=row0 + 1, column=1, sticky=tk.W, pady=1)
        ttk.Checkbutton(inner, variable=self.display_rows).pack(side=tk.LEFT)
        ttk.Spinbox(inner, from_=0, to=10, width=3,
                    textvariable=self.display_rows_order).pack(
            side=tk.LEFT, padx=(4, 0))
        ttk.Button(frame, text="Zoom window...",
                   command=self.open_zoom_window).grid(
            row=row0 + 2, column=0, columnspan=2, sticky=tk.EW, pady=(4, 1))
        for var in (self.display_plane, self.display_rows,
                    self.display_rows_order):
            var.trace_add("write", self._on_display_change)

    def _on_display_change(self, *args):
        """Re-render with the cached result - no recomputation."""
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(
            150, lambda: (self._draw(self._last_params)
                          if self.result is not None else None))

    def _build_figure(self):
        self.figure = Figure(figsize=(13, 6.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

    def _build_buttons(self):
        frame = ttk.Frame(self, padding=8)
        frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(
            frame,
            text=("'New channel' keeps the originals and switches editing to "
                  "the merged image; 'Replace current image' overwrites the "
                  "channel you are editing."),
            foreground="gray", wraplength=600,
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(frame, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="Replace current image",
                   command=self.apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="Merge to new channel",
                   command=self.apply_as_channel).pack(side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="Update preview", command=self.update_preview).pack(
            side=tk.RIGHT, padx=4)

    # ---- Parameters ----

    def get_params(self):
        params = {}
        for p in self.spec["params"]:
            try:
                params[p["name"]] = self.vars[p["name"]].get()
            except tk.TclError:
                return None
        return params

    def _update_param_visibility(self):
        """Show only the parameter rows that matter for the currently selected
        dropdown choices (e.g. the soft-min beta only when a soft-min is in
        use). Hidden parameters keep their values."""
        if not getattr(self, "_param_widgets", None):
            return
        current = {}
        for name, var in self.vars.items():
            try:
                current[name] = var.get()
            except tk.TclError:
                pass    # mid-typing; keep this row's last visibility
        for name, (label, widget) in self._param_widgets.items():
            if twoway_param_relevant(name, current):
                label.grid()
                widget.grid()
            else:
                label.grid_remove()
                widget.grid_remove()
        self._update_details_button()

    def _on_param_change(self, *args):
        self._update_param_visibility()
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(self.PREVIEW_DEBOUNCE_MS, self.update_preview)

    # ---- Preview ----

    def update_preview(self):
        self._after_id = None
        if self._busy:
            return
        params = self.get_params()
        if params is None:
            return
        self._busy = True
        self.status_var.set("computing...")
        self.update_idletasks()
        try:
            aux = None
            self._aux_names = []
            if params.get("combine") == "correlation":
                triples = aux_pairs_for(self.app.channels, self.fwd_title,
                                        params.get("corr_aux", "phase+error"))
                self._aux_names = [t[0] for t in triples]
                aux = [(f, b) for _, f, b in triples]
            self.result = gtw.process_two_way(
                self.fwd, self.bwd, aux_pairs=aux,
                **twoway_kwargs(params, detect=self.DETECT))
            self.status_var.set("")
        except Exception as e:
            self.status_var.set(str(e))
            self.result = None
            return
        finally:
            self._busy = False
        self._last_params = params
        self._draw(params)

    # ---- Drawing helpers shared by both two-way dialogs ----

    def _extent_of(self, img):
        ny, nx = img.shape
        return (0, nx * self.app.dx, 0, ny * self.app.dy)

    def _image_panel(self, ax, img, title, cm=None, symmetric=False):
        app = self.app
        extent = self._extent_of(img)
        if symmetric:
            v = np.percentile(np.abs(img - np.mean(img)), 99.0) or 1.0
            im = ax.imshow(img - np.mean(img), origin="upper",
                           cmap=cm or "coolwarm", extent=extent,
                           aspect="equal", vmin=-v, vmax=v)
        else:
            v0, v1 = np.percentile(img, [0.5, 99.5])
            im = ax.imshow(img, origin="upper",
                           cmap=cm or gcm.current(), extent=extent,
                           aspect="equal", vmin=v0, vmax=v1)
        ax.set_title(title, fontsize=9)
        self.figure.colorbar(im, ax=ax, fraction=0.046).set_label(app.z_units)

    @staticmethod
    def _normalized(img):
        v0, v1 = np.percentile(img, [0.5, 99.5])
        return np.clip((img - v0) / ((v1 - v0) or 1.0), 0.0, 1.0)

    def _overlay_panel(self, ax, fwd, bwd):
        """Forward and backward on top of each other: an opacity blend
        (backward drawn over forward with adjustable alpha), a red/cyan
        anaglyph where any residual misalignment shows as color fringes, or -
        with combine='correlation' - the local fwd/bwd correlation map or the
        per-pixel decision (averaged / forward kept / backward kept)."""
        style = self.overlay_style.get()
        res = self.result
        if style in ("corr map", "decision", "stripes"):
            needed = {"corr map": ("corr_map", "set Combine = 'correlation'"),
                      "decision": ("corr_decision",
                                   "set Combine = 'correlation' or 'stripes'"),
                      "stripes": ("stripe_mask_fwd",
                                  "set Combine = 'stripes'")}[style]
            if getattr(res, needed[0], None) is None:
                ax.text(0.5, 0.5, f"{needed[1]}\nto see this view",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9)
                ax.set_axis_off()
                return
            margin = float((self._last_params or {}).get("corr_margin", 0.7))
            extent = self._extent_of(res.merged)
            if style == "stripes":
                overlay = (res.stripe_mask_fwd.astype(float)
                           + 2.0 * res.stripe_mask_bwd)
                im = ax.imshow(overlay, origin="upper", cmap="viridis",
                               extent=extent, aspect="equal", vmin=0, vmax=3)
                self.figure.colorbar(im, ax=ax, fraction=0.046,
                                     ticks=[0, 1, 2, 3]).set_label(
                    "0 none / 1 fwd / 2 bwd / 3 both")
                ax.set_title(
                    f"Stripe artifacts: "
                    f"fwd {100 * res.stripe_mask_fwd.mean():.2f}%, "
                    f"bwd {100 * res.stripe_mask_bwd.mean():.2f}%",
                    fontsize=9)
                return
            if style == "corr map":
                im = ax.imshow(res.corr_map, origin="upper", cmap="viridis",
                               extent=extent, aspect="equal", vmin=-1, vmax=1)
                self.figure.colorbar(im, ax=ax, fraction=0.046).set_label(
                    "local correlation")
                shared = float(np.mean(res.corr_map >= margin))
                ax.set_title(f"Local fwd/bwd correlation - "
                             f"{100 * shared:.1f}% above margin {margin:g}",
                             fontsize=9)
            else:
                dec = res.corr_decision
                im = ax.imshow(dec, origin="upper", extent=extent,
                               aspect="equal", vmin=0, vmax=2,
                               cmap=matplotlib.colors.ListedColormap(
                                   ["#c8c8c8", "#d62728", "#1f77b4"]))
                self.figure.colorbar(im, ax=ax, fraction=0.046,
                                     ticks=[0.33, 1.0, 1.67]).ax \
                    .set_yticklabels(["avg", "fwd", "bwd"])
                ax.set_title(
                    f"Decision: averaged {100 * np.mean(dec == 0):.1f}%, "
                    f"fwd {100 * np.mean(dec == 1):.1f}%, "
                    f"bwd {100 * np.mean(dec == 2):.1f}%", fontsize=9)
            return
        nf, nb = self._normalized(fwd), self._normalized(bwd)
        alpha = float(self.overlay_alpha.get())
        if style == "anaglyph":
            comp = np.dstack([nf, nb, nb])
            title = "Overlay - anaglyph (fwd red / bwd cyan)"
        else:
            base = gcm.current()(nf)
            over = matplotlib.colormaps["gray"](nb)
            comp = (1.0 - alpha) * base + alpha * over
            title = f"Overlay - bwd at {alpha:.0%} opacity over fwd"
        ax.imshow(np.clip(comp, 0, 1), origin="upper",
                  extent=self._extent_of(fwd), aspect="equal")
        ax.set_title(title, fontsize=9)

    def _curves_panel(self, ax):
        """The hysteresis / lag curves in both directions, in one of two
        views (Display > Curves):

        ``mapping (0-1)`` - the classic hysteresis plot: backward coordinate
        against forward coordinate, both normalized to [0, 1], curving around
        the slope-1 identity line. Includes the fitted mapping, the measured
        column matches, and (when the power-law model was used) its f(t) and
        g(t) distortion curves bowing on either side of the diagonal.

        ``shift (px)`` - the same information as a deviation from identity in
        pixels, which makes small lags/bows much easier to read.
        """
        a = self.result.alignment
        n = len(a.shift_px)
        grid = np.arange(n)
        mapping_view = self.curve_view.get().startswith("mapping")

        if mapping_view:
            s = n - 1.0
            t = grid / s
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1,
                    label="identity (slope 1)")
            if a.measured_centers is not None:
                good = a.measured_quality > 0
                ax.plot(a.measured_centers[good] / s,
                        (a.measured_centers[good]
                         + a.measured_shift_px[good]) / s,
                        "o", ms=4, color="tab:blue", label="measured")
            ax.plot(t, (grid + a.shift_px) / s, "-", lw=2, color="tab:red",
                    label=f"fwd->bwd ({a.mapping})")
            res = a.hysteresis_result
            if res is not None:
                ax.plot(res.x_c, res.f_x, lw=1.2, color="tab:green",
                        label="model f(t) fwd")
                ax.plot(res.x_c, res.g_x, lw=1.2, color="tab:purple",
                        label="model g(t) bwd")
            if a.crop_cols is not None and a.crop_cols != (0, n):
                ax.axvspan(0, a.crop_cols[0] / s, color="gray", alpha=0.15)
                ax.axvspan(a.crop_cols[1] / s, 1, color="gray", alpha=0.15)
            ax.set_title("Hysteresis mapping (shaded = cropped)", fontsize=9)
            ax.set_xlabel("forward coordinate (0-1)")
            ax.set_ylabel("backward coordinate (0-1)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.legend(fontsize=6, loc="lower right")
            return

        if a.measured_centers is not None:
            good = a.measured_quality > 0
            ax.plot(a.measured_centers[good], a.measured_shift_px[good], "o",
                    ms=4, color="tab:blue", label="measured")
            if (~good).any():
                ax.plot(a.measured_centers[~good], a.measured_shift_px[~good],
                        "x", ms=4, color="lightgray", label="low contrast")
        ax.plot(grid, a.shift_px, "-", lw=2, color="tab:red",
                label=f"fwd->bwd ({a.mapping})")
        # inverse mapping: for each backward column, the matching forward one
        order = np.argsort(a.columns_bwd)
        inv = np.interp(grid, a.columns_bwd[order], a.columns_fwd[order]) - grid
        ax.plot(grid, inv, "--", lw=1.6, color="tab:green", label="bwd->fwd")
        if a.crop_cols is not None and a.crop_cols != (0, n):
            for c in a.crop_cols:
                ax.axvline(c, color="gray", lw=1.0, ls=":")
            ax.axvspan(0, a.crop_cols[0], color="gray", alpha=0.15)
            ax.axvspan(a.crop_cols[1], n, color="gray", alpha=0.15)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title("Hysteresis / lag curves (shaded = cropped)", fontsize=9)
        ax.set_xlabel("column (px)")
        ax.set_ylabel("shift (px)")
        ax.set_xlim(0, n)
        ax.legend(fontsize=7)

    def _merged_title(self):
        res = self.result
        ny, nx = res.merged.shape
        cropped = self.fwd.shape[1] - nx
        title = f"Merged result  {nx}x{ny} px"
        if cropped > 0:
            title += f"  (cropped {cropped} px)"
        return title

    def _display_images(self):
        """The forward, backward, and merged images as shown in the panels.

        Display-only leveling for observing the features: 'Plane level'
        subtracts each panel's own fitted plane and 'Row align (poly)'
        additionally subtracts each panel's own per-row polynomial background
        of the chosen order. The data that gets applied or saved is never
        leveled here."""
        res = self.result
        images = [res.fwd, res.bwd, res.merged]
        tags = []
        try:
            rows_on = self.display_rows.get()
            order = int(self.display_rows_order.get())
        except tk.TclError:
            rows_on, order = False, 2   # spinbox mid-typing
        if self.display_plane.get():
            images = [img - gtw.fit_plane(img) for img in images]
            tags.append("plane")
        if rows_on:
            images = [img - gtw.fit_rows_poly(img, order) for img in images]
            tags.append(f"rows p{order}")
        tag = f" ({' + '.join(tags)} leveled)" if tags else ""
        fwd, bwd, merged = images
        return fwd, bwd, merged, tag

    # ---- Zoom on a selected area (see ZoomAreaMixin) ----

    def _zoom_panels(self):
        if self.result is None:
            return None
        fwd_d, bwd_d, merged_d, tag = self._display_images()
        titles = [f"Forward ({self.fwd_title})", "Backward, aligned", "Merged"]
        return list(zip(titles, [fwd_d, bwd_d, merged_d])), tag

    def _redraw_zoom_source(self):
        self._draw(self._last_params)   # re-render to outline the area

    # ---- Merge-details windows (correlation / stripes) ----

    def _current_combine(self):
        try:
            return self.vars["combine"].get()
        except (KeyError, tk.TclError):
            return ""

    def _update_details_button(self):
        """The details button belongs to the gated merge modes: shown as
        'Correlation details...' or 'Stripe details...' when one of them is
        selected, hidden otherwise."""
        btn = getattr(self, "_details_btn", None)
        if btn is None:
            return
        combine = self._current_combine()
        if combine == "correlation":
            btn.configure(text="Correlation details...")
            btn.grid()
        elif combine == "stripes":
            btn.configure(text="Stripe details...")
            btn.grid()
        else:
            btn.grid_remove()

    def open_details_window(self):
        if self._current_combine() == "stripes":
            self.open_stripe_window()
        else:
            self.open_corr_window()

    def open_stripe_window(self):
        """Open (or raise) the stripe-merge diagnostics window."""
        if self._stripe_win is None or not self._stripe_win.winfo_exists():
            self._stripe_win = StripeWindow(self)
        else:
            self._stripe_win.lift()
        self._update_stripe_window()

    def _update_stripe_window(self):
        if self._stripe_win is None or not self._stripe_win.winfo_exists():
            return
        res = self.result
        p = self._last_params or {}
        merged_d = tag = extent = None
        if res is not None and getattr(res, "stripe_mask_fwd", None) is not None:
            _, _, merged_d, tag = self._display_images()
            extent = self._extent_of(res.merged)
        self._stripe_win.show(res, float(p.get("stripe_thresh", 3.0)),
                              float(p.get("stripe_pref", 1.0)), extent,
                              merged_d, tag, self.app.z_units)

    def open_corr_window(self):
        """Open (or raise) the correlation-merge diagnostics window."""
        if self._corr_win is None or not self._corr_win.winfo_exists():
            self._corr_win = CorrelationWindow(self)
        else:
            self._corr_win.lift()
        self._update_corr_window()

    def _update_corr_window(self):
        if self._corr_win is None or not self._corr_win.winfo_exists():
            return
        res = self.result
        margin = float((self._last_params or {}).get("corr_margin", 0.7))
        merged_d = tag = extent = None
        if res is not None and getattr(res, "corr_map", None) is not None:
            _, _, merged_d, tag = self._display_images()
            extent = self._extent_of(res.corr_map)
        self._corr_win.show(res, margin, self._aux_names, extent,
                            merged_d, tag, self.app.z_units)

    @staticmethod
    def _link_panels(*axes):
        """Share the x/y limits of the image panels, so toolbar zoom or pan
        on any one of them applies to all of them at once."""
        first = axes[0]
        for ax in axes[1:]:
            try:
                ax.sharex(first)
                ax.sharey(first)
            except AttributeError:      # older matplotlib
                first.get_shared_x_axes().join(first, ax)
                first.get_shared_y_axes().join(first, ax)

    def destroy(self):
        for win in (getattr(self, "_corr_win", None),
                    getattr(self, "_stripe_win", None)):
            if win is not None and win.winfo_exists():
                win.destroy()
        super().destroy()

    def _set_info(self):
        a = self.result.alignment
        self.info_var.set(
            f"{self.fwd_title} + {self.bwd_title}   |   "
            f"lag {a.lag_px:+.2f} px, bow {a.bow_px:.2f} px, "
            f"flip={a.flipped_backward}   |   "
            f"fwd/bwd rms {a.rms_before:.3g} -> {a.rms_after:.3g} "
            f"{self.app.z_units}, "
            f"corr {a.corr_before:.4f} -> {a.corr_after:.4f}"
        )

    def _draw(self, params):
        res = self.result
        fwd_d, bwd_d, merged_d, tag = self._display_images()
        self.figure.clf()
        axes = self.figure.subplots(2, 3)

        self._image_panel(axes[0, 0], fwd_d,
                          f"Forward ({self.fwd_title}){tag}")
        self._image_panel(axes[0, 1], bwd_d,
                          f"Backward ({self.bwd_title}), aligned{tag}")
        self._overlay_panel(axes[0, 2], fwd_d, bwd_d)

        self._curves_panel(axes[1, 0])
        self._image_panel(axes[1, 1], merged_d, self._merged_title() + tag)
        self._image_panel(axes[1, 2], res.fwd - res.merged,
                          self.spec["removed_label"], symmetric=True)

        self._link_panels(axes[0, 0], axes[0, 1], axes[0, 2],
                          axes[1, 1], axes[1, 2])
        self._mark_zoom_rect(axes[0, 0], axes[0, 1], axes[1, 1])
        self._attach_zoom_selector(axes[0, 0])
        self.figure.tight_layout()
        self.canvas.draw()
        self._set_info()
        self._update_zoom_window()
        self._update_corr_window()
        self._update_stripe_window()

    # ---- Apply ----

    def apply(self):
        """Overwrite the image currently being edited with the merged result."""
        params = self.get_params()
        if params is None:
            return
        if self.app.pipeline and not messagebox.askyesno(
            self.spec["label"],
            "This operation restarts from the raw forward/backward channels, "
            "so the steps already applied to this image will be discarded.\n\n"
            "Apply anyway?",
            parent=self,
        ):
            return
        self.app.apply_operation(self.op_key, params)
        self.destroy()

    def apply_as_channel(self):
        """Add the merged image as a new channel and switch editing to it,
        leaving the forward and backward channels untouched."""
        params = self.get_params()
        if params is None:
            return
        if self.result is None:
            self.update_preview()
            if self.result is None:
                return
        base = re.sub(r"\s*\[[^\]]*\]$", "", self.fwd_title).strip() or "Height"
        suffix = self.spec.get("channel_suffix", "[Merged]")
        title = self.app.add_channel(
            f"{base} {suffix}", self.result.merged,
            template=self.app.channels[self.fwd_title],
            pipeline_step=(self.op_key, params),
        )
        self.app.status_var.set(f"Created channel '{title}'")
        self.destroy()


class ParachuteDialog(TwoWayDialog):
    """Parachuting-artifact removal in its own window.

    Aligns the forward/backward pair (using the same machinery as the two-way
    merge dialog, with the alignment kept at its defaults unless changed
    here), then shows the H(delta, dz) height-difference histograms of BOTH
    scan directions with the decision line drawn on top, the flagged pixels,
    and the repaired result. Flagged pixels are replaced from the opposite
    scan; everything else is combined according to the Merge settings.
    """

    DETECT = True

    GROUPS = [
        ("Preprocess (both scans)",
         ["pre_plane", "pre_rows", "pre_rows_order"]),
        ("Alignment",
         ["mapping", "poly_order", "crop"]),
        ("Parachuting detection",
         ["slope_mode", "slope", "slope_scale", "offset", "max_delta"]),
        ("Merge of unflagged pixels",
         ["combine", "weight", "beta", "both_flagged"]),
    ]

    def __init__(self, app, op_key="parachute"):
        super().__init__(app, op_key)

    def _build_display_controls(self, outer):
        # no overlay/curves panel here - only the display leveling
        frame = ttk.LabelFrame(outer, text="Display", padding=6)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=4)
        self._build_level_controls(frame, 0)

    def _histogram_panel(self, ax, img, direction, slope, offset, max_delta,
                         tag):
        try:
            hist, deltas, edges = gtw.difference_histogram(
                img, direction, max_delta=max_delta, detrend=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"histogram failed:\n{e}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8)
            return
        im = ax.imshow(np.log10(hist.T + 1.0), origin="lower", aspect="auto",
                       extent=(0.5, deltas[-1] + 0.5, edges[0], edges[-1]),
                       cmap="viridis")
        self.figure.colorbar(im, ax=ax, fraction=0.046).set_label("log10 count")
        if np.isfinite(slope):
            d = np.array([0.0, float(deltas[-1])])
            ax.plot(d, -(slope * d + offset), color="tab:orange", lw=1.8,
                    label=f"decision line (slope {slope:.3g})")
            ax.legend(fontsize=7)
        ax.set_title(f"H($\\Delta$, $\\Delta z$) {tag}", fontsize=9)
        ax.set_xlabel("$\\Delta$ (pixel)")
        ax.set_ylabel(f"$\\Delta z$ ({self.app.z_units})")

    def _draw(self, params):
        res = self.result
        a = res.alignment
        offset = float(params["offset"])
        max_delta = int(params["max_delta"])
        dir_bwd = +1 if a.flipped_backward else -1

        self.figure.clf()
        axes = self.figure.subplots(2, 3)

        self._histogram_panel(axes[0, 0], res.fwd, +1, res.slope_fwd,
                              offset, max_delta, "forward")
        self._histogram_panel(axes[0, 1], res.bwd, dir_bwd, res.slope_bwd,
                              offset, max_delta, "backward")

        ax = axes[0, 2]
        overlay = res.mask_fwd.astype(float) + 2.0 * res.mask_bwd
        im = ax.imshow(overlay, origin="upper", cmap="viridis",
                       extent=self._extent_of(res.merged), aspect="equal",
                       vmin=0, vmax=3)
        self.figure.colorbar(im, ax=ax, fraction=0.046,
                             ticks=[0, 1, 2, 3]).set_label(
            "0 none / 1 fwd / 2 bwd / 3 both")
        ax.set_title(f"Flagged: fwd {100*res.fraction_fwd:.1f}%, "
                     f"bwd {100*res.fraction_bwd:.1f}%", fontsize=9)

        fwd_d, _, merged_d, tag = self._display_images()
        self._image_panel(axes[1, 0], fwd_d,
                          f"Forward ({self.fwd_title}){tag}")
        self._image_panel(axes[1, 1], merged_d, self._merged_title() + tag)
        self._image_panel(axes[1, 2], res.fwd - res.merged,
                          self.spec["removed_label"], symmetric=True)

        self._link_panels(axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
        self._mark_zoom_rect(axes[1, 0], axes[1, 1])
        self._attach_zoom_selector(axes[1, 0])
        self.figure.tight_layout()
        self.canvas.draw()
        self._set_info()
        self._update_zoom_window()


DIALOG_CLASSES["two_way"] = TwoWayDialog
DIALOG_CLASSES["parachute"] = ParachuteDialog



class QuickViewTab(ttk.Frame):
    """Flip through a folder of .gwy files, minimally preprocessed.

    Every image gets the same two steps - a fitted plane subtracted, then
    rows aligned with a second-order polynomial - which is what makes a raw
    scan readable without deciding anything about it. Nothing here is
    applied to the processing tab; this is for looking, to find the scans
    worth working on.

    Results are cached per (file, channel) so stepping back is instant; the
    cache is bounded because a folder can hold more images than fit in
    memory.
    """

    CACHE_LIMIT = 32

    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.folder = None
        self.files = []          # full paths, natural order
        self.index = -1
        self._cache = {}         # (path, channel) -> processed view dict
        self._cache_order = []   # keys, oldest first
        self._build()

    # ------------------------------------------------------------- layout --

    def _build(self):
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X)

        ttk.Button(bar, text="Select folder...",
                   command=self.select_folder).pack(side=tk.LEFT)
        self.folder_label = ttk.Label(bar, text="No folder selected")
        self.folder_label.pack(side=tk.LEFT, padx=(8, 0))

        nav = ttk.Frame(bar)
        nav.pack(side=tk.RIGHT)
        ttk.Label(nav, text="Channel:").pack(side=tk.LEFT)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(
            nav, textvariable=self.channel_var, state="readonly", width=22)
        self.channel_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.channel_combo.bind("<<ComboboxSelected>>",
                                lambda e: self.show(self.index))
        self.prev_btn = ttk.Button(nav, text="< Back",
                                   command=lambda: self.step(-1))
        self.prev_btn.pack(side=tk.LEFT)
        self.count_label = ttk.Label(nav, text="0 / 0", width=10,
                                     anchor=tk.CENTER)
        self.count_label.pack(side=tk.LEFT, padx=4)
        self.next_btn = ttk.Button(nav, text="Next >",
                                   command=lambda: self.step(1))
        self.next_btn.pack(side=tk.LEFT)

        self.name_label = ttk.Label(self, text="", anchor=tk.CENTER,
                                    font=("TkDefaultFont", 11, "bold"))
        self.name_label.pack(fill=tk.X, pady=(6, 0))

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

        self.status_var = tk.StringVar(
            value="Select a folder of .gwy files. Each one is shown with a "
                  "plane subtracted and rows aligned (polynomial, order 2).")
        ttk.Label(self, textvariable=self.status_var,
                  wraplength=900).pack(fill=tk.X, pady=(4, 0))

        # Arrow keys step through the folder once the image has the focus,
        # which a click on it gives. They are bound to the canvas and not to
        # the application, so they never fight with an entry field elsewhere.
        plot = self.canvas.get_tk_widget()
        plot.bind("<Button-1>", lambda e: plot.focus_set())
        for key, delta in (("<Left>", -1), ("<Right>", 1),
                           ("<Prior>", -1), ("<Next>", 1)):
            plot.bind(key, lambda e, d=delta: self.step(d))
        self._update_nav()

    # -------------------------------------------------------------- files --

    def select_folder(self):
        folder = filedialog.askdirectory(
            title="Select a folder of .gwy files",
            initialdir=os.path.dirname(self.app.filename or "") or ".")
        if not folder:
            return
        names = sorted((f for f in os.listdir(folder)
                        if f.lower().endswith(".gwy")), key=_natural_key)
        if not names:
            messagebox.showinfo(
                "No files", "No .gwy files found in the selected folder.")
            return
        self.folder = folder
        self.files = [os.path.join(folder, n) for n in names]
        self._cache.clear()
        self._cache_order.clear()
        self.folder_label.config(
            text=f"{os.path.basename(folder)}  ({len(self.files)} files)")
        self.index = -1
        self.show(0)

    def step(self, delta):
        if not self.files:
            return
        self.show(min(max(self.index + delta, 0), len(self.files) - 1))

    def _update_nav(self):
        n = len(self.files)
        self.count_label.config(
            text=f"{self.index + 1} / {n}" if n else "0 / 0")
        for btn, ok in ((self.prev_btn, self.index > 0),
                        (self.next_btn, 0 <= self.index < n - 1)):
            btn.state(["!disabled"] if ok else ["disabled"])

    # ------------------------------------------------------------ display --

    def show(self, index):
        """Load, preprocess and draw the file at `index`."""
        if not self.files or not 0 <= index < len(self.files):
            return
        path = self.files[index]
        name = os.path.basename(path)
        self.index = index
        self._update_nav()
        self.name_label.config(text=f"{index + 1}/{len(self.files)}  -  {name}")

        try:
            channels = gwy_loader.load_gwy(path)
        except Exception as e:
            self._draw_message(f"Could not read {name}:\n{e}")
            self.status_var.set(f"{name}: {e}")
            return
        if not channels:
            self._draw_message(f"{name} has no data channels.")
            return

        names = list(channels)
        self.channel_combo["values"] = names
        channel = pick_channel(names, self.channel_var.get())
        self.channel_var.set(channel)

        view = self._processed(path, channel, channels[channel])
        self._draw(view, name, channel)
        self.status_var.set(
            f"{name} - {channel}: plane subtracted, rows aligned "
            f"(polynomial, order 2)")

    def _processed(self, path, channel, field):
        key = (path, channel)
        if key in self._cache:
            return self._cache[key]
        view = channel_view(field)
        data = gp.level_by_plane_fit(view["data"])
        view["data"] = gp.align_rows(data, method="polynomial", order=2)
        self._cache[key] = view
        self._cache_order.append(key)
        while len(self._cache_order) > self.CACHE_LIMIT:
            self._cache.pop(self._cache_order.pop(0), None)
        return view

    def refresh_display(self):
        """Redraw with the current colour map (nothing is recomputed)."""
        if 0 <= self.index < len(self.files):
            path = self.files[self.index]
            key = (path, self.channel_var.get())
            if key in self._cache:
                self._draw(self._cache[key], os.path.basename(path), key[1])

    def _draw(self, view, name, channel):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(
            view["data"], origin="upper", cmap=gcm.current(),
            extent=(0, view["x_real"], 0, view["y_real"]), aspect="equal",
        )
        ax.set_title(channel)
        ax.set_xlabel(f"x ({view['spatial_units']})")
        ax.set_ylabel(f"y ({view['spatial_units']})")
        self.figure.colorbar(im, ax=ax, pad=0.05,
                             fraction=0.046).set_label(view["z_units"])
        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_message(self, text):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()


class ExportChoiceDialog(tk.Toplevel):
    """Ask what to write when a whole folder is exported.

    `result` is a dict of the three flags once OK is pressed, and None if the
    dialog was cancelled or closed.
    """

    CHOICES = (
        ("annotated", "Annotated PNG - axes, colour bar and scale bar", True),
        ("pure", "Pure image - one pixel per data point, in a 'pure' "
                 "subfolder", True),
        ("gwy", "Gwyddion .gwy file - the balanced channel", False),
    )

    def __init__(self, master, warning=None):
        super().__init__(master)
        self.title("Export the balanced folder")
        self.resizable(False, False)
        self.result = None

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="For every image in the folder, write:").pack(
            anchor=tk.W, pady=(0, 6))
        self.vars = {}
        for key, text, default in self.CHOICES:
            self.vars[key] = tk.BooleanVar(value=default)
            ttk.Checkbutton(frame, text=text,
                            variable=self.vars[key]).pack(anchor=tk.W, pady=1)
        ttk.Label(frame, wraplength=430, foreground="#606060",
                  text="All of them are drawn with the folder's shared range, "
                       "so they can be compared side by side. Balancing only "
                       "ever shifts an image, never rescales it, so the .gwy "
                       "holds the measured heights - and the full data, since "
                       "only the display is clipped to the "
                       "range.").pack(anchor=tk.W, pady=(8, 0))
        if warning:
            ttk.Label(frame, text=warning, wraplength=430,
                      foreground="#a04000").pack(anchor=tk.W, pady=(8, 0))

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT)
        ttk.Button(buttons, text="Choose folder...",
                   command=self._accept).pack(side=tk.RIGHT, padx=(0, 6))

        self.transient(master)
        self.grab_set()
        self.wait_window(self)

    def _accept(self):
        self.result = {k: bool(v.get()) for k, v in self.vars.items()}
        self.destroy()


class BalancedViewTab(ttk.Frame):
    """Show a whole folder on one colour scale.

    Pick a folder and a channel; every file is levelled the same way the
    quick view levels it, segmented into cells and substrate, and measured
    at those two places. The folder's measurements are then reduced to one
    shared range - see gwy_balance for what the three modes do to get
    there - and every image is drawn with it, so the same colour means the
    same thing in every image of the set.

    The range that comes out is a starting point, not a verdict: it can be
    typed over, and the contact sheet and diagnostics views are there to
    judge whether it shows the structure inside the cells without
    flattening it.
    """

    CACHE_LIMIT = 40         # preprocessed images kept in memory
    THUMB = 320              # longest side of a contact-sheet thumbnail
    VIEWS = ("Single image", "Contact sheet", "Diagnostics")

    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.folder = None
        self.files = []          # full paths, natural order
        self.index = -1
        self.channel = None      # channel the current analysis was run on
        self.measures = []       # gwy_balance.measure() per file
        self.metas = []          # channel_view metadata (no data) per file
        self.thumbs = []         # decimated data per file
        self.result = None       # gwy_balance.balance() output
        self.override = None     # range typed by the user, or None
        self._cache = {}         # (path, channel, levelled) -> view dict
        self._cache_order = []
        self._busy = False
        self._build()

    # ------------------------------------------------------------- layout --

    def _build(self):
        files = ttk.Frame(self)
        files.pack(fill=tk.X)
        ttk.Button(files, text="Select folder...",
                   command=self.select_folder).pack(side=tk.LEFT)
        self.folder_label = ttk.Label(files, text="No folder selected")
        self.folder_label.pack(side=tk.LEFT, padx=(8, 0))

        nav = ttk.Frame(files)
        nav.pack(side=tk.RIGHT)
        ttk.Label(nav, text="Channel:").pack(side=tk.LEFT)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(
            nav, textvariable=self.channel_var, state="readonly", width=22)
        self.channel_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.channel_combo.bind("<<ComboboxSelected>>",
                                lambda e: self.analyse())
        self.prev_btn = ttk.Button(nav, text="< Back",
                                   command=lambda: self.step(-1))
        self.prev_btn.pack(side=tk.LEFT)
        self.count_label = ttk.Label(nav, text="0 / 0", width=10,
                                     anchor=tk.CENTER)
        self.count_label.pack(side=tk.LEFT, padx=4)
        self.next_btn = ttk.Button(nav, text="Next >",
                                   command=lambda: self.step(1))
        self.next_btn.pack(side=tk.LEFT)

        # ---- how the range is found ----
        box = ttk.LabelFrame(self, text="Balance", padding=6)
        box.pack(fill=tk.X, pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value=gb.MODES[gb.DEFAULT_MODE])
        mode = ttk.Combobox(row, textvariable=self.mode_var, state="readonly",
                            values=list(gb.MODES.values()), width=26)
        mode.pack(side=tk.LEFT, padx=(4, 12))
        mode.bind("<<ComboboxSelected>>", lambda e: self.rebalance())

        ttk.Label(row, text="Cell size (% of frame):").pack(side=tk.LEFT)
        self.cell_var = tk.StringVar(value=f"{100 * gb.CELL_FRACTION:g}")
        ttk.Spinbox(row, textvariable=self.cell_var, width=5, from_=0.2,
                    to=20.0, increment=0.5).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(row, text="Cell percentiles:").pack(side=tk.LEFT)
        self.plo_var = tk.StringVar(value=f"{gb.P_LO:g}")
        self.phi_var = tk.StringVar(value=f"{gb.P_HI:g}")
        ttk.Spinbox(row, textvariable=self.plo_var, width=5, from_=0.0,
                    to=49.0, increment=0.5).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row, text="to").pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=self.phi_var, width=5, from_=51.0,
                    to=100.0, increment=0.5).pack(side=tk.LEFT, padx=(2, 12))

        self.level_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Level first", variable=self.level_var,
                        command=self.analyse).pack(side=tk.LEFT, padx=(0, 12))
        self.zero_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Baseline to zero", variable=self.zero_var,
                        command=self.rebalance).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(row, text="Recompute",
                   command=self.analyse).pack(side=tk.LEFT)

        # ---- the range itself, and how to look at it ----
        row2 = ttk.Frame(box)
        row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row2, text="Range:").pack(side=tk.LEFT)
        self.vmin_var = tk.StringVar()
        self.vmax_var = tk.StringVar()
        self.vmin_entry = ttk.Entry(row2, textvariable=self.vmin_var, width=10)
        self.vmin_entry.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row2, text="to").pack(side=tk.LEFT)
        self.vmax_entry = ttk.Entry(row2, textvariable=self.vmax_var, width=10)
        self.vmax_entry.pack(side=tk.LEFT, padx=(2, 4))
        self.units_label = ttk.Label(row2, text="")
        self.units_label.pack(side=tk.LEFT)
        for entry in (self.vmin_entry, self.vmax_entry):
            entry.bind("<Return>", lambda e: self.apply_range())
        ttk.Button(row2, text="Apply", command=self.apply_range).pack(
            side=tk.LEFT, padx=(8, 2))
        ttk.Button(row2, text="Auto", command=self.auto_range).pack(
            side=tk.LEFT)

        ttk.Button(row2, text="Export all...", command=self.export).pack(
            side=tk.RIGHT)
        self.view_var = tk.StringVar(value=self.VIEWS[0])
        view = ttk.Combobox(row2, textvariable=self.view_var,
                            state="readonly", values=self.VIEWS, width=14)
        view.pack(side=tk.RIGHT, padx=(4, 12))
        view.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(row2, text="View:").pack(side=tk.RIGHT)

        self.name_label = ttk.Label(self, text="", anchor=tk.CENTER,
                                    font=("TkDefaultFont", 11, "bold"))
        self.name_label.pack(fill=tk.X, pady=(6, 0))

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, pady=(4, 0))
        self.progress = ttk.Progressbar(bottom, mode="determinate",
                                        length=140)
        self.progress.pack(side=tk.RIGHT, padx=(8, 0))
        self.status_var = tk.StringVar(
            value="Select a folder of .gwy files. Every image is measured "
                  "where the cells are and where the substrate is, and the "
                  "whole folder is then drawn on one colour scale.")
        ttk.Label(bottom, textvariable=self.status_var,
                  wraplength=900).pack(side=tk.LEFT, fill=tk.X, expand=True)

        plot = self.canvas.get_tk_widget()
        plot.bind("<Button-1>", lambda e: plot.focus_set())
        for key, delta in (("<Left>", -1), ("<Right>", 1),
                           ("<Prior>", -1), ("<Next>", 1)):
            plot.bind(key, lambda e, d=delta: self.step(d))
        self._update_nav()

    # ----------------------------------------------------------- settings --

    def mode(self):
        """The gwy_balance mode key behind the label in the combo box."""
        label = self.mode_var.get()
        return next((k for k, v in gb.MODES.items() if v == label),
                    gb.DEFAULT_MODE)

    def _number(self, var, default, low, high):
        """A spin box's value, or `default` if it has been typed into and
        no longer makes sense."""
        try:
            value = float(var.get())
        except ValueError:
            value = default
        value = min(max(value, low), high)
        var.set(f"{value:g}")
        return value

    def settings(self):
        """The measurement parameters currently set, as gwy_balance wants
        them."""
        p_lo = self._number(self.plo_var, gb.P_LO, 0.0, 49.0)
        p_hi = self._number(self.phi_var, gb.P_HI, 51.0, 100.0)
        return {
            "cell_fraction": self._number(
                self.cell_var, 100 * gb.CELL_FRACTION, 0.2, 20.0) / 100.0,
            "p_lo": p_lo,
            "p_hi": p_hi,
        }

    # -------------------------------------------------------------- files --

    def select_folder(self):
        folder = filedialog.askdirectory(
            title="Select a folder of .gwy files",
            initialdir=os.path.dirname(self.app.filename or "") or ".")
        if not folder:
            return
        names = sorted((f for f in os.listdir(folder)
                        if f.lower().endswith(".gwy")), key=_natural_key)
        if not names:
            messagebox.showinfo(
                "No files", "No .gwy files found in the selected folder.")
            return
        self.folder = folder
        self.files = [os.path.join(folder, n) for n in names]
        self._cache.clear()
        self._cache_order.clear()
        self.folder_label.config(
            text=f"{os.path.basename(folder)}  ({len(self.files)} files)")
        self.index = 0
        try:
            channels = list(gwy_loader.load_gwy(self.files[0]))
        except Exception as e:
            messagebox.showerror("Could not read file",
                                 f"{os.path.basename(self.files[0])}:\n{e}")
            return
        self.channel_combo["values"] = channels
        self.channel_var.set(pick_channel(channels, self.channel_var.get()))
        self.analyse()

    def step(self, delta):
        if not self.files:
            return
        self.index = min(max(self.index + delta, 0), len(self.files) - 1)
        self._update_nav()
        self.redraw()

    def _update_nav(self):
        n = len(self.files)
        self.count_label.config(
            text=f"{self.index + 1} / {n}" if n else "0 / 0")
        for btn, ok in ((self.prev_btn, self.index > 0),
                        (self.next_btn, 0 <= self.index < n - 1)):
            btn.state(["!disabled"] if ok else ["disabled"])

    def _prepared(self, path, channel):
        """The levelled image of one file, from the cache when possible."""
        levelled = bool(self.level_var.get())
        key = (path, channel, levelled)
        if key in self._cache:
            return self._cache[key]
        fields = gwy_loader.load_gwy(path)
        if not fields:
            raise ValueError("no data channels")
        view = channel_view(fields[pick_channel(list(fields), channel)])
        if levelled:
            data = gp.level_by_plane_fit(view["data"])
            view["data"] = gp.align_rows(data, method="polynomial", order=2)
        self._cache[key] = view
        self._cache_order.append(key)
        while len(self._cache_order) > self.CACHE_LIMIT:
            self._cache.pop(self._cache_order.pop(0), None)
        return view

    def _thumbnail(self, data):
        step = max(1, int(np.ceil(max(data.shape) / self.THUMB)))
        return np.array(data[::step, ::step], dtype=np.float32)

    # ------------------------------------------------------------ measure --

    def analyse(self):
        """Measure every file in the folder, then balance the folder."""
        if self._busy or not self.files:
            return
        self._busy = True
        # Drop the old folder's balance first: the pass below yields to the
        # event loop to stay responsive, and a click that lands in the
        # middle of it must not draw the new files through the old numbers.
        self.result = None
        self._skipped = []
        try:
            channel = self.channel_var.get()
            params = self.settings()
            kept, measures, metas, thumbs, skipped = [], [], [], [], []
            self.progress.config(maximum=len(self.files), value=0)
            for i, path in enumerate(self.files):
                name = os.path.basename(path)
                self.status_var.set(
                    f"Measuring {i + 1}/{len(self.files)}: {name}")
                self.progress.config(value=i)
                self.update()
                try:
                    view = self._prepared(path, channel)
                    measures.append(gb.measure(view["data"], **params))
                except Exception as e:
                    skipped.append(f"{name}: {e}")
                    continue
                metas.append({k: v for k, v in view.items() if k != "data"})
                thumbs.append(self._thumbnail(view["data"]))
                kept.append(path)
            self.progress.config(value=0)
            if not kept:
                self._draw_message("None of the files in this folder could "
                                   "be read.\n\n" + "\n".join(skipped[:10]))
                self.status_var.set("Nothing to balance.")
                return
            self.files = kept
            self.channel = channel
            self.measures, self.metas, self.thumbs = measures, metas, thumbs
            self.index = min(max(self.index, 0), len(kept) - 1)
            self._skipped = skipped
            self.rebalance()
        finally:
            self._busy = False

    def rebalance(self):
        """Redo the folder-wide range (cheap - nothing is measured again)."""
        if not self.measures:
            return
        self.result = gb.balance(self.measures, self.mode())
        self.zeroed = bool(self.zero_var.get())
        if self.zeroed:
            self.result = gb.zero_baseline(self.result)
        self.override = None
        self._show_range()
        self._update_nav()
        self.redraw()
        self._report()

    def _editable_ends(self):
        """Which ends of the range the folder decides, and so can be typed
        over. In `substrate` mode that is the bottom only: each image's top
        is its own, and there is no one number to put in the box."""
        if not self.result:
            return False, False
        return bool(self.result.get("shared_min")), bool(self.result["shared"])

    def _show_range(self):
        """Put the current range into the entry boxes."""
        low_ok, high_ok = self._editable_ends()
        vmin, vmax = (self._range_for(self.index) if self.result
                      else (None, None))
        for entry, var, ok, value in (
                (self.vmin_entry, self.vmin_var, low_ok, vmin),
                (self.vmax_entry, self.vmax_var, high_ok, vmax)):
            var.set(f"{value:.4g}" if ok else "")
            entry.config(state="normal" if ok else "disabled")
        self.units_label.config(text=self.metas[0]["z_units"]
                                if self.metas else "")

    def _range_for(self, index):
        low, high = self.result["ranges"][index]
        if self.override:
            low = self.override[0] if self.override[0] is not None else low
            high = self.override[1] if self.override[1] is not None else high
        return low, high

    def apply_range(self):
        """Draw everything with the range typed into the boxes."""
        low_ok, high_ok = self._editable_ends()
        if not (low_ok or high_ok):
            return
        # A rejected entry leaves the range that was already in force - both
        # the numbers and the picture - rather than dropping back to auto.
        previous = self.override
        try:
            low = float(self.vmin_var.get()) if low_ok else None
            high = float(self.vmax_var.get()) if high_ok else None
        except ValueError:
            self.status_var.set("The range needs a number.")
            self._show_range()
            return
        self.override = (low, high)
        if any(hi <= lo for lo, hi in
               (self._range_for(i) for i in range(len(self.files)))):
            self.override = previous
            self.status_var.set("The top of the range must be above the "
                                "bottom, in every image.")
            self._show_range()
            return
        self.redraw()
        self._report()

    def auto_range(self):
        """Go back to the range the folder's measurements give."""
        if not self.result:
            return
        self.override = None
        self._show_range()
        self.redraw()
        self._report()

    def _report(self):
        """Say what the balance did, and what it had trouble with."""
        if not self.result:
            return
        r = self.result
        units = self.metas[0]["z_units"] if self.metas else ""
        bad = sum(m["degenerate"] for m in self.measures)
        parts = [f"{len(self.files)} images, {self.channel}"]
        typed = " (typed in)" if self.override else ""
        vmin, vmax = self._range_for(self.index)
        if r["shared"]:
            parts.append(f"range {vmin:.4g} to {vmax:.4g} {units}{typed}")
        elif r.get("shared_min"):
            tops = [hi for _, hi in
                    (self._range_for(i) for i in range(len(self.files)))]
            parts.append(f"range from {vmin:.4g} {units}{typed} for all, up to "
                         f"each image's own cells ({min(tops):.4g} to "
                         f"{max(tops):.4g} {units})")
        else:
            parts.append("each image on its own range")
        outside = [100 * float(np.mean((d < self._range_for(i)[0])
                                       | (d > self._range_for(i)[1])))
                   for i, d in enumerate(self._shifted_thumbs())]
        parts.append(f"{min(outside):.0f}-{max(outside):.0f}% of pixels "
                     f"outside the range")
        if bad:
            parts.append(f"{bad} image(s) with no substrate in frame - "
                         f"lower quartile used instead")
        if getattr(self, "_skipped", None):
            parts.append(f"{len(self._skipped)} unreadable file(s) skipped")
        self.status_var.set(".  ".join(parts) + ".")

    # ------------------------------------------------------------ display --

    def refresh_display(self):
        """Redraw with the current colour map (nothing is recomputed)."""
        if self.result:
            self.redraw()

    def redraw(self):
        if not self.result:
            return
        self._show_range()
        view = self.view_var.get()
        try:
            if view == "Contact sheet":
                self._draw_sheet()
            elif view == "Diagnostics":
                self._draw_diagnostics()
            else:
                self._draw_single()
        except Exception as e:                  # a file vanished mid-session
            self._draw_message(f"Could not draw this image:\n{e}")

    def _balanced(self, index):
        """One image on the balanced scale, with its metadata."""
        view = self._prepared(self.files[index], self.channel)
        data = gb.apply_levels(view["data"], self.result["offsets"][index])
        return data, view

    def _shifted_thumbs(self):
        """The contact-sheet thumbnails on the balanced scale."""
        return [gb.apply_levels(thumb, offset) for thumb, offset
                in zip(self.thumbs, self.result["offsets"])]

    def _z_label(self, index):
        units = self.metas[index]["z_units"]
        if self.result["offsets"][index] and not getattr(self, "zeroed",
                                                         False):
            return f"{units} above the substrate"
        return units

    def _draw_single(self):
        index = self.index
        data, view = self._balanced(index)
        name = os.path.basename(self.files[index])
        vmin, vmax = self._range_for(index)
        self.name_label.config(
            text=f"{index + 1}/{len(self.files)}  -  {name}")

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(data, origin="upper", cmap=gcm.current(),
                       extent=(0, view["x_real"], 0, view["y_real"]),
                       aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(self.channel)
        ax.set_xlabel(f"x ({view['spatial_units']})")
        ax.set_ylabel(f"y ({view['spatial_units']})")
        self.figure.colorbar(im, ax=ax, pad=0.05, fraction=0.046).set_label(
            self._z_label(index))
        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_sheet(self):
        """Every image at once, on the shared range."""
        n = len(self.thumbs)
        cols = min(5, max(1, int(np.ceil(np.sqrt(n)))))
        rows = int(np.ceil(n / cols))
        if self.result["shared"]:
            how = ", one shared range"
        elif self.result.get("shared_min"):
            how = ", shared bottom, each image's own top"
        else:
            how = ", each on its own range"
        self.name_label.config(text=f"{n} images, {self.channel}{how}")
        self.figure.clf()
        axes = np.atleast_1d(self.figure.subplots(rows, cols)).ravel()
        for i, data in enumerate(self._shifted_thumbs()):
            ax = axes[i]
            vmin, vmax = self._range_for(i)
            ax.imshow(data, origin="upper", cmap=gcm.current(), vmin=vmin,
                      vmax=vmax, aspect="equal")
            ax.set_title(os.path.basename(self.files[i])[:24], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == self.index:              # mark where Next/Back is
                for side in ax.spines.values():
                    side.set(color="tab:cyan", linewidth=2.5)
        for ax in axes[n:]:
            ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_diagnostics(self):
        """What the segmentation found, and how the folder sits in the
        range."""
        index = self.index
        data, view = self._balanced(index)
        measure = self.measures[index]
        vmin, vmax = self._range_for(index)
        offset = self.result["offsets"][index]
        self.name_label.config(
            text=f"{index + 1}/{len(self.files)}  -  "
                 f"{os.path.basename(self.files[index])}")

        self.figure.clf()
        axes = self.figure.subplots(2, 2)

        ax = axes[0, 0]
        ax.imshow(data, origin="upper", cmap=gcm.current(), vmin=vmin,
                  vmax=vmax, aspect="equal")
        ax.contour(measure["mask"], levels=[0.5], colors="tab:cyan",
                   linewidths=0.8)
        ax.set_title(f"cells {100 * measure['coverage']:.0f}% of the frame"
                     + (" (no substrate found)" if measure["degenerate"]
                        else ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[0, 1]
        anchors = [(measure["background"], "substrate", "tab:blue"),
                   (measure["low"], "cell low", "tab:green"),
                   (measure["median"], "cell median", "tab:olive"),
                   (measure["high"], "cell high", "tab:red")]
        lo, hi = np.percentile(data, [0.2, 99.8])
        ax.hist(data.ravel(), bins=300, range=(lo, hi), color="0.7")
        for value, label, colour in anchors:
            ax.axvline(value + offset, color=colour, lw=1.2, label=label)
        ax.axvspan(vmin, vmax, color="tab:orange", alpha=0.15,
                   label="shown range")
        ax.set_title("where this image sits", fontsize=9)
        ax.set_xlabel(self._z_label(index))
        ax.set_yticks([])
        ax.legend(fontsize=7)

        ax = axes[1, 0]
        x = np.arange(len(self.measures))
        for key, label, colour in (("background", "substrate", "tab:blue"),
                                   ("low", "cell low", "tab:green"),
                                   ("median", "cell median", "tab:olive"),
                                   ("high", "cell high", "tab:red")):
            y = [m[key] + o for m, o
                 in zip(self.measures, self.result["offsets"])]
            ax.plot(x, y, ".-", color=colour, lw=1, ms=4, label=label)
        ax.axhspan(vmin, vmax, color="tab:orange", alpha=0.15)
        ax.axvline(index, color="0.4", lw=1, ls=":")
        ax.set_title("the folder on the balanced scale", fontsize=9)
        ax.set_xlabel("image")
        ax.set_ylabel(self._z_label(index))
        ax.legend(fontsize=7)

        # How much each image loses to the range - the number to watch when
        # deciding whether one shared range is doing more harm than good.
        ax = axes[1, 1]
        shifted = self._shifted_thumbs()
        below = [100 * float(np.mean(d < self._range_for(i)[0]))
                 for i, d in enumerate(shifted)]
        above = [100 * float(np.mean(d > self._range_for(i)[1]))
                 for i, d in enumerate(shifted)]
        ax.plot(x, [100 * m["coverage"] for m in self.measures], ".-",
                color="tab:purple", lw=1, ms=4, label="cell coverage (%)")
        ax.plot(x, below, ".-", color="tab:blue", lw=1, ms=4,
                label="clipped dark (%)")
        ax.plot(x, above, ".-", color="tab:red", lw=1, ms=4,
                label="clipped bright (%)")
        ax.axvline(index, color="0.4", lw=1, ls=":")
        ax.set_title("per image", fontsize=9)
        ax.set_xlabel("image")
        ax.legend(fontsize=7)

        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_message(self, text):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    # ------------------------------------------------------------- export --

    def _targets(self, folder, choice):
        """Where each image would be written, as {index: {kind: path}}."""
        pure_dir = os.path.join(folder, "pure")
        out = []
        for path in self.files:
            base = f"{os.path.splitext(os.path.basename(path))[0]}_balanced"
            here = {}
            if choice["annotated"]:
                here["annotated"] = os.path.join(folder, base + ".png")
            if choice["pure"]:
                here["pure"] = os.path.join(pure_dir, base + ".png")
            if choice["gwy"]:
                here["gwy"] = os.path.join(folder, base + ".gwy")
            out.append(here)
        return out

    def export(self):
        """Write the whole folder out, all of it on the shared range."""
        if not self.result:
            messagebox.showinfo("Nothing to export", "Select a folder first.")
            return
        dialog = ExportChoiceDialog(self.winfo_toplevel())
        choice = dialog.result
        if not choice:
            return
        if not any(choice.values()):
            self.status_var.set("Nothing selected to export.")
            return

        folder = filedialog.askdirectory(
            title="Save the balanced folder to...",
            initialdir=self.folder or ".")
        if not folder:
            return
        targets = self._targets(folder, choice)
        clashes = [p for t in targets for p in t.values()
                   if os.path.exists(p)]
        if clashes and not messagebox.askyesno(
                "Overwrite?",
                f"{len(clashes)} file(s) will be overwritten, starting with "
                f"{os.path.basename(clashes[0])}.\n\nGo ahead?"):
            return
        if choice["pure"]:
            os.makedirs(os.path.join(folder, "pure"), exist_ok=True)

        self._busy = True
        written = 0
        try:
            self.progress.config(maximum=len(self.files), value=0)
            for i, target in enumerate(targets):
                name = os.path.basename(self.files[i])
                self.status_var.set(
                    f"Writing {i + 1}/{len(self.files)}: {name}")
                self.progress.config(value=i)
                self.update()
                written += self._export_one(i, target)
            self.progress.config(value=0)
            kinds = ", ".join(k for k in ("annotated", "pure", "gwy")
                              if choice[k])
            self.status_var.set(f"Wrote {written} file(s) ({kinds}) for "
                                f"{len(self.files)} images to {folder}.")
        except Exception as e:
            self.status_var.set(f"Export stopped after {written} file(s): {e}")
            messagebox.showerror("Export error", str(e))
        finally:
            self._busy = False

    def _export_one(self, index, target):
        """Write one image in every form asked for; returns the file count."""
        data, view = self._balanced(index)
        vmin, vmax = self._range_for(index)
        base = os.path.splitext(os.path.basename(self.files[index]))[0]

        if "annotated" in target:
            figure = render_annotated_figure(
                data, view["x_real"], view["y_real"],
                f"{base} - {self.channel}", view["spatial_units"],
                self._z_label(index), vmin=vmin, vmax=vmax)
            figure.savefig(target["annotated"], dpi=200, bbox_inches="tight")
        if "pure" in target:
            save_pure_image(data, target["pure"], view["x_real"],
                            view["y_real"], vmin=vmin, vmax=vmax)
        if "gwy" in target:
            # Appending is what save_channel_to_gwy does with an existing
            # file; here the export should be repeatable, so the old file
            # goes first - the overwrite was confirmed above.
            if os.path.exists(target["gwy"]):
                os.remove(target["gwy"])
            save_channel_to_gwy(
                target["gwy"], f"{base} - {self.channel} (balanced)",
                data / view["z_factor"],              # back to SI units
                xreal=view["x_real"] / view["xy_factor"],
                yreal=view["y_real"] / view["xy_factor"],
                unit_xy=view["unit_xy_str"], unit_z=view["unit_z_str"],
            )
        return len(target)


class GwyProcessorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GWY Processor")
        self.geometry("1250x880")

        # --- State ---
        self.filename = None
        self.channels = {}          # {title: GwyDataField}
        self.field = None           # currently selected GwyDataField
        self.original_data = None   # data as loaded (display units)
        self.data = None            # current processed data (display units)
        self.undo_stack = []        # list of state snapshots (see _snapshot)
        self.redo_stack = []        # snapshots popped by undo, for redo
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
        # Two tabs: the processing workbench and the folder quick view.
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        main_tab = ttk.Frame(self.tabs)
        self.tabs.add(main_tab, text="Processing")

        # Left: controls, Right: plot
        left = ttk.Frame(main_tab, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(main_tab)
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

        # ---- Display section: false-colour gradient ----
        disp = ttk.LabelFrame(left, text="Display", padding=6)
        disp.pack(fill=tk.X, pady=(0, 6))
        row = ttk.Frame(disp)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Colour map:").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value=gcm.current_name())
        self.cmap_combo = ttk.Combobox(
            row, textvariable=self.cmap_var, state="readonly",
            values=gcm.names(), height=20,
        )
        self.cmap_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.select_cmap())
        self.cmap_strip = tk.Canvas(disp, height=16, highlightthickness=1,
                                    highlightbackground="#909090")
        self.cmap_strip.pack(fill=tk.X, pady=(4, 0))
        self.cmap_strip.bind("<Configure>", lambda e: self._draw_cmap_strip())

        # ---- Operations section: one button per dialog ----
        proc = ttk.LabelFrame(left, text="Operations", padding=6)
        proc.pack(fill=tk.X, pady=(0, 6))

        for row in OPERATION_ROWS:
            line = ttk.Frame(proc)
            line.pack(fill=tk.X, pady=1)
            for i, key in enumerate(row):
                if key == "@fft_spectrum":
                    btn = ttk.Button(line, text="View FFT spectrum",
                                     command=self.show_fft)
                else:
                    suffix = "" if OPERATIONS[key].get("instant") else "..."
                    btn = ttk.Button(
                        line,
                        text=OPERATIONS[key]["label"] + suffix,
                        command=lambda k=key: self.open_operation(k),
                    )
                btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                         padx=(0, 2) if i < len(row) - 1 else 0)

        # ---- Undo / reset ----
        hist = ttk.Frame(left)
        hist.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(hist, text="Undo", command=self.undo).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(hist, text="Redo", command=self.redo).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(hist, text="Reset to original", command=self.reset).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # The status line and the output buttons are anchored to the bottom
        # and claimed before the log, so a short window shrinks the log
        # instead of pushing the buttons off the screen.
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, wraplength=260).pack(
            side=tk.BOTTOM, fill=tk.X, pady=(6, 0)
        )

        # ---- Save / batch ----
        out = ttk.LabelFrame(left, text="Output", padding=6)
        out.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(out, text="Save processed image...", command=self.save_image).pack(
            fill=tk.X, pady=1
        )
        ttk.Button(out, text="Save channel to .gwy...", command=self.save_to_gwy).pack(
            fill=tk.X, pady=1
        )
        ttk.Button(out, text="Batch process folder...", command=self.batch_dialog).pack(
            fill=tk.X, pady=1
        )

        # ---- Log section ----
        log_frame = ttk.LabelFrame(left, text="Processing log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.log_list = tk.Listbox(log_frame, height=6)
        self.log_list.pack(fill=tk.BOTH, expand=True)
        ttk.Button(log_frame, text="Save log...", command=self.save_log).pack(
            fill=tk.X, pady=(4, 0)
        )

        # ---- Plot area ----
        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

        # ---- Folder tabs ----
        self.quick = QuickViewTab(self.tabs, self)
        self.tabs.add(self.quick, text="Quick view")
        self.balanced = BalancedViewTab(self.tabs, self)
        self.tabs.add(self.balanced, text="Balanced view")

    # ------------------------------------------------------------- Colour --

    def select_cmap(self):
        """Switch the false-colour gradient used for topography everywhere."""
        name = gcm.set_current(self.cmap_var.get())
        self.cmap_var.set(name)
        self._draw_cmap_strip()
        if self.data is not None:
            self.redraw()
        self.quick.refresh_display()
        self.balanced.refresh_display()
        self.status_var.set(f"Colour map: {name}")

    def _draw_cmap_strip(self):
        """Paint the selected gradient as a strip under the combo box."""
        canvas = self.cmap_strip
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 2:
            return
        cmap = gcm.current()
        for x in range(width):
            r, g, b, _ = cmap(x / max(width - 1, 1))
            canvas.create_line(
                x, 0, x, height,
                fill=f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}",
            )

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

        view = channel_view(self.field)
        self.spatial_units = view["spatial_units"]
        self.z_units = view["z_units"]
        self.z_factor = view["z_factor"]        # display units -> SI
        self.unit_xy_str = view["unit_xy_str"]  # original SI unit strings,
        self.unit_z_str = view["unit_z_str"]    # kept for .gwy export

        data = view["data"]
        ny, nx = data.shape
        self.x_real = view["x_real"]
        self.y_real = view["y_real"]
        self.dx = self.x_real / nx
        self.dy = self.y_real / ny

        self.original_data = data.copy()
        self._orig_extent = (self.x_real, self.y_real)
        self.data = data
        self.undo_stack = []
        self.redo_stack = []
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

    def add_channel(self, title, data, template=None, pipeline_step=None,
                    select=True):
        """Insert a derived image as a new in-memory channel and (by default)
        switch editing to it. The file on disk is not touched; use
        'Save channel to .gwy...' to write it out.

        `data` is in display units and is stored back in SI units, so the new
        channel behaves exactly like one read from the file.

        `pipeline_step` is an (op_key, params) pair recorded as the first step
        of the new channel's pipeline, so a batch run can reproduce it.

        Returns the title actually used (uniquified if it was taken)."""
        template = template or self.field
        z_factor = getattr(self, "z_factor", 1.0) or 1.0

        unique = title
        n = 2
        while unique in self.channels:
            unique = f"{title} ({n})"
            n += 1

        ny, nx = data.shape
        t_ny, t_nx = template.data.shape
        field = gwy_loader.GwyDataField(
            np.ascontiguousarray(np.asarray(data, dtype=np.float64) / z_factor),
            xreal=nx * float(template.xreal or t_nx) / t_nx,
            yreal=ny * float(template.yreal or t_ny) / t_ny,
            si_unit_xy=_unit_of(template, "si_unit_xy") or None,
            si_unit_z=_unit_of(template, "si_unit_z") or None,
        )
        self.channels[unique] = field
        self.channel_combo["values"] = list(self.channels)

        if select:
            # select_channel() resets the history, so suppress its prompt by
            # clearing the pipeline first - the merge is a fresh starting point
            self.pipeline = []
            self.channel_var.set(unique)
            self.select_channel()
            if pipeline_step is not None:
                # record it so 'Batch process folder' replays the merge on
                # every file, re-measuring the shift for each one
                self.pipeline.append(pipeline_step)
                self._log(describe_step(*pipeline_step))
        return unique

    def channel_context(self):
        """Extra channels handed to operations that need more than the current
        image (the two-way merge needs the forward/backward pair)."""
        if not self.channels:
            return None
        fwd_title, bwd_title = gtw.find_pair(self.channels, self.channel_var.get())
        z = getattr(self, "z_factor", 1.0)
        context = {"fwd_title": fwd_title, "bwd_title": bwd_title,
                   "fwd": None, "bwd": None, "channels": self.channels}
        if fwd_title in self.channels:
            context["fwd"] = self.channels[fwd_title].data.astype(np.float64) * z
        if bwd_title in self.channels:
            context["bwd"] = self.channels[bwd_title].data.astype(np.float64) * z
        return context

    def apply_operation(self, op_key, params):
        """Apply one operation, push undo state, record pipeline + log.
        Called by the operation dialogs on Apply."""
        spec = OPERATIONS[op_key]
        func = spec["func"]
        try:
            if spec.get("needs_pair"):
                new_data = func(self.data, params, self.dx, self.dy,
                                self.channel_context())
            else:
                new_data = func(self.data, params, self.dx, self.dy)
        except Exception as e:
            messagebox.showerror(
                "Processing error", f"{describe_step(op_key, params)} failed:\n{e}"
            )
            return
        self._push_undo()
        self.data = new_data
        # Operations like crop change the image dimensions; keep the
        # physical extents consistent (pixel size dx/dy never changes).
        ny, nx = new_data.shape
        self.x_real = nx * self.dx
        self.y_real = ny * self.dy
        self.pipeline.append((op_key, params))
        self._log(describe_step(op_key, params))
        self.redraw()

    def reapply_last_operation(self, op_key, params):
        """Replace the last pipeline step with a new parameter set: the image
        goes back to the state before that step and the operation is applied
        again. Used by dialogs that re-edit their own last step (the range
        clip) instead of stacking a second one on top of it, which for a
        clip could only ever narrow the range further."""
        if not (self.pipeline and self.undo_stack
                and self.pipeline[-1][0] == op_key):
            self.apply_operation(op_key, params)
            return
        old = self.pipeline[-1]
        self._restore(self.undo_stack.pop())
        self._log(f"UNDO (re-edit): {describe_step(*old)}")
        self.apply_operation(op_key, params)

    def show_fft(self):
        """Open a window showing the current FFT magnitude spectrum."""
        if not self._require_data():
            return
        mag, extent = gp.get_2d_fft_magnitude(self.data, dx=self.dx, dy=self.dy)
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

    # ------------------------------------------------ Undo / redo / logging --

    def _snapshot(self):
        """The whole editing state, so undo/redo restore the pipeline too and
        not just the pixels (Reset clears the pipeline, for instance)."""
        return (self.data, self.x_real, self.y_real, list(self.pipeline))

    def _restore(self, snapshot):
        self.data, self.x_real, self.y_real, pipeline = snapshot
        self.pipeline = list(pipeline)

    def _push_undo(self):
        """Remember the current state before changing it. A new change makes
        whatever was undone unreachable, so the redo stack is dropped."""
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        target = self.undo_stack.pop()
        undone = (describe_step(*self.pipeline[-1])
                  if len(self.pipeline) > len(target[3]) else "previous state")
        self.redo_stack.append(self._snapshot())
        self._restore(target)
        self._log(f"UNDO: {undone}")
        self.redraw()

    def redo(self):
        if not self.redo_stack:
            self.status_var.set("Nothing to redo")
            return
        target = self.redo_stack.pop()
        redone = (describe_step(*target[3][-1])
                  if len(target[3]) > len(self.pipeline) else "next state")
        self.undo_stack.append(self._snapshot())
        self._restore(target)
        self._log(f"REDO: {redone}")
        self.redraw()

    def reset(self):
        if self.original_data is None:
            return
        if not self.pipeline and not self.undo_stack:
            return
        self._push_undo()
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
            cmap=gcm.current(),
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
        so all processed channels can be collected in one Gwyddion file.

        Every other channel of the loaded image is written along with it, so
        the saved file is a complete copy of the measurement plus the
        processed result. The name defaults to the source file with a
        '_processed' suffix, next to the original."""
        if not self._require_data():
            return
        base = os.path.splitext(os.path.basename(self.filename or "image"))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".gwy",
            filetypes=[("Gwyddion files", "*.gwy")],
            initialfile=f"{base}_processed.gwy",
            initialdir=os.path.dirname(self.filename or "") or ".",
            confirmoverwrite=False,  # existing files are appended to, not replaced
        )
        if not path:
            return
        title = f"{base} - {self.channel_var.get()} (processed)"
        # The processed data may have been cropped, so its physical size is
        # the source size scaled by the shape ratio, not the source size.
        ny, nx = self.data.shape
        t_ny, t_nx = self.field.data.shape
        try:
            n, title, extras = save_channel_to_gwy(
                path, title,
                self.data / self.z_factor,  # back to SI units
                xreal=nx * float(self.field.xreal or t_nx) / t_nx,
                yreal=ny * float(self.field.yreal or t_ny) / t_ny,
                unit_xy=self.unit_xy_str, unit_z=self.unit_z_str,
                extra_channels=list(self.channels.items()),
            )
        except Exception as e:
            messagebox.showerror("Save error", f"Could not write .gwy file:\n{e}")
            return
        extra_txt = f" (+ {len(extras)} original channels)" if extras else ""
        self._log(f"Saved channel {n} '{title}'{extra_txt} to "
                  f"{os.path.basename(path)}")
        self.status_var.set(
            f"Appended channel {n}{extra_txt} to {os.path.basename(path)}")

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

                # Two-way operations need this file's own forward/backward
                # pair; the shift is re-measured for every image because the
                # scanner lag differs from scan to scan.
                fwd_title, bwd_title = gtw.find_pair(channels, channel)
                context = {
                    "fwd_title": fwd_title, "bwd_title": bwd_title,
                    "fwd": (channels[fwd_title].data.astype(np.float64) * z_factor
                            if fwd_title in channels else data),
                    "bwd": (channels[bwd_title].data.astype(np.float64) * z_factor
                            if bwd_title in channels else None),
                    "channels": channels,
                }

                processed = apply_pipeline(data, pipeline, dx, dy, context)

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
