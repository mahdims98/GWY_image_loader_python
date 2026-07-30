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
      * Two-way merge of the forward and backward scans (gwy_twoway):
        scanner lag / hysteresis alignment, parachuting-artifact
        detection and soft-min merging
  - Keep a log of every change applied to the image
  - Undo changes step by step (or reset to the original data)
  - Batch-process every .gwy file in a folder by replaying the
    current processing pipeline on the selected channel

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
import matplotlib.image as mpimage
import matplotlib.patheffects as patheffects
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector, SpanSelector

import gwy_loader
import gwy_processing as gp
import gwy_twoway as gtw


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


def twoway_kwargs(params, detect=False):
    """Translate a dialog's flat parameter dict into gwy_twoway keywords.
    Tolerates missing keys (the merge and parachuting dialogs expose
    different subsets), falling back to the gwy_twoway defaults."""
    g = params.get
    flip = {"auto": "auto", "yes": True, "no": False}[g("flip_backward", "auto")]
    manual = g("slope_mode", "manual") == "manual"
    return dict(
        mapping=g("mapping", "xcorr"),
        warp=g("warp", "bwd_to_fwd"),
        poly_order=int(g("poly_order", 2)),
        n_blocks=int(g("n_blocks", 16)),
        max_lag=int(g("max_lag", 20)),
        match_level=g("match_level", "plane"),
        match_poly_order=int(g("match_poly_order", 2)),
        flip_backward=flip,
        crop=bool(g("crop", True)),
        detect=detect,
        slope=float(g("slope", 1.0)) if (detect and manual) else None,
        slope_scale=float(g("slope_scale", 1.0)),
        offset=float(g("offset", 0.0)),
        max_delta=int(g("max_delta", 20)),
        combine=g("combine", "average"),
        weight=float(g("weight", 0.5)),
        slope_gain=float(g("slope_gain", 2.0)),
        consensus_size=int(g("consensus_size", 5)),
        beta=float(g("beta", 0.0)),
        both_flagged=g("both_flagged", "paper"),
        corr_margin=float(g("corr_margin", 0.7)),
        corr_window=int(g("corr_window", 11)),
        corr_combine=g("corr_combine", "average"),
    )


def twoway_param_relevant(name, p):
    """Whether a two-way / parachuting dialog parameter has any effect under
    the currently selected dropdown choices. Used to hide the irrelevant
    parameter rows; unknown names are always relevant."""
    g = p.get
    mapping = g("mapping", "xcorr")
    combine = g("combine", "average")
    corr = combine == "correlation"
    corr_combine = g("corr_combine", "average") if corr else None
    measured = mapping in ("xcorr", "model_scaled", "measured")
    rules = {
        # alignment
        "poly_order": mapping == "xcorr",
        "n_blocks": measured,
        "max_lag": measured,
        "match_level": mapping != "none",
        "match_poly_order": (mapping != "none"
                             and g("match_level", "plane") == "poly_rows"),
        # merge
        "weight": combine == "average" or corr_combine == "average",
        "slope_gain": combine == "slope" or corr_combine == "slope",
        "consensus_size": (combine == "consensus"
                           or corr_combine == "consensus"),
        "beta": combine == "softmin" or corr_combine == "softmin",
        "corr_margin": corr,
        "corr_window": corr,
        "corr_aux": corr,
        "corr_combine": corr,
        # parachuting detection
        "slope": g("slope_mode", "manual") == "manual",
        "slope_scale": g("slope_mode", "manual") == "auto",
    }
    return rules.get(name, True)


#: Auxiliary channel base names consulted by the correlation merge, keyed by
#: the ``corr_aux`` dialog choice.
AUX_CHOICES = {
    "phase+error": ("Phase", "Error"),
    "phase": ("Phase",),
    "error": ("Error",),
    "none": (),
}


def aux_pairs_for(channels, fwd_title, which="phase+error"):
    """The raw ``(name, fwd, bwd)`` auxiliary-channel triples (phase / error)
    that belong to a height channel, as referee data for
    ``combine='correlation'``. Missing channels are silently skipped - the
    merge falls back to its consensus rule when no referee is available."""
    triples = []
    if not channels or not fwd_title:
        return triples
    for name in AUX_CHOICES.get(which, ()):
        aux_f = re.sub(r"^.*?(?=\s*\[)", name, fwd_title)
        if aux_f == fwd_title or aux_f not in channels:
            continue
        aux_b = gtw.backward_title(aux_f)
        if aux_b and aux_b in channels:
            triples.append((name,
                            channels[aux_f].data.astype(np.float64),
                            channels[aux_b].data.astype(np.float64)))
    return triples


def _require_pair(context):
    if not context or context.get("bwd") is None:
        raise ValueError(
            "This operation needs a forward AND a backward channel "
            "(e.g. 'Height [Fwd]' and 'Height [Bwd]')."
        )


def _op_two_way(data, params, dx, dy, context=None):
    """Merge the forward and backward scans of the current file.

    This operation ignores the incoming `data` - it always starts from the raw
    forward/backward channel pair supplied in `context` - so it belongs at the
    very start of a pipeline."""
    _require_pair(context)
    aux = None
    if params.get("combine") == "correlation":
        aux = [(f, b) for _, f, b in aux_pairs_for(
            context.get("channels"), context.get("fwd_title"),
            params.get("corr_aux", "phase+error"))]
    result = gtw.process_two_way(context["fwd"], context["bwd"],
                                 aux_pairs=aux,
                                 **twoway_kwargs(params, detect=False))
    return result.merged


def _op_parachute(data, params, dx, dy, context=None):
    """Parachuting-artifact removal: align the forward/backward pair, flag the
    airborne-tip pixels in each scan, replace them from the opposite scan and
    combine the rest. Starts from the raw channel pair, like _op_two_way."""
    _require_pair(context)
    result = gtw.process_two_way(context["fwd"], context["bwd"],
                                 **twoway_kwargs(params, detect=True))
    return result.merged


def _describe_combine(params):
    combine = params["combine"]
    if combine == "average":
        w = float(params["weight"])
        return f"average {w:.2f} fwd / {1 - w:.2f} bwd"
    if combine == "softmin":
        return f"softmin beta={params['beta']}"
    if combine == "slope":
        return f"slope-select gain={params.get('slope_gain', 2.0)}"
    if combine == "consensus":
        return f"consensus size={params.get('consensus_size', 5)}"
    if combine == "correlation":
        shared = params.get("corr_combine", "average")
        if shared == "softmin":
            shared += f" beta={params.get('beta', 0.0)}"
        return (f"correlation margin={params.get('corr_margin', 0.7)}, "
                f"win={params.get('corr_window', 11)}px, "
                f"referee={params.get('corr_aux', 'phase+error')}, "
                f"shared={shared}")
    return combine


def _describe_two_way(params):
    parts = [f"map={params['mapping']}"]
    if params["mapping"] == "xcorr":
        parts.append(f"order={params['poly_order']}")
    parts.append(f"warp={params['warp']}")
    parts.append(_describe_combine(params))
    if params.get("crop", True):
        parts.append("cropped")
    return ", ".join(parts)


def _describe_parachute(params):
    slope = ("auto x" + str(params.get("slope_scale", 1.0))
             if params.get("slope_mode") == "auto" else str(params["slope"]))
    parts = [
        f"map={params['mapping']}",
        f"slope={slope}",
        f"offset={params['offset']}",
        f"max delta={params['max_delta']}",
        _describe_combine(params),
    ]
    if params.get("crop", True):
        parts.append("cropped")
    return ", ".join(parts)


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
    # Two-way (forward/backward) operations. Not in OPERATION_ORDER: they need
    # a channel *pair* rather than the single current image, so they get their
    # own buttons and dialogs, and `needs_pair` tells apply_pipeline to hand
    # them the forward/backward context.
    "two_way": {
        "label": "Two-way merge (Fwd/Bwd)",
        "func": _op_two_way,
        "needs_pair": True,
        "channel_suffix": "[Merged]",
        "params": [
            # -- hysteresis / lag alignment
            {"name": "mapping", "label": "Shift model", "type": "choice",
             "default": "xcorr",
             "values": ["xcorr", "model_scaled", "model", "measured", "none"]},
            {"name": "poly_order", "label": "Poly order", "type": "int",
             "default": 2, "min": 0, "max": 6},
            {"name": "n_blocks", "label": "Match blocks", "type": "int",
             "default": 16, "min": 1, "max": 128},
            {"name": "max_lag", "label": "Max lag (px)", "type": "int",
             "default": 20, "min": 1, "max": 200},
            {"name": "match_level", "label": "Level for match", "type": "choice",
             "default": "plane", "values": ["plane", "poly_rows", "none"]},
            {"name": "match_poly_order", "label": "Match row-poly order",
             "type": "int", "default": 2, "min": 0, "max": 10},
            {"name": "warp", "label": "Warp", "type": "choice",
             "default": "bwd_to_fwd",
             "values": ["bwd_to_fwd", "split", "linearize_both"]},
            {"name": "flip_backward", "label": "Flip backward", "type": "choice",
             "default": "auto", "values": ["auto", "yes", "no"]},
            {"name": "crop", "label": "Crop to imaged area", "type": "bool",
             "default": True},
            # -- merge
            {"name": "combine", "label": "Combine", "type": "choice",
             "default": "average",
             "values": ["average", "correlation", "slope", "consensus",
                        "softmin", "min", "max", "forward", "backward"]},
            {"name": "weight", "label": "Forward weight (0-1)", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1.0},
            {"name": "slope_gain", "label": "Slope gain", "type": "float",
             "default": 2.0, "min": 0.0, "max": 100.0},
            {"name": "consensus_size", "label": "Consensus box (px)", "type": "int",
             "default": 5, "min": 1, "max": 100},
            {"name": "beta", "label": "Soft-min beta (1/z)", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1e6},
            # -- combine='correlation' only
            {"name": "corr_margin", "label": "Corr margin (0-1)", "type": "float",
             "default": 0.7, "min": -1.0, "max": 1.0},
            {"name": "corr_window", "label": "Corr window (px)", "type": "int",
             "default": 11, "min": 3, "max": 101},
            {"name": "corr_aux", "label": "Referee channels", "type": "choice",
             "default": "phase+error",
             "values": ["phase+error", "phase", "error", "none"]},
            {"name": "corr_combine", "label": "Shared combine", "type": "choice",
             "default": "average",
             "values": ["average", "softmin", "slope", "consensus",
                        "min", "max"]},
        ],
        "removed_label": "Difference (forward - merged)",
        "describe": _describe_two_way,
    },
    "parachute": {
        "label": "Parachuting removal (Fwd/Bwd)",
        "func": _op_parachute,
        "needs_pair": True,
        "channel_suffix": "[Deparachuted]",
        "params": [
            # -- alignment (kept minimal; tune it in the two-way merge dialog)
            {"name": "mapping", "label": "Shift model", "type": "choice",
             "default": "xcorr",
             "values": ["xcorr", "model_scaled", "model", "measured", "none"]},
            {"name": "poly_order", "label": "Poly order", "type": "int",
             "default": 2, "min": 0, "max": 6},
            {"name": "crop", "label": "Crop to imaged area", "type": "bool",
             "default": True},
            # -- parachuting detection
            {"name": "slope_mode", "label": "Fall rate", "type": "choice",
             "default": "manual", "values": ["manual", "auto"]},
            {"name": "slope", "label": "Slope (z/px)", "type": "float",
             "default": 1.0, "min": 0.0, "max": 1e9},
            {"name": "slope_scale", "label": "Auto scale", "type": "float",
             "default": 1.0, "min": 0.01, "max": 10.0},
            {"name": "offset", "label": "Offset (z)", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1e9},
            {"name": "max_delta", "label": "Max lag delta (px)", "type": "int",
             "default": 20, "min": 1, "max": 200},
            # -- merge of the unflagged pixels
            {"name": "combine", "label": "Combine", "type": "choice",
             "default": "average",
             "values": ["average", "slope", "consensus", "softmin",
                        "min", "max", "forward", "backward"]},
            {"name": "weight", "label": "Forward weight (0-1)", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1.0},
            {"name": "beta", "label": "Soft-min beta (1/z)", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1e6},
            {"name": "both_flagged", "label": "Both flagged", "type": "choice",
             "default": "paper", "values": ["paper", "min", "softmin"]},
        ],
        "removed_label": "Difference (forward - result)",
        "describe": _describe_parachute,
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


def apply_pipeline(data, pipeline, dx, dy, context=None):
    """Apply a list of (op_key, params) steps to `data` and return the result.

    `context` carries extra channels for operations that need more than the
    current image (currently the two-way merge, which needs the forward and
    backward pair); see the `needs_pair` flag in OPERATIONS."""
    for op_key, params in pipeline:
        spec = OPERATIONS[op_key]
        if spec.get("needs_pair"):
            data = spec["func"](data, params, dx, dy, context)
        else:
            data = spec["func"](data, params, dx, dy)
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


class ZoomWindow(tk.Toplevel):
    """A large side-by-side view of one region of the forward, backward, and
    merged images, so edges can be inspected close up. The region is picked
    by dragging a rectangle on the Forward panel of the parent dialog; until
    one is picked the full images are shown. The three views share one color
    scale so heights are directly comparable, and the window follows every
    preview update and display-leveling change."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.title("Zoom - drag a rectangle on the Forward panel "
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
        cmap = gp.get_gwyddion_cmap()
        im = None
        for ax, (title, img) in zip(axes, panels):
            im = ax.imshow(img, origin="upper", cmap=cmap, extent=extent,
                           aspect="equal", vmin=v0, vmax=v1)
            ax.set_title(title, fontsize=10)
        self.figure.colorbar(im, ax=list(axes), fraction=0.03,
                             pad=0.02).set_label(z_units)
        self.figure.suptitle(subtitle, fontsize=10)
        self.canvas.draw()


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
                       cmap=gp.get_gwyddion_cmap(), extent=extent,
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


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class TwoWayDialog(tk.Toplevel):
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
        ("Alignment (hysteresis + lag)",
         ["mapping", "poly_order", "n_blocks", "max_lag", "match_level",
          "match_poly_order", "warp", "flip_backward", "crop"]),
        ("Merge",
         ["combine", "corr_combine", "weight", "slope_gain",
          "consensus_size", "beta",
          "corr_margin", "corr_window", "corr_aux"]),
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
        self._zoom_rect = None      # (x0, x1, y0, y1) in physical units
        self._zoom_win = None
        self._zoom_selector = None
        self._corr_win = None
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
                     values=["blend", "anaglyph", "corr map", "decision"],
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
        ttk.Button(frame, text="Correlation details...",
                   command=self.open_corr_window).grid(
            row=6, column=0, columnspan=2, sticky=tk.EW, pady=1)
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
                           cmap=cm or gp.get_gwyddion_cmap(), extent=extent,
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
        if style in ("corr map", "decision"):
            if getattr(res, "corr_map", None) is None:
                ax.text(0.5, 0.5, "set Combine = 'correlation'\n"
                        "to see the correlation views",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9)
                ax.set_axis_off()
                return
            margin = float((self._last_params or {}).get("corr_margin", 0.7))
            extent = self._extent_of(res.corr_map)
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
            base = gp.get_gwyddion_cmap()(nf)
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

    # ---- Zoom on a selected area ----

    def open_zoom_window(self):
        """Open (or raise) the large zoom view of the selected area."""
        if self._zoom_win is None or not self._zoom_win.winfo_exists():
            self._zoom_win = ZoomWindow(self)
        else:
            self._zoom_win.lift()
        self._update_zoom_window()

    def _attach_zoom_selector(self, ax):
        """Drag-to-select on the Forward panel; the rectangle is shown big
        in the zoom window. Re-created on every draw (figure was cleared)."""
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
        self._draw(self._last_params)   # re-render to outline the area

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

    def _update_zoom_window(self):
        if (self._zoom_win is None or not self._zoom_win.winfo_exists()
                or self.result is None):
            return
        fwd_d, bwd_d, merged_d, tag = self._display_images()
        sl = self._zoom_slices(fwd_d.shape)
        if sl is None:
            crops = [fwd_d, bwd_d, merged_d]
            extent = self._extent_of(fwd_d)
            where = "full image (drag on the Forward panel to pick an area)"
        else:
            rows, cols = sl
            crops = [fwd_d[rows, cols], bwd_d[rows, cols],
                     merged_d[rows, cols]]
            ny = fwd_d.shape[0]
            dx, dy = self.app.dx, self.app.dy
            extent = (cols.start * dx, cols.stop * dx,
                      (ny - rows.stop) * dy, (ny - rows.start) * dy)
            where = (f"area {cols.stop - cols.start}x{rows.stop - rows.start}"
                     f" px at ({extent[0]:.3g}, {extent[2]:.3g}) "
                     f"{self.app.spatial_units}")
        titles = [f"Forward ({self.fwd_title})", "Backward, aligned", "Merged"]
        self._zoom_win.show(list(zip(titles, crops)), extent,
                            f"{where}{tag}", self.app.z_units)

    def _mark_zoom_rect(self, *axes):
        """Outline the zoomed area on the dialog's own image panels."""
        if self._zoom_rect is None:
            return
        x0, x1, y0, y1 = self._zoom_rect
        for ax in axes:
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="red", lw=1.2))

    # ---- Correlation-merge details window ----

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
        for win in (getattr(self, "_zoom_win", None),
                    getattr(self, "_corr_win", None)):
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

        for i, key in enumerate(("two_way", "parachute")):
            ttk.Button(
                proc,
                text=OPERATIONS[key]["label"] + "...",
                command=lambda k=key: self.open_operation(k),
            ).pack(fill=tk.X, pady=(6 if i == 0 else 1, 1))

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
