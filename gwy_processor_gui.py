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
      * Percentile range clipping (filter_by_percentile), re-editable:
        a clip opened right after another one edits that same step, so
        the range can be widened again and not only narrowed
      * FFT filtering (filter_by_2d_fft_mask): lowpass/highpass, circular
        notches, rectangles and straight bands combined in one dialog,
        with a large interactive spectrum (click to place cutoff/notches,
        drag to notch a rectangle), optional smooth mask edges and a
        zoom window comparing the image before and after the filter
      * Stripe removal (gwy_destripe): the multidirectional stripe
        remover of Liang et al. (2016) or the variational general
        stripe remover of Rottmayer et al. (2025), selected in the
        dialog, which shows only the chosen method's parameters
      * Scar removal (remove_scars)
      * Set baseline to zero (set_baseline_to_zero)
      * Two-way merge of the forward and backward scans (gwy_twoway):
        scanner lag / hysteresis alignment, parachuting-artifact
        detection and soft-min merging
  - Keep a log of every change applied to the image
  - Undo and redo changes step by step (or reset to the original data)
  - Batch-process every .gwy file in a folder by replaying the
    current processing pipeline on the selected channel
  - Save the result as an image or back into a .gwy file, next to
    every other channel of the measurement

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
import gwy_destripe as gd
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


def _mdsr_kwargs(params):
    """MDSR parameters as gwy_destripe keywords ('directions' comes from a
    combobox, so it arrives as a string)."""
    return dict(
        angle=float(params.get("angle", 0.0)),
        directions=int(params.get("directions", gd.DEFAULTS["directions"])),
        levels=int(params.get("levels", gd.DEFAULTS["levels"])),
        sigma=float(params.get("sigma", gd.DEFAULTS["sigma"])),
        sigma_a=float(params.get("sigma_a", gd.DEFAULTS["sigma_a"])),
        max_angle=float(params.get("max_angle", gd.DEFAULTS["max_angle"])),
    )


def _gsr_kwargs(params):
    """GSR parameters as gwy_destripe keywords."""
    return dict(
        angle=float(params.get("angle", 0.0)),
        mu1=float(params.get("mu1", gd.GSR_DEFAULTS["mu1"])),
        mu2=float(params.get("mu2", gd.GSR_DEFAULTS["mu2"])),
        iterations=int(params.get("iterations", gd.GSR_DEFAULTS["iterations"])),
    )


def _op_destripe(data, params, dx, dy):
    """Stripe removal by either of the two methods; `method` selects."""
    if str(params.get("method", "MDSR")).upper() == "GSR":
        return gd.gsr(data, **_gsr_kwargs(params))
    return gd.mdsr(data, pad=bool(params.get("pad", False)),
                   **_mdsr_kwargs(params))


def _fft_auto_items(data, params, dx, dy):
    """Auto-detect spectral noise on `data` against its local radial
    background (gp.detect_fft_noise): streak columns/rows and extended
    patches become rectangles, sharp peaks circular notches.
    Returns (notches, rects)."""
    notches, rects = gp.detect_fft_noise(
        data, dx=dx, dy=dy,
        protect_radius=params.get("protect_radius", 3.0),
        peak_db=params.get("threshold_db", 12.0),
        max_items=50,
    )
    return [list(n) for n in notches], [list(r) for r in rects]


def _op_fft(data, params, dx, dy):
    """Unified FFT filter: one frequency mask combining an optional radial
    lowpass/highpass with circular notches, rectangles and straight
    bands, optionally with smoothed (soft) edges."""
    radius = params.get("radius", 0.5)
    notches = [list(n) for n in params.get("notches", [])]
    rects = [list(r) for r in params.get("rects", [])]
    if params.get("auto"):
        # Re-detect on THIS image (batch-friendly: every image gets its
        # own detection instead of replaying fixed frequencies)
        a_notches, a_rects = _fft_auto_items(data, params, dx, dy)
        notches += a_notches
        rects += a_rects

    mask = np.ones(data.shape, dtype=bool)
    mode = params.get("mode", "none")
    if mode in ("lowpass", "highpass"):
        mask &= gp.build_pass_mask(
            data.shape, dx=dx, dy=dy, mode=mode, cutoff=params["cutoff"]
        )
    if notches:
        mask &= gp.build_notch_mask(
            data.shape, dx=dx, dy=dy, notches=notches, radius=radius
        )
    if rects:
        mask &= gp.build_rect_mask(data.shape, dx=dx, dy=dy, rects=rects)
    x_bands = params.get("x_bands", [])
    y_bands = params.get("y_bands", [])
    if x_bands or y_bands:
        mask &= gp.build_band_mask(
            data.shape, dx=dx, dy=dy,
            x_bands=x_bands, y_bands=y_bands,
            half_width=radius,
        )
    mask = gp.smooth_fft_mask(mask, dx=dx, dy=dy,
                              width=params.get("smooth", 0.0))
    # the DC bin always survives, whatever was drawn over it: no filter
    # here may shift the mean height of the image
    ny, nx = data.shape
    mask[ny // 2, nx // 2] = True
    return gp.filter_by_2d_fft_mask(data, mask)


def twoway_kwargs(params, detect=False):
    """Translate a dialog's flat parameter dict into gwy_twoway keywords.
    Tolerates missing keys (the merge and parachuting dialogs expose
    different subsets), falling back to the gwy_twoway defaults."""
    g = params.get
    flip = {"auto": "auto", "yes": True, "no": False}[g("flip_backward", "auto")]
    manual = g("slope_mode", "manual") == "manual"
    return dict(
        pre_plane=bool(g("pre_plane", False)),
        pre_rows=bool(g("pre_rows", False)),
        pre_rows_order=int(g("pre_rows_order", 2)),
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
        stripe_thresh=float(g("stripe_thresh", 3.0)),
        stripe_min_len=int(g("stripe_min_len", 3)),
        stripe_pref=float(g("stripe_pref", 1.0)),
    )


def twoway_param_relevant(name, p):
    """Whether a two-way / parachuting dialog parameter has any effect under
    the currently selected dropdown choices. Used to hide the irrelevant
    parameter rows; unknown names are always relevant."""
    g = p.get
    mapping = g("mapping", "xcorr")
    combine = g("combine", "average")
    corr = combine == "correlation"
    stripes = combine == "stripes"
    corr_combine = (g("corr_combine", "average")
                    if (corr or stripes) else None)
    measured = mapping in ("xcorr", "model_scaled", "measured")
    rules = {
        # preprocessing
        "pre_rows_order": bool(g("pre_rows", False)),
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
        "corr_combine": corr or stripes,
        "stripe_thresh": stripes,
        "stripe_min_len": stripes,
        "stripe_pref": stripes,
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
    if combine in ("correlation", "stripes"):
        shared = params.get("corr_combine", "average")
        if shared == "softmin":
            shared += f" beta={params.get('beta', 0.0)}"
        if combine == "correlation":
            return (f"correlation margin={params.get('corr_margin', 0.7)}, "
                    f"win={params.get('corr_window', 11)}px, "
                    f"referee={params.get('corr_aux', 'phase+error')}, "
                    f"shared={shared}")
        return (f"stripes thr={params.get('stripe_thresh', 3.0)}sigma, "
                f"minlen={params.get('stripe_min_len', 3)}px, "
                f"pref={params.get('stripe_pref', 1.0)}, shared={shared}")
    return combine


def _describe_pre_level(params):
    parts = []
    if params.get("pre_plane"):
        parts.append("plane")
    if params.get("pre_rows"):
        parts.append(f"rows p{params.get('pre_rows_order', 2)}")
    return "pre-level " + "+".join(parts) if parts else None


def _describe_two_way(params):
    parts = [f"map={params['mapping']}"]
    pre = _describe_pre_level(params)
    if pre:
        parts.insert(0, pre)
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
    pre = _describe_pre_level(params)
    if pre:
        parts.insert(0, pre)
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


def _validate_destripe(params):
    if str(params.get("method", "MDSR")).upper() == "GSR":
        if params["mu1"] <= 0 or params["mu2"] <= 0:
            return "mu1 and mu2 must be positive"
        if params["iterations"] < 1:
            return "There must be at least one iteration"
        return None
    if params["sigma"] <= 0:
        return "Damping width must be positive"
    if params["sigma_a"] <= 0:
        return "Angular falloff must be positive"
    if params["levels"] < 1:
        return "There must be at least one scale"
    if int(params.get("directions", 8)) & (int(params.get("directions", 8)) - 1):
        return "Directions must be a power of two"
    if params["max_angle"] < 0:
        return "Max direction cannot be negative"
    return None


def _describe_destripe(params):
    angle = f"{params.get('angle', 0.0):g} deg"
    if str(params.get("method", "MDSR")).upper() == "GSR":
        return (f"GSR, {angle}, mu1={params.get('mu1', 0.0):.4g}, "
                f"mu2={params.get('mu2', 0.0):.4g}, "
                f"{params.get('iterations', 0)} iterations")
    return (f"MDSR, {angle}, sigma={params.get('sigma', 0.0):g}, "
            f"{params.get('directions', 8)} dirs, "
            f"{params.get('levels', 5)} scales"
            + (", mirrored edges" if params.get("pad") else ""))


def _validate_fft(params):
    if params.get("mode", "none") in ("lowpass", "highpass") and params["cutoff"] <= 0:
        return "Cutoff frequency must be positive"
    if params["radius"] <= 0:
        return "Notch radius must be positive"
    if params["protect_radius"] < 0:
        return "Protect radius cannot be negative"
    if params.get("smooth", 0.0) < 0:
        return "Edge smoothing cannot be negative"
    return None


def _describe_fft(params):
    parts = []
    mode = params.get("mode", "none")
    if mode in ("lowpass", "highpass"):
        parts.append(f"{mode}@{params['cutoff']}")
    n_notch = len(params.get("notches", []))
    if n_notch:
        parts.append(f"{n_notch} notches")
    if params.get("rects"):
        parts.append(f"{len(params['rects'])} rects")
    if params.get("auto"):
        parts.append(f"auto-detect@{params.get('threshold_db')}dB")
    if params.get("x_bands"):
        parts.append(f"{len(params['x_bands'])} v-bands")
    if params.get("y_bands"):
        parts.append(f"{len(params['y_bands'])} h-bands")
    if n_notch or params.get("auto") or params.get("x_bands") or params.get("y_bands"):
        parts.append(f"radius={params['radius']}")
    if params.get("smooth", 0.0) > 0:
        parts.append(f"smooth={params['smooth']}")
    return ", ".join(parts) if parts else "no-op"


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
        "label": "Poly background",   # short: shares a button row with Plane level
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
    "destripe": {
        "label": "Stripe removal",
        "func": _op_destripe,
        # `method` picks which of the two algorithms runs; the dialog shows
        # only the parameters that belong to the selected one.
        "params": [
            {"name": "method", "label": "Method", "type": "choice",
             "default": "MDSR", "values": ["MDSR", "GSR"]},
            {"name": "angle", "label": "Stripe angle (deg)", "type": "float",
             "default": 0.0, "min": -180.0, "max": 180.0},
            # --- MDSR (Fourier filtering in the contourlet domain)
            {"name": "sigma", "label": "Damping width (bins)", "type": "float",
             "default": gd.DEFAULTS["sigma"], "min": 0.0, "max": 1e4},
            {"name": "directions", "label": "Directions", "type": "choice",
             "default": "8", "values": ["4", "8", "16", "32"]},
            {"name": "levels", "label": "Scales", "type": "int",
             "default": gd.DEFAULTS["levels"], "min": 1, "max": 10},
            {"name": "sigma_a", "label": "Angular falloff (rad)", "type": "float",
             "default": gd.DEFAULTS["sigma_a"], "min": 0.0, "max": 10.0},
            {"name": "max_angle", "label": "Max direction (deg)", "type": "float",
             "default": gd.DEFAULTS["max_angle"], "min": 0.0, "max": 90.0},
            {"name": "pad", "label": "Mirror edges", "type": "bool",
             "default": False},
            # --- GSR (variational)
            {"name": "mu1", "label": "mu1 (removal)", "type": "float",
             "default": gd.GSR_DEFAULTS["mu1"], "min": 0.0, "max": 100.0},
            {"name": "mu2", "label": "mu2 (retention)", "type": "float",
             "default": gd.GSR_DEFAULTS["mu2"], "min": 0.0, "max": 100.0},
            {"name": "iterations", "label": "Iterations", "type": "int",
             "default": gd.GSR_DEFAULTS["iterations"], "min": 1, "max": 100000},
        ],
        "removed_label": "Removed stripes",
        "validate": _validate_destripe,
        "describe": _describe_destripe,
    },
    "fft_filter": {
        "label": "FFT filter",
        "func": _op_fft,
        "params": [
            {"name": "mode", "label": "Pass filter", "type": "choice",
             "default": "none", "values": ["none", "lowpass", "highpass"]},
            {"name": "cutoff", "label": "Cutoff (1/spatial unit)", "type": "float",
             "default": 10.0, "min": 0.0, "max": 1e9},
            {"name": "radius", "label": "Notch radius", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1e9},
            {"name": "threshold_db", "label": "Detect threshold (dB)", "type": "float",
             "default": 12.0, "min": 0.0, "max": 200.0},
            {"name": "protect_radius", "label": "Protect center radius", "type": "float",
             "default": 3.0, "min": 0.0, "max": 1e9},
            {"name": "smooth", "label": "Edge smoothing (freq)", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1e9},
            {"name": "auto", "label": "Auto re-detect (per image)", "type": "bool",
             "default": False},
        ],
        "removed_label": "Removed component (noise)",
        "validate": _validate_fft,
        "describe": _describe_fft,
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
    # Two-way (forward/backward) operations: they need a channel *pair*
    # rather than the single current image, so they get their own dialogs,
    # and `needs_pair` tells apply_pipeline to hand them the forward/backward
    # context.
    "two_way": {
        "label": "Two-way merge (Fwd/Bwd)",
        "func": _op_two_way,
        "needs_pair": True,
        "channel_suffix": "[Merged]",
        "params": [
            # -- background correction of both scans (real preprocessing)
            {"name": "pre_plane", "label": "Plane removal", "type": "bool",
             "default": False},
            {"name": "pre_rows", "label": "Row align (poly)", "type": "bool",
             "default": False},
            {"name": "pre_rows_order", "label": "Row poly order", "type": "int",
             "default": 2, "min": 0, "max": 10},
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
             "values": ["average", "correlation", "stripes", "slope",
                        "consensus", "softmin", "min", "max", "forward",
                        "backward"]},
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
            # -- combine='stripes' only
            {"name": "stripe_thresh", "label": "Stripe threshold (sigma)",
             "type": "float", "default": 3.0, "min": 0.1, "max": 100.0},
            {"name": "stripe_min_len", "label": "Stripe min length (px)",
             "type": "int", "default": 3, "min": 1, "max": 512},
            {"name": "stripe_pref", "label": "Clean-scan weight (0.5-1)",
             "type": "float", "default": 1.0, "min": 0.5, "max": 1.0},
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
            # -- background correction of both scans (real preprocessing)
            {"name": "pre_plane", "label": "Plane removal", "type": "bool",
             "default": False},
            {"name": "pre_rows", "label": "Row align (poly)", "type": "bool",
             "default": False},
            {"name": "pre_rows_order", "label": "Row poly order", "type": "int",
             "default": 2, "min": 0, "max": 10},
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

# Layout of the operation buttons in the main window: one entry per row,
# several keys in a row put the buttons side by side. The order follows the
# way the steps are normally used on a scan, not the order OPERATIONS
# happens to declare them. "@fft_spectrum" is the view-only spectrum window,
# not an operation.
OPERATION_ROWS = [
    ("two_way",),
    ("parachute",),
    ("plane_level", "polynomial"),
    ("align_rows",),
    ("fft_filter",),
    ("destripe",),
    ("remove_scars",),
    ("@fft_spectrum",),
    ("zero_baseline",),
    ("crop",),
    ("percentile",),
]

# Flat list of the single-image operations, in button order
OPERATION_ORDER = [
    key for row in OPERATION_ROWS for key in row
    if not key.startswith("@") and not OPERATIONS[key].get("needs_pair")
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


def _gwy_channel_titles(container):
    """The titles of the channels already present in a .gwy container."""
    titles = []
    for k in container.keys():
        parts = k.split("/")
        if len(parts) == 4 and parts[1].isdigit() and parts[2:] == ["data",
                                                                    "title"]:
            titles.append(container[k])
    return titles


def save_channel_to_gwy(path, title, data, xreal=None, yreal=None,
                        unit_xy="", unit_z="", extra_channels=()):
    """
    Save `data` (in SI units) as a channel of a Gwyddion .gwy file.

    If the file already exists, the channel is APPENDED with the next free
    channel number, so repeated saves collect all processed channels in
    one .gwy file.

    `extra_channels` is a sequence of (title, GwyDataField) written next to
    it - typically the untouched channels of the source measurement, so the
    saved file stands on its own. A channel whose title is already in the
    file is skipped, so saving repeatedly never duplicates them.

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
    have.add(unique)

    written = []
    for extra_title, extra_field in extra_channels:
        if extra_title in have:
            continue
        n += 1
        container[f"/{n}/data"] = extra_field
        container[f"/{n}/data/title"] = extra_title
        have.add(extra_title)
        written.append(extra_title)

    container.tofile(path)
    return n - len(written), unique, written


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


class DestripeSweepWindow(tk.Toplevel):
    """
    A grid of stripe-removal results over two chosen parameters, for finding
    the setting that takes the stripes out without eating the structures.

    Each axis sweeps one parameter of the current method around the value
    set in the dialog. Gains are stepped by a factor (geometric: with factor
    2 and a 3x3 grid the rows are mu1/2, mu1, 2*mu1), counts and angles by an
    increment - 'Same rate' keeps both axes on the same step, which for GSR's
    mu1/mu2 makes the diagonal the "scale both together" direction the paper
    describes.

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
    }
    DEFAULT_AXES = {"GSR": ("mu1", "mu2"), "MDSR": ("sigma", "levels")}

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
            self.link_var.set(True)     # _sync_link drops it if it cannot hold
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
        self._sync_link()

    def _modes_match(self):
        row = getattr(self, "_names", {}).get(self.row_var.get())
        col = getattr(self, "_names", {}).get(self.col_var.get())
        if row is None or col is None:
            return False
        return self.SWEEP_STEPS[row][0] == self.SWEEP_STEPS[col][0]

    def _sync_link(self):
        """'Same rate' only means something when both axes step the same
        way; otherwise the column keeps its own step."""
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
                      cmap=gp.get_gwyddion_cmap(), extent=self._extent(),
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
            label = ttk.Label(frame, text=p["label"] + ":")
            label.pack(side=tk.LEFT, padx=(8, 2))
            self.param_labels[p["name"]] = label
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
        self._status_label = ttk.Label(frame, textvariable=self.status_var,
                                       foreground="red")
        self._status_label.pack(side=tk.RIGHT, padx=8)

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

    def _toolbar_busy(self):
        toolbar = getattr(self.canvas, "toolbar", None)
        return toolbar is not None and getattr(toolbar, "mode", "")

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
            result, origin="upper", cmap=gp.get_gwyddion_cmap(),
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
            self.hint_var.set(
                "Stripe angle: 0 = horizontal scan lines, 90 = vertical  |  "
                "Drag on the result panel to pick the zoom area"
                + ("  |  GSR: more iterations = better converged, slower"
                   if method == "GSR" else "")
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

        im0 = ax0.imshow(result, origin="upper", cmap=gp.get_gwyddion_cmap(),
                         extent=extent, aspect="equal")
        ax0.set_title(f"Preview: {self._method()} result  "
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

        if self._method() == "GSR":
            self._draw_input_panel(ax2, data, extent)
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
        im = ax.imshow(data, origin="upper", cmap=gp.get_gwyddion_cmap(),
                       extent=extent, aspect="equal")
        ax.set_title("Input (before)")
        ax.set_xlabel(f"x ({app.spatial_units})")
        self.figure.colorbar(im, ax=ax, fraction=0.046).set_label(app.z_units)

    def _draw_mask_panel(self, ax, data, params):
        """The composite MDSR frequency mask, on the same physical frequency
        axes as the FFT filter dialog."""
        app = self.app
        ny, nx = data.shape
        mask = gd.mdsr_mask(data.shape, **_mdsr_kwargs(params))
        freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=app.dx))
        freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=app.dy))
        hx, hy = 0.5 / (nx * app.dx), 0.5 / (ny * app.dy)
        freq_extent = [freq_x[0] - hx, freq_x[-1] + hx,
                       freq_y[-1] + hy, freq_y[0] - hy]
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
                       cmap=gp.get_gwyddion_cmap(), extent=extent,
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
