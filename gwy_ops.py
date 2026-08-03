"""
What the processing steps are, and what each one takes.

Every step that can be applied to an image is declared once here, in
OPERATIONS: the function that does it, the parameters it takes and the range
each one is allowed, how to check a set of values before running, and how to
say in one line what was done. A dialog builds its widgets from those
declarations and the batch runner replays them with `apply_pipeline`, so an
operation is added by writing an entry, not by writing a window.

The work itself is done elsewhere - gwy_processing, gwy_flatten,
gwy_destripe, gwy_twoway. What this module adds is the joinery: translating a
dialog's flat dictionary of values into each module's keywords, the checks
that have to happen before the call rather than inside it, and the sentence
that goes into the log afterwards.

Reading a channel is here for the same reason. `channel_view` turns a stored
GwyDataField into the array, the physical extents and the unit labels that
every step is handed, and the main window and the quick view have to do that
identically or the same file looks like two different files.

Nothing here draws anything or knows what is drawing it. No GUI toolkit is
imported and no window is ever mentioned, so a script that only wants to
replay a pipeline can import this on its own. That is also what made the
front end replaceable: it was rewritten from Tkinter to Qt without a line of
this file changing.
"""

import re

import numpy as np

import gwy_processing as gp
import gwy_flatten as gf
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


def _smart_kwargs(params):
    """Dialog values as gwy_flatten keywords."""
    return dict(
        detect=str(params.get("detect", gf.DEFAULTS["detect"])),
        threshold=str(params.get("threshold", gf.DEFAULTS["threshold"])),
        feature_size=float(params.get("feature_size",
                                      gf.DEFAULTS["feature_size"])),
        neighbourhood=float(params.get("neighbourhood",
                                       gf.DEFAULTS["neighbourhood"])),
        sensitivity=float(params.get("sensitivity",
                                     gf.DEFAULTS["sensitivity"])),
        expand=int(params.get("expand", gf.DEFAULTS["expand"])),
        edge=float(params.get("edge", gf.DEFAULTS["edge"])),
        grow=int(params.get("grow", gf.DEFAULTS["grow"])),
        min_area=float(params.get("min_area", gf.DEFAULTS["min_area"])),
        fit=str(params.get("fit", gf.DEFAULTS["fit"])),
        order=int(params.get("order", gf.DEFAULTS["order"])),
        window=int(params.get("window", gf.DEFAULTS["window"])),
        passes=int(params.get("passes", gf.DEFAULTS["passes"])),
    )


def _exclusion_mask(shape, rects, dx, dy):
    """Rectangles dragged on the preview, as a boolean mask.

    They are carried in the parameters - and so into the pipeline, the log
    and any replay - in physical units, like the FFT dialog's notches, so
    they still mean the same part of the sample if the pixel size changes.
    Images are drawn with origin='upper', so y counts up from the bottom of
    the extent while row 0 is at the top."""
    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    for rect in rects or ():
        x0, x1, y0, y1 = (float(v) for v in rect)
        ix0 = max(0, int(np.floor(min(x0, x1) / dx)))
        ix1 = min(nx, int(np.ceil(max(x0, x1) / dx)))
        iy0 = max(0, int(np.floor(ny - max(y0, y1) / dy)))
        iy1 = min(ny, int(np.ceil(ny - min(y0, y1) / dy)))
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
    return mask


def _smart_flatten(data, params, dx, dy):
    """The whole `gwy_flatten` result, so the dialog can show the mask and
    the direction `auto` settled on; the operation itself keeps the image."""
    exclude = _exclusion_mask(data.shape, params.get("exclude"), dx, dy)
    return gf.flatten(data, exclude=exclude if exclude.any() else None,
                      **_smart_kwargs(params))


def _op_smart_level(data, params, dx, dy):
    return _smart_flatten(data, params, dx, dy)["data"]


def _validate_smart(params):
    if params.get("threshold") == "otsu" and params.get("detect") == "both":
        return ("Otsu splits the image in two, so 'both' would mask all of "
                "it. Choose convex or concave, or use the adaptive threshold.")
    if not 0.0 < params.get("feature_size", 1.0) <= 100.0:
        return "The feature size is a percentage of the image, above 0"
    if not 0.0 < params["neighbourhood"] <= 100.0:
        return "The neighbourhood is a percentage of the image, above 0"
    if params["sensitivity"] <= 0:
        return "The threshold offset must be positive"
    if params["edge"] < 0:
        return "The edge gate cannot be negative"
    if params["min_area"] < 0:
        return "The smallest feature cannot be negative"
    if params["window"] and params["window"] < 3:
        return "A sliding window is at least 3 px wide (0 fits in one piece)"
    if params["passes"] < 1:
        return "There must be at least one pass"
    return None


def _describe_smart(params):
    where = {"rows": "along rows", "columns": "down columns",
             "both": "along rows then down columns",
             "surface": "as a surface",
             "auto": "along whichever way the scan lines run"}.get(
                 params.get("fit", "rows"), "?")
    window = params.get("window", 0)
    drawn = len(params.get("exclude") or ())
    return (f"{params.get('threshold', '?')}/{params.get('detect', '?')} mask, "
            f"order {params.get('order', 0)} {where}"
            + (f", {window}px window" if window else "")
            + f", {params.get('passes', 1)} passes"
            + (f", {drawn} area{'s' if drawn > 1 else ''} excluded by hand"
               if drawn else ""))


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


def _chen_kwargs(params):
    """DeStripe parameters as gwy_destripe keywords."""
    return dict(
        window=int(params.get("window", gd.CHEN_DEFAULTS["window"])),
        cvar_k=float(params.get("cvar_k", gd.CHEN_DEFAULTS["cvar_k"])),
        density=float(params.get("density", gd.CHEN_DEFAULTS["density"])),
        min_run=int(params.get("min_run", gd.CHEN_DEFAULTS["min_run"])),
        keep_mean=bool(params.get("keep_mean",
                                  gd.CHEN_DEFAULTS["keep_mean"])),
    )


def _op_destripe(data, params, dx, dy):
    """Stripe removal by any of the three methods; `method` selects."""
    method = str(params.get("method", "MDSR")).upper()
    if method == "GSR":
        return gd.gsr(data, **_gsr_kwargs(params))
    if method == "DESTRIPE":
        return gd.destripe_chen(data, **_chen_kwargs(params))
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
    method = str(params.get("method", "MDSR")).upper()
    if method == "GSR":
        if params["mu1"] <= 0 or params["mu2"] <= 0:
            return "mu1 and mu2 must be positive"
        if params["iterations"] < 1:
            return "There must be at least one iteration"
        return None
    if method == "DESTRIPE":
        if params["window"] < 1:
            return "The neighbourhood must be at least 1 pixel wide"
        if params["cvar_k"] < 0:
            return "The CVAR threshold cannot be negative"
        if not 0.0 < params["density"] <= 1.0:
            return "The central density must be between 0 and 1"
        if params["min_run"] < 1:
            return "A line must be at least one pixel long"
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
    method = str(params.get("method", "MDSR")).upper()
    if method == "GSR":
        return (f"GSR, {angle}, mu1={params.get('mu1', 0.0):.4g}, "
                f"mu2={params.get('mu2', 0.0):.4g}, "
                f"{params.get('iterations', 0)} iterations")
    if method == "DESTRIPE":
        return (f"DeStripe, k={params.get('cvar_k', 0.0):g}, "
                f"window {2 * int(params.get('window', 1)) + 1}px, "
                f"density {params.get('density', 0.0):g}, "
                f"run {params.get('min_run', 0)}"
                + ("" if params.get("keep_mean", True) else ", mean filtered"))
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
    "smart_level": {
        "label": "Smart background",
        "func": _op_smart_level,
        # Declared grouped, and in the order the dialog shows them:
        # threshold, then refine the outline, then fit what is left.
        "params": [
            {"name": "detect", "label": "Features", "type": "choice",
             "default": gf.DEFAULTS["detect"], "values": list(gf.DETECT)},
            {"name": "threshold", "label": "Find by", "type": "choice",
             "default": gf.DEFAULTS["threshold"], "values": list(gf.THRESHOLDS)},
            {"name": "feature_size", "label": "Feature size (%)",
             "type": "float", "default": gf.DEFAULTS["feature_size"],
             "min": 0.0, "max": 100.0},
            {"name": "neighbourhood", "label": "Neighbourhood (%)",
             "type": "float", "default": gf.DEFAULTS["neighbourhood"],
             "min": 0.0, "max": 100.0},
            {"name": "sensitivity", "label": "Threshold (sigma)",
             "type": "float", "default": gf.DEFAULTS["sensitivity"],
             "min": 0.0, "max": 100.0},
            {"name": "min_area", "label": "Smallest (% of frame)",
             "type": "float", "default": gf.DEFAULTS["min_area"],
             "min": 0.0, "max": 100.0},
            {"name": "expand", "label": "Expand (px)", "type": "int",
             "default": gf.DEFAULTS["expand"], "min": 0, "max": 500},
            {"name": "edge", "label": "Edge gate (sigma)", "type": "float",
             "default": gf.DEFAULTS["edge"], "min": 0.0, "max": 100.0},
            {"name": "grow", "label": "Margin (px)", "type": "int",
             "default": gf.DEFAULTS["grow"], "min": 0, "max": 100},
            {"name": "fit", "label": "Fit", "type": "choice",
             "default": gf.DEFAULTS["fit"], "values": list(gf.FITS)},
            {"name": "order", "label": "Order", "type": "int",
             "default": gf.DEFAULTS["order"], "min": 0, "max": 6},
            {"name": "window", "label": "Sliding window (px, 0 = whole)",
             "type": "int", "default": gf.DEFAULTS["window"],
             "min": 0, "max": 4096},
            {"name": "passes", "label": "Passes", "type": "int",
             "default": gf.DEFAULTS["passes"], "min": 1, "max": 5},
        ],
        "removed_label": "Removed background",
        "validate": _validate_smart,
        "describe": _describe_smart,
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
             "default": "MDSR", "values": ["MDSR", "GSR", "DeStripe"]},
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
            # --- DeStripe (noisy pixels of the log spectrum)
            {"name": "cvar_k", "label": "CVAR threshold (sigma)",
             "type": "float", "default": gd.CHEN_DEFAULTS["cvar_k"],
             "min": 0.0, "max": 100.0},
            {"name": "window", "label": "Neighbourhood NS", "type": "int",
             "default": gd.CHEN_DEFAULTS["window"], "min": 1, "max": 10},
            {"name": "density", "label": "Central density", "type": "float",
             "default": gd.CHEN_DEFAULTS["density"], "min": 0.01, "max": 1.0},
            {"name": "min_run", "label": "Line length (px)", "type": "int",
             "default": gd.CHEN_DEFAULTS["min_run"], "min": 1, "max": 100},
            {"name": "keep_mean", "label": "Keep mean height", "type": "bool",
             "default": gd.CHEN_DEFAULTS["keep_mean"]},
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
    ("smart_level",),
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
# Reading a channel
# ---------------------------------------------------------------------------

def channel_view(field):
    """Everything needed to display one channel: the data converted to
    display units, the physical size in those units, and the unit labels.

    Shared by the main window and the quick view so both read a channel the
    same way."""
    xy_unit = _unit_of(field, "si_unit_xy")
    z_unit = _unit_of(field, "si_unit_z")
    xy_factor, spatial_units = spatial_scale(xy_unit)
    z_factor, z_units = z_scale(z_unit)
    data = field.data.astype(np.float64) * z_factor
    ny, nx = data.shape
    return {
        "data": data,
        "x_real": (field.xreal or nx) * xy_factor,
        "y_real": (field.yreal or ny) * xy_factor,
        "spatial_units": spatial_units,
        "z_units": z_units,
        "xy_factor": xy_factor,
        "z_factor": z_factor,
        "unit_xy_str": xy_unit,
        "unit_z_str": z_unit,
    }


def _natural_key(name):
    """Sort key that orders file_2 before file_10."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def pick_channel(names, wanted):
    """Which of `names` to show, given the channel the user last chose.

    Keep `wanted` if this file has it. If it does not - the first file of a
    folder, or one that names its channels differently - keep the choice
    loosely by matching the first word, and fall back to a height channel,
    then to whatever comes first.
    """
    if wanted in names:
        return wanted
    match = None
    if wanted:
        prefix = wanted.split(" ")[0]
        match = next((n for n in names if n.startswith(prefix)), None)
    return match or next((n for n in names if "Height" in n), names[0])
