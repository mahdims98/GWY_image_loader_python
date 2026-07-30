"""
gwy_twoway.py
=============

Two-way (forward/backward) scan processing for AFM images:

  1. Hysteresis alignment  - map the backward image onto the forward column
     grid, using the power-law distortion model fitted by
     ``hysteresis_compensation.py`` (or a simpler quadratic / measured mapping).

  2. Parachuting detection - flag pixels where the tip was airborne because the
     surface dropped away faster than the tip could follow, using the
     H(delta, dz) height-difference histogram of Kubo et al.

  3. Soft-min merge        - combine the two scans, discarding the flagged
     pixels in favour of the opposite scan and soft-min averaging elsewhere.

Method reference
----------------
S. Kubo, K. Umeda, N. Kodera, S. Takada,
"Removing the parachuting artifact using two-way scanning data in high-speed
atomic force microscopy", Biophysics and Physicobiology 20, e200006 (2023).

The hysteresis stage differs from the paper: instead of the paper's quadratic
fit to a cosine-similarity column match, it re-uses the existing power-law
distortion model (``Hysteresis compensation Python/hysteresis_compensation.py``),
which is fitted to a single forward/backward pair. The paper's quadratic
regularization is available as ``mapping="quadratic"``.

All heights are handled in whatever unit the caller supplies (the GUI works in
nm), so height-dependent parameters (``slope``, ``offset``, ``beta``) are in
that same unit. Lateral quantities are in pixels.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter

# --------------------------------------------------------------------------- #
#  Locate the hysteresis-compensation module
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))

#: Directories searched for ``hysteresis_compensation.py``, in order. The
#: environment variable GWY_HYSTERESIS_PATH takes precedence over all of them.
HYSTERESIS_SEARCH_PATHS = [
    os.environ.get("GWY_HYSTERESIS_PATH", ""),
    _HERE,
    os.path.join(_HERE, "hysteresis"),
    os.path.abspath(os.path.join(_HERE, "..", "Hysteresis compensation Python")),
    r"D:\Software\Mahdi\Hysteresis compensation Python",
]

_hc_module = None


def load_hysteresis_module():
    """Import ``hysteresis_compensation`` from wherever it lives on this
    machine. Cached after the first successful import.

    Raises ImportError with the searched locations if it cannot be found."""
    global _hc_module
    if _hc_module is not None:
        return _hc_module
    try:
        import hysteresis_compensation as hc  # already importable
        _hc_module = hc
        return hc
    except ImportError:
        pass
    tried = []
    for folder in HYSTERESIS_SEARCH_PATHS:
        if not folder:
            continue
        path = os.path.join(folder, "hysteresis_compensation.py")
        tried.append(path)
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location("hysteresis_compensation", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["hysteresis_compensation"] = module
        spec.loader.exec_module(module)
        _hc_module = module
        return module
    raise ImportError(
        "hysteresis_compensation.py not found. Looked in:\n  "
        + "\n  ".join(tried)
        + "\nSet the GWY_HYSTERESIS_PATH environment variable to its folder."
    )


# --------------------------------------------------------------------------- #
#  Column resampling helpers
# --------------------------------------------------------------------------- #
def resample_columns(image, columns, kind="spline"):
    """Sample every row of `image` at the (fractional) column positions
    `columns`. Positions are clamped to the valid range, so the edges are
    held constant rather than extrapolated."""
    image = np.asarray(image, dtype=float)
    nx = image.shape[1]
    q = np.clip(np.asarray(columns, dtype=float), 0.0, nx - 1.0)
    if kind == "linear":
        x_old = np.arange(nx)
        return np.stack([np.interp(q, x_old, row) for row in image])
    cs = CubicSpline(np.arange(nx), image, axis=1)
    return cs(q)


def resample_mask(mask, columns):
    """Nearest-neighbour resampling for boolean masks (keeps them boolean)."""
    mask = np.asarray(mask, dtype=bool)
    nx = mask.shape[1]
    idx = np.clip(np.rint(np.asarray(columns, dtype=float)), 0, nx - 1).astype(int)
    return mask[:, idx]


def _monotone(y):
    """Force a sequence to be non-decreasing (running maximum)."""
    return np.maximum.accumulate(np.asarray(y, dtype=float))


def _invert_curve(u_src, u_dst, query):
    """Invert a monotone increasing mapping u_dst = M(u_src), evaluated at
    `query` in the destination coordinate."""
    return np.interp(query, _monotone(u_dst), u_src)


# --------------------------------------------------------------------------- #
#  Stage 1: hysteresis alignment
# --------------------------------------------------------------------------- #
@dataclass
class Alignment:
    """Result of matching the backward scan onto the forward column grid."""
    fwd: np.ndarray                  # forward image, possibly resampled
    bwd: np.ndarray                  # backward image on the same column grid
    columns_fwd: np.ndarray          # source columns sampled from the forward image
    columns_bwd: np.ndarray          # source columns sampled from the backward image
    shift_px: np.ndarray             # fitted forward->backward shift, per column
    mapping: str = "xcorr"           # which mapping was used
    flipped_backward: bool = False   # whether the backward image was mirrored
    crop_cols: tuple = None          # (c0, c1): columns where BOTH scans have
                                     # real data; outside, one scan never
                                     # imaged the area and the edge was held
    max_shift_px: float = 0.0
    lag_px: float = 0.0              # mean shift (the constant feedback lag)
    bow_px: float = 0.0              # peak-to-peak of the non-constant part
    corr_before: float = float("nan")   # fwd/bwd similarity before alignment
    corr_after: float = float("nan")    # ... and after
    rms_before: float = float("nan")    # fwd/bwd rms difference before alignment
    rms_after: float = float("nan")     # ... and after
    measured_centers: np.ndarray = field(repr=False, default=None)
    measured_shift_px: np.ndarray = field(repr=False, default=None)
    measured_quality: np.ndarray = field(repr=False, default=None)
    fit_rms: float = float("nan")
    hysteresis_result: object = field(repr=False, default=None)


def _similarity(a, b):
    """Zero-mean normalized correlation between two images."""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def _rms_difference(a, b, edge=25):
    """RMS of (a - b) with each row's mean offset removed, ignoring `edge`
    columns on either side (where resampling holds the edge constant)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d - d.mean(axis=1, keepdims=True)
    if edge and d.shape[1] > 2 * edge:
        d = d[:, edge:-edge]
    return float(np.std(d))


def backward_needs_flip(fwd, bwd):
    """True if the stored backward channel is mirrored with respect to the
    forward one (i.e. saved in scan-time order rather than spatially aligned).

    Most controllers already un-flip the backward scan before saving, in which
    case this returns False."""
    return _similarity(fwd, bwd[:, ::-1]) > _similarity(fwd, bwd)


def fit_plane(image):
    """Least-squares plane z = a + b*x + c*y fitted to the image, returned as
    an array of the same shape."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    A = np.column_stack([np.ones(image.size), X.ravel(), Y.ravel()])
    coeffs, *_ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    return (A @ coeffs).reshape(image.shape)


def remove_plane(image):
    """Subtract the fitted plane from the image."""
    image = np.asarray(image, dtype=float)
    return image - fit_plane(image)


def fit_rows_poly(image, order=2):
    """Per-row polynomial background: fit a 1-D polynomial of degree ``order``
    to every row (the 'align rows: polynomial' operation) and return the
    fitted background, same shape as the image."""
    image = np.asarray(image, dtype=float)
    nx = image.shape[1]
    t = np.linspace(-1.0, 1.0, nx)
    order = max(0, int(order))
    # Vandermonde least squares for all rows at once
    A = np.vander(t, order + 1, increasing=True)          # (nx, order+1)
    coeffs, *_ = np.linalg.lstsq(A, image.T, rcond=None)  # (order+1, ny)
    return (A @ coeffs).T


def align_rows_poly(image, order=2):
    """Subtract the per-row polynomial background from the image."""
    image = np.asarray(image, dtype=float)
    return image - fit_rows_poly(image, order)


def _level_for_match(image, level, poly_order=2):
    """Preprocessing applied before column matching only (never to the output
    data): the raw heights are dominated by the sample tilt, which swamps the
    feature contrast the matching needs.

    ``level``: 'plane' (least-squares plane), 'poly_rows' (per-row polynomial
    of degree ``poly_order``, the align-rows-polynomial operation), or
    'none'."""
    if level == "plane":
        return remove_plane(image)
    if level == "poly_rows":
        return align_rows_poly(image, poly_order)
    if level in ("none", None, ""):
        return np.asarray(image, dtype=float)
    raise ValueError(f"unknown match level {level!r}")


def measure_shift_profile(fwd, bwd, n_blocks=16, max_lag=20,
                          match_level="plane", match_poly_order=2,
                          min_quality=0.2):
    """Measure the forward->backward column shift directly, by block-wise
    normalized cross-correlation.

    The fitted plane is removed from each image first (``match_level='plane'``,
    the default), so the match is driven by surface features rather than by
    the sample tilt. This leveling affects the matching only, never the data.

    Returns ``(centers, shift_px, quality)``: the column index at the centre of
    each block, the sub-pixel shift such that forward column ``j`` corresponds
    to backward column ``j + shift``, and the peak correlation of each block.
    Blocks whose peak correlation falls below ``min_quality`` (featureless
    regions) get ``quality = 0`` and should be ignored by the caller."""
    fwd = _level_for_match(fwd, match_level, match_poly_order)
    bwd = _level_for_match(bwd, match_level, match_poly_order)
    nx = fwd.shape[1]

    n_blocks = max(1, int(n_blocks))
    width = nx // n_blocks
    max_lag = int(max_lag)
    lags = np.arange(-max_lag, max_lag + 1)

    centers, shifts, quality = [], [], []
    for k in range(n_blocks):
        s0, s1 = k * width, (k + 1) * width
        a = fwd[:, s0:s1]
        a = (a - a.mean()).ravel()
        norm_a = np.sqrt(np.dot(a, a))
        scores = np.full(len(lags), -1.0)
        for i, lag in enumerate(lags):
            b0, b1 = s0 + lag, s1 + lag
            if b0 < 0 or b1 > nx:
                continue
            c = bwd[:, b0:b1]
            c = (c - c.mean()).ravel()
            denom = norm_a * np.sqrt(np.dot(c, c))
            if denom > 0:
                scores[i] = float(np.dot(a, c) / denom)
        i = int(np.argmax(scores))
        peak = float(scores[i])
        # parabolic sub-pixel refinement around the discrete peak
        sub = 0.0
        if 0 < i < len(lags) - 1:
            y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                sub = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
        centers.append(0.5 * (s0 + s1 - 1))
        shifts.append(lags[i] + sub)
        quality.append(peak if peak >= min_quality else 0.0)

    return np.array(centers), np.array(shifts), np.array(quality)


def _hysteresis_model_shape(fwd, bwd, l_points, n_var, maxiter, seed,
                            match_level="plane", match_poly_order=2,
                            hysteresis_result=None):
    """Run (or reuse) the power-law hysteresis fit and return
    ``(shift_shape_u, result)``: the model's forward->backward shift in
    normalized [0,1] coordinates, sampled on ``result.x_c``.

    The fitted plane is removed first (``match_level='plane'``): the original
    hysteresis pipeline matches raw columns, and on tilted samples the
    background dominates the correlation."""
    hc = load_hysteresis_module()
    res = hysteresis_result
    if res is None:
        res = hc.hysteresis_detect(
            _level_for_match(fwd, match_level, match_poly_order),
            _level_for_match(bwd, match_level, match_poly_order),
            l_points=int(l_points),
            flip_backward=False,          # orientation handled by the caller
            n_var=int(n_var),
            de_kwargs=dict(maxiter=int(maxiter), seed=seed),
        )
    # The fit compares the model composition g(f^-1(t)) against whichever of
    # mapping_tr / mapping_rt lies above the diagonal; reproduce that here.
    f_mono = _monotone(res.f_x)
    f_inv = hc._invert_monotone(res.x_c, f_mono, res.x_c)
    composition = _monotone(hc.model_g(f_inv, res.x))
    if res.fit_uses_tr:
        target = composition                       # forward -> backward
    else:
        target = _invert_curve(res.x_c, composition, res.x_c)   # invert it
    return target - res.x_c, res


def align_two_way(
    fwd,
    bwd,
    mapping="xcorr",
    flip_backward="auto",
    warp="bwd_to_fwd",
    poly_order=2,
    n_blocks=16,
    max_lag=20,
    match_level="plane",
    match_poly_order=2,
    min_quality=0.2,
    l_points=400,
    n_var=10,
    maxiter=40,
    seed=0,
    smooth_measured=2.0,
    max_shift_px=None,
    interp="spline",
    hysteresis_result=None,
):
    """Put the backward scan on the same column grid as the forward scan.

    Parameters
    ----------
    fwd, bwd : 2-D arrays
        Forward and backward height images (rows = slow axis).
    mapping : {'xcorr', 'model', 'model_scaled', 'measured', 'none'}
        How the forward->backward column shift is estimated and regularized.

        ``xcorr``        - measure the shift by block cross-correlation
                           (:func:`measure_shift_profile`) and fit a polynomial
                           of degree ``poly_order``. Degree 0 is a pure
                           constant lag, 2 adds the hysteresis bow. This is the
                           default because a real scanner usually shows a large
                           constant feedback lag plus a small bow, and a
                           constant term is what the pinned-endpoint hysteresis
                           model cannot represent.
        ``model``        - the power-law distortion model of
                           ``hysteresis_compensation``, used as fitted. It is
                           pinned to zero shift at both scan ends, so it
                           describes pure hysteresis with no lag.
        ``model_scaled`` - fit ``c0 + c1 * model_shape(t)`` to the measured
                           shift profile: the hysteresis model supplies the
                           bow *shape*, the data supply the lag ``c0`` and the
                           amplitude ``c1``.
        ``measured``     - the block-measured profile itself, interpolated and
                           lightly smoothed. No shape assumption, but noisier.
        ``none``         - identity; no correction.
    flip_backward : {'auto', True, False}
        Mirror the backward image before matching. ``auto`` decides by
        correlation (see :func:`backward_needs_flip`). Controllers that already
        un-flip the backward scan need no mirroring.
    warp : {'bwd_to_fwd', 'split', 'linearize_both'}
        ``bwd_to_fwd``     - keep the forward image untouched and resample the
                             backward image onto its grid (what the paper does).
        ``split``          - move each image half of the shift, so neither is
                             preferred and the output sits between the two.
        ``linearize_both`` - resample both images onto the undistorted physical
                             coordinate implied by the model. Requires
                             ``mapping='model'`` or ``'model_scaled'``.
    poly_order : int
        Degree of the polynomial fitted to the measured shift profile
        (``mapping='xcorr'``). 0 = constant lag, 2 = lag + bow.
    n_blocks, max_lag, match_level, match_poly_order, min_quality
        Passed to :func:`measure_shift_profile`. ``match_level`` ('plane',
        'poly_rows' with degree ``match_poly_order``, or 'none') is also applied before the power-law model fit
        (``mapping='model'/'model_scaled'``): without leveling, the sample
        tilt dominates the column correlation on tilted samples and the
        measured mapping is meaningless. The leveling is a matching aid only
        and never touches the output data.
    l_points, n_var, maxiter, seed
        Power-law model size and differential-evolution settings.
    smooth_measured : float
        Gaussian smoothing (in blocks) of the measured profile for
        ``mapping='measured'``.
    max_shift_px : float or None
        Clamp the fitted shift to +-this many pixels.
    hysteresis_result : HysteresisResult or None
        Reuse a previously computed power-law fit instead of refitting.
    """
    fwd = np.asarray(fwd, dtype=float)
    bwd = np.asarray(bwd, dtype=float)
    if fwd.shape != bwd.shape:
        raise ValueError(f"forward {fwd.shape} and backward {bwd.shape} shapes differ")
    nx = fwd.shape[1]

    if flip_backward == "auto":
        flip_backward = backward_needs_flip(fwd, bwd)
    flip_backward = bool(flip_backward)
    bwd_oriented = bwd[:, ::-1] if flip_backward else bwd

    corr_before = _similarity(fwd, bwd_oriented)
    rms_before = _rms_difference(fwd, bwd_oriented)
    grid = np.arange(nx, dtype=float)

    if mapping == "none":
        return Alignment(
            fwd=fwd, bwd=bwd_oriented, columns_fwd=grid, columns_bwd=grid,
            shift_px=np.zeros(nx), mapping="none",
            flipped_backward=flip_backward, crop_cols=(0, nx),
            corr_before=corr_before, corr_after=corr_before,
            rms_before=rms_before, rms_after=rms_before,
        )

    # ---- measure the shift profile (needed by everything except 'model') --- #
    centers = meas_shift = quality = None
    if mapping in ("xcorr", "model_scaled", "measured"):
        centers, meas_shift, quality = measure_shift_profile(
            fwd, bwd_oriented, n_blocks=n_blocks, max_lag=max_lag,
            match_level=match_level, match_poly_order=match_poly_order,
            min_quality=min_quality)
        good = quality > 0
        if good.sum() < 2:
            raise RuntimeError(
                "not enough usable blocks to measure the forward/backward "
                "shift (the image may have too little contrast); "
                "try mapping='none' or a smaller n_blocks")

    res = None
    t_grid = grid / (nx - 1.0)

    if mapping == "xcorr":
        order = min(int(poly_order), int(good.sum()) - 1)
        coeffs = np.polyfit(centers[good] / (nx - 1.0), meas_shift[good],
                            max(order, 0), w=quality[good])
        shift = np.polyval(coeffs, t_grid)

    elif mapping == "measured":
        s = meas_shift.copy()
        s[~good] = np.interp(centers[~good], centers[good], meas_shift[good]) \
            if good.sum() else 0.0
        if smooth_measured and smooth_measured > 0:
            s = gaussian_filter1d(s, smooth_measured, mode="nearest")
        shift = np.interp(grid, centers, s)

    elif mapping in ("model", "model_scaled"):
        shape_u, res = _hysteresis_model_shape(
            fwd, bwd_oriented, l_points, n_var, maxiter, seed,
            match_level=match_level, match_poly_order=match_poly_order,
            hysteresis_result=hysteresis_result)
        shape_px = np.interp(t_grid, res.x_c, shape_u) * (nx - 1)
        if mapping == "model":
            shift = shape_px
        else:
            # least squares c0 + c1 * shape against the measured profile
            basis = np.stack([
                np.ones(good.sum()),
                np.interp(centers[good] / (nx - 1.0), res.x_c, shape_u) * (nx - 1),
            ], axis=1)
            w = quality[good][:, None]
            c, *_ = np.linalg.lstsq(basis * w, meas_shift[good] * w[:, 0], rcond=None)
            shift = c[0] + c[1] * shape_px
    else:
        raise ValueError(f"unknown mapping {mapping!r}")

    if max_shift_px is not None:
        shift = np.clip(shift, -float(max_shift_px), float(max_shift_px))

    # ---- apply the warp ---------------------------------------------------- #
    if warp == "bwd_to_fwd":
        columns_fwd = grid
        columns_bwd = grid + shift
    elif warp == "split":
        columns_fwd = grid - 0.5 * shift
        columns_bwd = grid + 0.5 * shift
    elif warp == "linearize_both":
        if res is None:
            raise ValueError(
                "warp='linearize_both' requires mapping='model' or 'model_scaled'")
        # Both images sample a common physical coordinate s: the forward image
        # at column f(s) and the backward image at column g(s) (swapped when
        # the fit targeted the retrace->trace mapping).
        f_curve, g_curve = res.f_x, res.g_x
        if not res.fit_uses_tr:
            f_curve, g_curve = g_curve, f_curve
        columns_fwd = np.interp(t_grid, res.x_c, _monotone(f_curve)) * (nx - 1)
        columns_bwd = np.interp(t_grid, res.x_c, _monotone(g_curve)) * (nx - 1)
    else:
        raise ValueError(f"unknown warp {warp!r}")

    out_fwd = fwd if warp == "bwd_to_fwd" else resample_columns(
        fwd, columns_fwd, interp)
    out_bwd = resample_columns(bwd_oriented, columns_bwd, interp)

    # Columns where both scans sampled real data. Where the (shifted) source
    # position falls outside the image, resample_columns held the edge value -
    # that area was never imaged in that direction and should be cropped off.
    eps = 1e-6
    valid = ((columns_fwd >= -eps) & (columns_fwd <= nx - 1 + eps)
             & (columns_bwd >= -eps) & (columns_bwd <= nx - 1 + eps))
    idx = np.nonzero(valid)[0]
    crop_cols = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, nx)

    non_constant = shift - shift.mean()
    return Alignment(
        fwd=out_fwd,
        bwd=out_bwd,
        columns_fwd=columns_fwd,
        columns_bwd=columns_bwd,
        shift_px=shift,
        mapping=mapping,
        flipped_backward=flip_backward,
        crop_cols=crop_cols,
        max_shift_px=float(np.max(np.abs(shift))),
        lag_px=float(shift.mean()),
        bow_px=float(non_constant.max() - non_constant.min()),
        corr_before=corr_before,
        corr_after=_similarity(out_fwd, out_bwd),
        rms_before=rms_before,
        rms_after=_rms_difference(out_fwd, out_bwd),
        measured_centers=centers,
        measured_shift_px=meas_shift,
        measured_quality=quality,
        fit_rms=float(getattr(res, "fit_rms", np.nan)) if res is not None else float("nan"),
        hysteresis_result=res,
    )


# --------------------------------------------------------------------------- #
#  Stage 2: parachuting detection
# --------------------------------------------------------------------------- #
def _detrend_rows(image):
    """Subtract a straight line from every row. Used only to keep the sample
    tilt out of the height-difference statistics; never applied to the output."""
    image = np.asarray(image, dtype=float)
    nx = image.shape[1]
    t = np.linspace(-1.0, 1.0, nx)
    # least squares slope/offset per row, closed form
    slope = (image * t).sum(axis=1) / (t * t).sum()
    offset = image.mean(axis=1)
    return image - (slope[:, None] * t[None, :] + offset[:, None])


def difference_histogram(image, direction=+1, max_delta=20, dz_bins=120,
                         dz_range=None, detrend=True):
    """The H(delta, dz) histogram of Kubo et al.

    For every scan line and every pair of pixels separated by ``delta`` along
    the fast axis, histogram the height change dz = z(j+delta) - z(j) measured
    in scan-time order.

    Returns ``(hist, deltas, dz_edges)`` with ``hist`` of shape
    ``(max_delta, dz_bins)``.

    `direction` is +1 when the scan runs toward increasing column index and -1
    when it runs the other way (a backward channel that the controller has
    already mirrored back into spatial order)."""
    image = np.asarray(image, dtype=float)
    if detrend:
        image = _detrend_rows(image)
    if direction < 0:
        image = image[:, ::-1]

    deltas = np.arange(1, int(max_delta) + 1)
    if dz_range is None:
        sample = image[:, deltas[-1]:] - image[:, :-deltas[-1]]
        lim = float(np.percentile(np.abs(sample), 99.9)) or 1.0
        dz_range = (-lim, lim)
    dz_edges = np.linspace(dz_range[0], dz_range[1], int(dz_bins) + 1)

    hist = np.empty((len(deltas), int(dz_bins)), dtype=float)
    for i, d in enumerate(deltas):
        dz = (image[:, d:] - image[:, :-d]).ravel()
        hist[i], _ = np.histogram(dz, bins=dz_edges)
    return hist, deltas, dz_edges


def estimate_fall_slope(image, direction=+1, max_delta=20, percentile=0.2,
                        detrend=True):
    """Estimate the tip's maximum fall rate (the lower border of the H(delta,dz)
    histogram) in height-units per pixel, as a positive number.

    A low percentile of dz is taken for each delta and a line through the
    origin is fitted to it. Returns ``(slope, envelope, deltas)``."""
    image = np.asarray(image, dtype=float)
    if detrend:
        image = _detrend_rows(image)
    if direction < 0:
        image = image[:, ::-1]
    deltas = np.arange(1, int(max_delta) + 1)
    envelope = np.array([
        np.percentile(image[:, d:] - image[:, :-d], percentile) for d in deltas
    ])
    # least squares through the origin: dz = -slope * delta
    slope = -float(np.dot(deltas, envelope) / np.dot(deltas, deltas))
    return max(slope, 0.0), envelope, deltas


def detect_parachuting(image, slope, offset=0.0, direction=+1, max_delta=20,
                       detrend=True):
    """Flag pixels lying below the decision line dz = -(slope*delta + offset).

    A pixel is flagged when, for some lag ``delta`` in ``1..max_delta``, the
    height has fallen from the pixel ``delta`` steps earlier in scan time by
    more than the tip can physically follow - i.e. the tip was still airborne
    when this pixel was recorded.

    ``slope`` is in height-units per pixel (positive), ``offset`` in
    height-units (positive makes the criterion stricter)."""
    image = np.asarray(image, dtype=float)
    ref = _detrend_rows(image) if detrend else image
    if direction < 0:
        ref = ref[:, ::-1]

    mask = np.zeros(ref.shape, dtype=bool)
    for d in range(1, int(max_delta) + 1):
        limit = -(float(slope) * d + float(offset))
        mask[:, d:] |= (ref[:, d:] - ref[:, :-d]) <= limit
    if direction < 0:
        mask = mask[:, ::-1]
    return mask


# --------------------------------------------------------------------------- #
#  Stage 3: soft-min merge
# --------------------------------------------------------------------------- #
def softmin(x, y, beta):
    """Kubo et al. eq. (3): a minimum-biased average of two heights.

        softmin(x,y) = -1/b * ln(e^-bx + e^-by) + ln(2)/b * exp(-b^2 (x-y)^2)

    The second term is the correction that makes softmin(x, x) == x exactly.
    beta -> 0 gives the arithmetic mean, beta -> infinity the hard minimum.

    Evaluated in the numerically stable form
    ``min - ln(1+e^-b*d)/b + ln(2)/b * e^-(b*d)^2`` with ``d = |x-y|``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = float(beta)
    if beta <= 0:
        return 0.5 * (x + y)
    lo = np.minimum(x, y)
    d = np.abs(x - y)
    bd = beta * d
    return lo - np.log1p(np.exp(-bd)) / beta + (np.log(2.0) / beta) * np.exp(-bd * bd)


def combine_scans(fwd, bwd, combine="average", weight=0.5, beta=0.0,
                  slope_gain=2.0, consensus_size=5):
    """Combine two aligned scans pixel by pixel, ignoring any parachuting mask.

    ``combine``:
      ``average``   - weighted average ``weight*fwd + (1-weight)*bwd``.
                      ``weight=0.5`` is the plain mean (lowest noise, no bias),
                      ``1`` keeps the forward scan, ``0`` the backward one.
      ``slope``     - slope-directional blend. The tip tracks *rising* edges
                      well and lags on *falling* ones, and the two scans move
                      in opposite directions - so at every pixel the scan that
                      was climbing there is the trustworthy one. The local
                      x-gradient g of the mean image sets a smooth weight
                      ``w_fwd = sigmoid(slope_gain * g / std(g))``: uphill in
                      +x trusts the forward scan, downhill the backward one.
                      Flat areas fall back to the plain mean. Sharpens edges
                      compared to a plain average without a hard seam.
      ``consensus`` - outlier rejection: at each pixel keep whichever scan is
                      closer to the local (``consensus_size`` box) mean of the
                      two. Rejects single-scan glitches; keeps real structure.
      ``softmin``   - the paper's soft-minimum with parameter ``beta``.
      ``min``/``max`` - hard minimum / maximum of the two.
      ``forward``/``backward`` - take one scan unchanged (useful for
                      comparing against the merge, or for using the alignment
                      and parachuting stages on their own).

    ``combine='correlation'`` is not handled here: it needs the auxiliary
    channels and lives in :func:`correlation_select` /
    :func:`process_two_way`.
    """
    fwd = np.asarray(fwd, dtype=float)
    bwd = np.asarray(bwd, dtype=float)
    if combine == "average":
        w = float(np.clip(weight, 0.0, 1.0))
        return w * fwd + (1.0 - w) * bwd
    if combine == "slope":
        mean = 0.5 * (fwd + bwd)
        grad = np.gradient(gaussian_filter(mean, 1.0), axis=1)
        scale = float(np.std(grad)) or 1.0
        w = 1.0 / (1.0 + np.exp(-float(slope_gain) * grad / scale))
        return w * fwd + (1.0 - w) * bwd
    if combine == "consensus":
        ref = uniform_filter(0.5 * (fwd + bwd), size=max(1, int(consensus_size)))
        return np.where(np.abs(fwd - ref) <= np.abs(bwd - ref), fwd, bwd)
    if combine == "softmin":
        return softmin(fwd, bwd, beta)
    if combine == "min":
        return np.minimum(fwd, bwd)
    if combine == "max":
        return np.maximum(fwd, bwd)
    if combine == "forward":
        return fwd.copy()
    if combine == "backward":
        return bwd.copy()
    raise ValueError(f"unknown combine {combine!r}")


def merge_two_way(fwd, bwd, mask_fwd=None, mask_bwd=None, combine="average",
                  weight=0.5, beta=0.0, slope_gain=2.0, consensus_size=5,
                  both_flagged="paper"):
    """Merge two aligned scans into one image.

    Away from any flagged pixel the two scans are combined with
    :func:`combine_scans`. Where one scan is flagged as parachuting its value
    is discarded in favour of the other scan (parachuting can only ever make a
    height too *high*).

    ``both_flagged`` decides what to do where both scans are flagged:
      ``paper``    - follow the paper's precedence: forward flagged -> take
                     backward (even if the backward is flagged too)
      ``min``      - take the lower of the two
      ``softmin``  - fall back to the combined value
    """
    fwd = np.asarray(fwd, dtype=float)
    bwd = np.asarray(bwd, dtype=float)
    out = combine_scans(fwd, bwd, combine, weight, beta, slope_gain,
                        consensus_size)
    return _replace_flagged(out, fwd, bwd, mask_fwd, mask_bwd, both_flagged)


def _replace_flagged(out, fwd, bwd, mask_fwd, mask_bwd, both_flagged="paper"):
    """Overwrite flagged (parachuting) pixels of a combined image with the
    value from the opposite scan; see :func:`merge_two_way`."""
    if mask_fwd is None and mask_bwd is None:
        return out

    mf = np.zeros(fwd.shape, bool) if mask_fwd is None else np.asarray(mask_fwd, bool)
    mb = np.zeros(fwd.shape, bool) if mask_bwd is None else np.asarray(mask_bwd, bool)

    both = mf & mb
    if both_flagged == "paper":
        out = np.where(mf, bwd, out)          # forward flagged -> backward
        out = np.where(mb & ~mf, fwd, out)    # only backward flagged -> forward
    else:
        out = np.where(mf & ~mb, bwd, out)
        out = np.where(mb & ~mf, fwd, out)
        if both_flagged == "min":
            out = np.where(both, np.minimum(fwd, bwd), out)
        elif both_flagged != "softmin":
            raise ValueError(f"unknown both_flagged {both_flagged!r}")
    return out


# --------------------------------------------------------------------------- #
#  Correlation-gated merge (average where the scans agree, referee elsewhere)
# --------------------------------------------------------------------------- #
def apply_alignment(alignment, image, which="bwd", interp="spline"):
    """Warp an arbitrary same-shape image (e.g. a phase or error channel)
    exactly the way the height image of scan direction `which` was warped by
    :func:`align_two_way` (including the mirror of the backward scan)."""
    image = np.asarray(image, dtype=float)
    if which == "bwd":
        if alignment.flipped_backward:
            image = image[:, ::-1]
        cols = alignment.columns_bwd
    elif which == "fwd":
        cols = alignment.columns_fwd
    else:
        raise ValueError(f"unknown direction {which!r}")
    if np.array_equal(cols, np.arange(image.shape[1], dtype=float)):
        return image.copy()
    return resample_columns(image, cols, interp)


def local_correlation(a, b, window=11):
    """Windowed zero-mean normalized cross-correlation between two images:
    at every pixel, the Pearson correlation of the two `window` x `window`
    neighbourhoods. 1 = identical local pattern, 0 = unrelated, -1 = inverted.
    Insensitive to any local offset or gain difference between the images."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = max(3, int(window))
    ma = uniform_filter(a, w)
    mb = uniform_filter(b, w)
    cov = uniform_filter(a * b, w) - ma * mb
    va = np.maximum(uniform_filter(a * a, w) - ma * ma, 0.0)
    vb = np.maximum(uniform_filter(b * b, w) - mb * mb, 0.0)
    return np.clip(cov / (np.sqrt(va * vb) + 1e-30), -1.0, 1.0)


def correlation_select(fwd, bwd, aux_pairs=(), margin=0.7, window=11,
                       weight=0.5, shared=None, return_scores=False):
    """Correlation-gated merge of two aligned scans.

    The local (windowed) correlation between the scans decides, pixel by
    pixel, whether the two directions imaged the same feature:

    * ``corr >= margin`` - the feature is shared: take the combined value
      ``shared`` (any :func:`combine_scans` output, e.g. a soft-min); when
      ``shared`` is None the weighted average
      ``weight`` * fwd + (1-``weight``) * bwd is used.
    * ``corr < margin``  - the scans disagree there. The auxiliary channels
      (phase / error) referee: each direction's height is locally correlated
      against the direction-averaged auxiliary pattern (which is much less
      affected by a height artifact of one direction), and the pixel is kept
      from the direction with the highest mean |correlation|. Without any
      auxiliary channel the direction closer to the local mean wins instead.

    Returns ``(merged, corr_map, decision)`` where ``decision`` is 0 where the
    scans were combined, 1 where the forward pixel was kept and 2 where the
    backward pixel was kept. With ``return_scores=True`` a fourth element is
    appended: a dict with the referee internals (``score_fwd``, ``score_bwd``,
    and the ``aux_refs`` reference patterns), for diagnostics.
    """
    fwd = np.asarray(fwd, dtype=float)
    bwd = np.asarray(bwd, dtype=float)
    corr = local_correlation(fwd, bwd, window)
    if shared is None:
        w = float(np.clip(weight, 0.0, 1.0))
        avg = w * fwd + (1.0 - w) * bwd
    else:
        avg = np.asarray(shared, dtype=float)

    disputed = corr < float(margin)
    decision = np.zeros(fwd.shape, np.uint8)

    score_f = np.zeros(fwd.shape)
    score_b = np.zeros(fwd.shape)
    refs = []
    for aux_f, aux_b in aux_pairs:
        ref = 0.5 * (np.asarray(aux_f, dtype=float)
                     + np.asarray(aux_b, dtype=float))
        refs.append(ref)
        # |corr|: an edge shows up in phase/error with an arbitrary sign
        score_f += np.abs(local_correlation(fwd, ref, window))
        score_b += np.abs(local_correlation(bwd, ref, window))
    if not refs:
        ref = uniform_filter(avg, max(3, int(window)))
        score_f = -np.abs(fwd - ref)
        score_b = -np.abs(bwd - ref)

    take_f = score_f >= score_b
    merged = np.where(disputed, np.where(take_f, fwd, bwd), avg)
    decision[disputed & take_f] = 1
    decision[disputed & ~take_f] = 2
    if return_scores:
        return merged, corr, decision, {
            "score_fwd": score_f, "score_bwd": score_b, "aux_refs": refs}
    return merged, corr, decision


# --------------------------------------------------------------------------- #
#  Full pipeline
# --------------------------------------------------------------------------- #
@dataclass
class TwoWayResult:
    merged: np.ndarray
    fwd: np.ndarray                 # forward image on the output grid
    bwd: np.ndarray                 # backward image on the output grid
    mask_fwd: np.ndarray
    mask_bwd: np.ndarray
    alignment: Alignment
    slope_fwd: float = float("nan")
    slope_bwd: float = float("nan")
    fraction_fwd: float = 0.0       # fraction of pixels flagged in each scan
    fraction_bwd: float = 0.0
    corr_map: np.ndarray = field(repr=False, default=None)
    corr_decision: np.ndarray = field(repr=False, default=None)
    corr_score_fwd: np.ndarray = field(repr=False, default=None)
    corr_score_bwd: np.ndarray = field(repr=False, default=None)
    corr_aux_refs: list = field(repr=False, default=None)

    @property
    def removed(self):
        """What the merge changed relative to the raw forward scan."""
        return self.fwd - self.merged


#: Defaults for every tunable, shared by the GUI and the batch pipeline.
DEFAULTS = dict(
    # -- preprocessing applied to BOTH scans before anything else. Unlike
    #    match_level (a matching aid) this changes the output data: it puts
    #    the two scans on one common background, so pixels replaced from the
    #    opposite scan sit flush with their neighbours.
    pre_plane=False,     # subtract each scan's own fitted plane
    pre_rows=False,      # per-row polynomial alignment of each scan
    pre_rows_order=2,    # ... polynomial degree
    # -- alignment
    align=True,
    mapping="xcorr",
    flip_backward="auto",
    warp="bwd_to_fwd",
    poly_order=2,
    n_blocks=16,
    max_lag=20,
    match_level="plane",
    match_poly_order=2,
    min_quality=0.2,
    l_points=400,
    n_var=10,
    maxiter=40,
    max_shift_px=None,
    smooth_measured=2.0,
    interp="spline",
    # -- parachuting detection
    detect=True,
    max_delta=20,
    slope=None,          # None -> estimate from the histogram envelope
    slope_scale=1.0,     # decision slope = slope_scale * estimated fall rate
    offset=0.0,
    detrend=True,
    envelope_percentile=0.2,
    # -- merge
    combine="average",
    weight=0.5,
    beta=0.0,
    slope_gain=2.0,
    consensus_size=5,
    both_flagged="paper",
    corr_margin=0.7,     # combine='correlation': shared-feature threshold
    corr_window=11,      # ... local correlation window (px)
    corr_combine="average",  # ... how the shared pixels are combined
                             # (any combine_scans mode, e.g. 'softmin')
    # -- output
    crop=True,           # trim edge columns not imaged in both directions
)

#: Alignment keys forwarded to :func:`align_two_way`.
_ALIGN_KEYS = ("mapping", "flip_backward", "warp", "poly_order", "n_blocks",
               "max_lag", "match_level", "match_poly_order", "min_quality",
               "l_points", "n_var",
               "maxiter", "smooth_measured", "max_shift_px", "interp")


def process_two_way(fwd, bwd, hysteresis_result=None, aux_pairs=None,
                    **params):
    """Run alignment -> detection -> merge. Keyword arguments override
    :data:`DEFAULTS`; see that dict for the full list of tunables.

    ``aux_pairs`` - optional list of raw ``(aux_fwd, aux_bwd)`` channel pairs
    (phase, error, ...) used as referees by ``combine='correlation'``. They
    are warped with the same alignment as the height images.

    Returns a :class:`TwoWayResult`."""
    p = dict(DEFAULTS)
    unknown = set(params) - set(p)
    if unknown:
        raise TypeError(f"unknown parameter(s): {sorted(unknown)}")
    p.update(params)

    fwd = np.asarray(fwd, dtype=float)
    bwd = np.asarray(bwd, dtype=float)

    # ---- 0. background correction of both scans -------------------------- #
    # Optional and applied to the DATA (each scan levelled by its own fit),
    # before the hysteresis is found: with a common zero background, a pixel
    # replaced from the opposite scan fits its neighbours.
    if p["pre_plane"]:
        fwd = remove_plane(fwd)
        bwd = remove_plane(bwd)
    if p["pre_rows"]:
        fwd = align_rows_poly(fwd, p["pre_rows_order"])
        bwd = align_rows_poly(bwd, p["pre_rows_order"])

    # ---- 1. hysteresis alignment ---------------------------------------- #
    if p["align"] and p["mapping"] != "none":
        alignment = align_two_way(
            fwd, bwd,
            hysteresis_result=hysteresis_result,
            **{k: p[k] for k in _ALIGN_KEYS},
        )
    else:
        alignment = align_two_way(fwd, bwd, mapping="none",
                                  flip_backward=p["flip_backward"])
    a_fwd, a_bwd = alignment.fwd, alignment.bwd

    # Scan-time direction of each image on the output grid. The forward scan
    # always runs left to right. The backward scan runs right to left unless
    # the raw channel was stored mirrored and we flipped it back.
    dir_fwd = +1
    dir_bwd = +1 if alignment.flipped_backward else -1

    # ---- 2. parachuting detection --------------------------------------- #
    slope_f = slope_b = float("nan")
    mask_f = np.zeros(a_fwd.shape, bool)
    mask_b = np.zeros(a_bwd.shape, bool)
    if p["detect"]:
        if p["slope"] is None:
            slope_f, _, _ = estimate_fall_slope(
                a_fwd, dir_fwd, p["max_delta"], p["envelope_percentile"],
                p["detrend"])
            slope_b, _, _ = estimate_fall_slope(
                a_bwd, dir_bwd, p["max_delta"], p["envelope_percentile"],
                p["detrend"])
            slope_f *= p["slope_scale"]
            slope_b *= p["slope_scale"]
        else:
            slope_f = slope_b = float(p["slope"])
        mask_f = detect_parachuting(a_fwd, slope_f, p["offset"], dir_fwd,
                                    p["max_delta"], p["detrend"])
        mask_b = detect_parachuting(a_bwd, slope_b, p["offset"], dir_bwd,
                                    p["max_delta"], p["detrend"])

    # ---- 3. merge -------------------------------------------------------- #
    corr_map = corr_decision = corr_scores = None
    if p["combine"] == "correlation":
        pairs = [(apply_alignment(alignment, af, "fwd", p["interp"]),
                  apply_alignment(alignment, ab, "bwd", p["interp"]))
                 for af, ab in (aux_pairs or [])]
        shared = combine_scans(a_fwd, a_bwd, p["corr_combine"], p["weight"],
                               p["beta"], p["slope_gain"],
                               p["consensus_size"])
        merged, corr_map, corr_decision, corr_scores = correlation_select(
            a_fwd, a_bwd, pairs, margin=p["corr_margin"],
            window=p["corr_window"], weight=p["weight"], shared=shared,
            return_scores=True)
        merged = _replace_flagged(merged, a_fwd, a_bwd, mask_f, mask_b,
                                  p["both_flagged"])
    else:
        merged = merge_two_way(a_fwd, a_bwd, mask_f, mask_b,
                               combine=p["combine"], weight=p["weight"],
                               beta=p["beta"], slope_gain=p["slope_gain"],
                               consensus_size=p["consensus_size"],
                               both_flagged=p["both_flagged"])

    # ---- 4. crop to the doubly-imaged region ----------------------------- #
    if p["crop"] and alignment.crop_cols is not None:
        c0, c1 = alignment.crop_cols
        if (c0, c1) != (0, merged.shape[1]):
            merged = merged[:, c0:c1]
            a_fwd = a_fwd[:, c0:c1]
            a_bwd = a_bwd[:, c0:c1]
            mask_f = mask_f[:, c0:c1]
            mask_b = mask_b[:, c0:c1]
            if corr_map is not None:
                corr_map = corr_map[:, c0:c1]
                corr_decision = corr_decision[:, c0:c1]
                corr_scores = {
                    "score_fwd": corr_scores["score_fwd"][:, c0:c1],
                    "score_bwd": corr_scores["score_bwd"][:, c0:c1],
                    "aux_refs": [r[:, c0:c1]
                                 for r in corr_scores["aux_refs"]],
                }

    corr_scores = corr_scores or {}
    return TwoWayResult(
        merged=merged, fwd=a_fwd, bwd=a_bwd,
        mask_fwd=mask_f, mask_bwd=mask_b,
        alignment=alignment,
        slope_fwd=slope_f, slope_bwd=slope_b,
        fraction_fwd=float(mask_f.mean()),
        fraction_bwd=float(mask_b.mean()),
        corr_map=corr_map, corr_decision=corr_decision,
        corr_score_fwd=corr_scores.get("score_fwd"),
        corr_score_bwd=corr_scores.get("score_bwd"),
        corr_aux_refs=corr_scores.get("aux_refs"),
    )


# --------------------------------------------------------------------------- #
#  Convenience: read a forward/backward pair out of a .gwy file
# --------------------------------------------------------------------------- #
def backward_title(title):
    """Guess the backward-channel title matching a forward one
    ('Height [Fwd]' -> 'Height [Bwd]'). Returns None if it looks unmatched."""
    for a, b in (("[Fwd]", "[Bwd]"), ("Fwd", "Bwd"),
                 ("forward", "backward"), ("Forward", "Backward"),
                 ("trace", "retrace"), ("Trace", "Retrace")):
        if a in title:
            return title.replace(a, b)
    return None


def find_pair(channels, title):
    """Given the channel dict from gwy_loader and a channel title, return
    ``(forward_title, backward_title)`` or ``(title, None)`` if there is no
    matching backward channel."""
    partner = backward_title(title)
    if partner and partner in channels:
        return title, partner
    # maybe `title` is itself the backward channel
    for a, b in (("[Bwd]", "[Fwd]"), ("Bwd", "Fwd")):
        if a in title and title.replace(a, b) in channels:
            return title.replace(a, b), title
    # a derived channel such as 'Height [Merged]': fall back to the raw pair
    # of the same base name, so the dialog still works after merging once
    match = re.match(r"^(.*?)\s*\[[^\]]*\]$", title)
    if match:
        base = match.group(1).strip()
        f, b = f"{base} [Fwd]", f"{base} [Bwd]"
        if f in channels and b in channels:
            return f, b
    return title, None
