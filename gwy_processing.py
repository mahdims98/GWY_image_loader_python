import gwy_colormaps
import gwy_loader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# --- Core Processing Functions ---


def level_by_plane_fit(data):
    """
    Subtracts a fitted plane (background) from the data (leveling).
    This is useful for removing large-scale tilt from AFM images.

    This implementation follows the method described in the Gwyddion source code
    by constructing and solving the normal equations for the least-squares fit.
    It fits the equation Z = a*X + b*Y + c to the data and returns (Z - fitted_plane).

    Args:
        data (np.ndarray): A 2D numpy array representing the image data.

    Returns:
        np.ndarray: The data with the background plane subtracted.
    """
    ny, nx = data.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = data.flatten()

    # Construct the normal equations: (A.T * A) * x = (A.T * b)
    # where A is the design matrix [x_flat, y_flat, 1] and b is the data z_flat.
    
    # Calculate the sums needed for the matrices, as done in Gwyddion's C code.
    sum_1 = len(z_flat)
    sum_x = np.sum(x_flat)
    sum_y = np.sum(y_flat)
    sum_z = np.sum(z_flat)
    sum_x2 = np.sum(x_flat**2)
    sum_y2 = np.sum(y_flat**2)
    sum_xy = np.sum(x_flat * y_flat)
    sum_xz = np.sum(x_flat * z_flat)
    sum_yz = np.sum(y_flat * z_flat)

    # This is the matrix M = (A.T * A)
    M = np.array([
        [sum_x2, sum_xy, sum_x],
        [sum_xy, sum_y2, sum_y],
        [sum_x,  sum_y,  sum_1]
    ])

    # This is the vector V = (A.T * b)
    V = np.array([sum_xz, sum_yz, sum_z])

    # Solve M * coeffs = V for the coefficients [a, b, c]
    try:
        coeffs = np.linalg.solve(M, V)
    except np.linalg.LinAlgError:
        # Matrix is singular, cannot fit a unique plane. Fallback to zero coeffs.
        coeffs = np.zeros(3)

    a, b, c = coeffs
    # Calculate the fitted plane over the entire image grid
    plane = a * X + b * Y + c

    return data - plane


def level_by_polynomial(data, order=1):
    """
    Subtracts a fitted 2D polynomial background from the data.

    This function fits a polynomial of the form:
    P(x, y) = sum_{i=0..order} sum_{j=0..order, i+j<=order} c_{ij} * x^i * y^j

    This is a common method for background correction and leveling in SPM data,
    replicating the 'Polynomial Background' feature in Gwyddion.

    Args:
        data (np.ndarray): A 2D numpy array representing the image data.
        order (int): The degree of the polynomial to fit. `order=1` is
                     equivalent to plane fitting.

    Returns:
        np.ndarray: The data with the fitted polynomial background subtracted.
    """
    ny, nx = data.shape
    X, Y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = data.flatten()

    # Build the design matrix A for the least-squares problem
    # Each column corresponds to a term x^i * y^j
    cols = []
    for i in range(order + 1):
        for j in range(order + 1):
            if i + j <= order:
                cols.append(x_flat**i * y_flat**j)
    A = np.vstack(cols).T

    # Solve (A.T * A) * coeffs = (A.T * z) for the polynomial coefficients
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback if the fit is unstable
        return data

    # Reconstruct the fitted polynomial surface
    background = np.dot(A, coeffs).reshape(ny, nx)

    return data - background


def level_by_polynomial_xy(data, x_order=1, y_order=1):
    """
    Subtracts a fitted 2D polynomial background with independent orders in
    x and y (like Gwyddion's 'Polynomial Background' horizontal/vertical
    degrees).

    The fitted surface is:
    P(x, y) = sum_{i=0..x_order} sum_{j=0..y_order} c_{ij} * x^i * y^j

    Coordinates are normalized to [-1, 1] for numerical stability at
    higher orders.

    Args:
        data (np.ndarray): A 2D numpy array representing the image data.
        x_order (int): Polynomial degree along the x (column) direction.
        y_order (int): Polynomial degree along the y (row) direction.

    Returns:
        np.ndarray: The data with the fitted polynomial background subtracted.
    """
    ny, nx = data.shape
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    cols = [
        (X**i * Y**j).ravel()
        for i in range(x_order + 1)
        for j in range(y_order + 1)
    ]
    A = np.column_stack(cols)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=None)
    except np.linalg.LinAlgError:
        return data.copy()

    background = (A @ coeffs).reshape(ny, nx)
    return data - background


def align_rows(data, method='polynomial', order=1):
    """
    Corrects horizontal scan lines by subtracting a calculated offset from each row.
    This function replicates Gwyddion's 'Align Rows' functionality.

    Args:
        data (np.ndarray): A 2D numpy array representing the image data.
        method (str): The alignment method to use.
                      - 'polynomial': Fits and subtracts a 1D polynomial from each row.
                      - 'median_diff': Subtracts the median of differences between
                                       a row and its neighbors.
        order (int): The degree of the polynomial to use when method is 'polynomial'.
                     A common choice is 1 for tilt correction.

    Returns:
        np.ndarray: The data with rows aligned.
    """
    ny, nx = data.shape
    corrected_data = data.copy()

    if method == 'polynomial':
        x = np.arange(nx)
        for y in range(ny):
            row = corrected_data[y, :]
            coeffs = np.polyfit(x, row, order)
            background = np.polyval(coeffs, x)
            corrected_data[y, :] = row - background

    elif method == 'median_diff':
        # This method calculates an offset for each row based on its difference
        # from its neighbors, which is robust for images with large features.
        if ny < 2:
            return corrected_data # Not enough rows to compare

        offsets = np.zeros(ny)
        # First row
        offsets[0] = np.median(corrected_data[0, :] - corrected_data[1, :])
        # Middle rows
        for y in range(1, ny - 1):
            diff_prev = corrected_data[y, :] - corrected_data[y - 1, :]
            diff_next = corrected_data[y, :] - corrected_data[y + 1, :]
            offsets[y] = 0.5 * (np.median(diff_prev) + np.median(diff_next))
        # Last row
        offsets[ny - 1] = np.median(corrected_data[ny - 1, :] - corrected_data[ny - 2, :])

        # The offsets array now contains the amount each row should be shifted.
        # We subtract these offsets from the original data.
        for y in range(ny):
            corrected_data[y, :] -= offsets[y]

    else:
        raise ValueError("Unknown method: '{}'. Choose from 'polynomial' or 'median_diff'.".format(method))

    return corrected_data


def crop(data, x0, x1, y0, y1, dx=1.0, dy=1.0, y_from_top=False):
    """
    Crops the data to a rectangular region given in real (spatial) units.

    Args:
        data (np.ndarray): A 2D numpy array.
        x0, x1 (float): Horizontal crop range in spatial units.
        y0, y1 (float): Vertical crop range in spatial units.
        dx, dy (float): Pixel sizes.
        y_from_top (bool): If True, y is measured from the top row down
                           (array convention). If False (default), y is
                           measured from the bottom up, matching plots
                           drawn with origin='upper' and
                           extent=(0, x_real, 0, y_real).

    Returns:
        np.ndarray: The cropped data.

    Raises:
        ValueError: If the requested region is empty.
    """
    ny, nx = data.shape
    j0 = max(0, int(np.floor(x0 / dx)))
    j1 = min(nx, int(np.ceil(x1 / dx)))
    if y_from_top:
        i0 = max(0, int(np.floor(y0 / dy)))
        i1 = min(ny, int(np.ceil(y1 / dy)))
    else:
        i0 = max(0, ny - int(np.ceil(y1 / dy)))
        i1 = min(ny, ny - int(np.floor(y0 / dy)))
    if j1 <= j0 or i1 <= i0:
        raise ValueError("Empty crop region")
    return data[i0:i1, j0:j1].copy()


def set_baseline_to_zero(data):
    """
    Adjusts the data so that the lowest value of the data becomes the new zero.
    This is useful for aligning the lowest features of an image at zero.

    Args:
        data (np.ndarray): A 2D numpy array.

    Returns:
        np.ndarray: The baseline-adjusted data.
    """
    baseline = np.min(data)
    return data - baseline


def filter_by_percentile(data, min_percentile=0.5, max_percentile=99.5):
    """
    Clips the data to a specified percentile range.
    Values below the min_percentile are set to the min_percentile value, and
    values above the max_percentile are set to the max_percentile value.
    This is effective for removing spike noise or outliers.

    Args:
        data (np.ndarray): A 2D numpy array.
        min_percentile (float): The minimum percentile (0-100).
        max_percentile (float): The maximum percentile (0-100).

    Returns:
        np.ndarray: The clipped data.
    """
    vmin = np.percentile(data, min_percentile)
    vmax = np.percentile(data, max_percentile)
    return np.clip(data, vmin, vmax)


def remove_scars(data, threshold=3.0, min_length=5):
    """
    Removes horizontal scars (line defects) from AFM images.

    This implements a Pythonic version of the scar removal logic found in 
    Gwyddion. It identifies horizontal strokes that deviate significantly from 
    their vertical neighbors and interpolates them to fill in the gaps.

    Args:
        data (np.ndarray): A 2D numpy array representing the image data.
        threshold (float): Threshold for scar detection, relative to the RMS
                           variation between adjacent lines. Default is 3.0.
        min_length (int): Minimum length of a scar in pixels to be removed.
                          Default is 5.

    Returns:
        np.ndarray: The corrected data array with scars interpolated.
    """
    ny, nx = data.shape
    if ny < 3:
        return data.copy()

    # 1. Detect deviations from vertical neighbors
    deviation = np.zeros_like(data)
    deviation[1:-1, :] = data[1:-1, :] - 0.5 * (data[:-2, :] + data[2:, :])
    deviation[0, :] = data[0, :] - data[1, :]
    deviation[-1, :] = data[-1, :] - data[-2, :]

    # 2. Determine threshold based on RMS of deviations
    rms_dev = np.std(deviation)
    abs_thresh = threshold * rms_dev

    # 3. Create initial mask of anomalies
    is_anomaly = np.abs(deviation) > abs_thresh

    # 4. Filter anomalies by minimum length along horizontal lines
    scar_mask = np.zeros_like(is_anomaly, dtype=bool)
    for y in range(ny):
        padded = np.pad(is_anomaly[y, :], (1, 1), mode='constant', constant_values=False)
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for start, end in zip(starts, ends):
            if (end - start) >= min_length:
                scar_mask[y, start:end] = True

    # 5. Interpolate scar pixels from vertical neighbors
    corrected_data = data.copy()
    for x in range(nx):
        col_mask = scar_mask[:, x]
        if np.any(col_mask):
            valid_indices = np.where(~col_mask)[0]
            if len(valid_indices) >= 2:
                scar_indices = np.where(col_mask)[0]
                valid_values = corrected_data[valid_indices, x]
                corrected_data[scar_indices, x] = np.interp(scar_indices, valid_indices, valid_values)

    return corrected_data


def _fft_freq_grids(shape, dx=1.0, dy=1.0):
    """(FX, FY) frequency coordinate grids on the fftshift-ed layout."""
    ny, nx = shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return np.meshgrid(freq_x, freq_y)


def get_2d_fft_magnitude(data, dx=1.0, dy=1.0):
    """
    Calculates the 2D FFT magnitude spectrum (in decibels) and frequency extents.

    No window is applied, so the spectrum shown is exactly the one the FFT
    filters operate on. The FFT is normalized by the number of pixels: the
    DC bin holds the mean value of the image and the dB scale is
    independent of the image size.

    Args:
        data (np.ndarray): 2D numpy array.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.

    Returns:
        (np.ndarray, list): The dB magnitude on the fftshift-ed grid and
        the matching [left, right, bottom, top] extent for an
        origin='upper' imshow. The extent runs along the frequency-bin
        EDGES (each bin is drawn centred on its frequency), so cursor
        coordinates on the plot map exactly onto bin frequencies.
    """
    ny, nx = data.shape
    fshift = np.fft.fftshift(np.fft.fft2(data)) / (nx * ny)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-12)

    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    hx = 0.5 / (nx * dx)   # half a frequency bin
    hy = 0.5 / (ny * dy)
    extent = [freq_x[0] - hx, freq_x[-1] + hx,
              freq_y[-1] + hy, freq_y[0] - hy]
    return magnitude_spectrum, extent


def build_pass_mask(shape, dx=1.0, dy=1.0, mode='lowpass', cutoff=10.0):
    """
    Builds a boolean frequency-domain mask for a radial lowpass or
    highpass filter, on the fftshift-ed grid (matching
    `filter_by_2d_fft_mask`).

    The DC bin is always kept, so a highpass filters the texture without
    shifting the mean height of the image.

    Args:
        shape (tuple): (ny, nx) shape of the image.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        mode (str): 'lowpass' or 'highpass'.
        cutoff (float): Cutoff frequency in the same inverse units as dx/dy.

    Returns:
        np.ndarray: Boolean mask (True = keep frequency).
    """
    FX, FY = _fft_freq_grids(shape, dx, dy)
    F_dist = np.sqrt(FX**2 + FY**2)
    if mode == 'lowpass':
        return F_dist <= cutoff
    if mode == 'highpass':
        mask = F_dist > cutoff
        mask[F_dist == 0] = True   # keep the mean height
        return mask
    raise ValueError("Unknown mode: '{}'".format(mode))


def filter_by_2d_fft(data, cutoff_freq, mode='lowpass', dx=1.0, dy=1.0):
    """
    Applies a hard radial 2D FFT lowpass or highpass filter.

    Args:
        data (np.ndarray): 2D numpy array.
        cutoff_freq (float): Cutoff frequency in the same inverse units as dx/dy.
        mode (str): 'lowpass' or 'highpass'.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.

    Returns:
        np.ndarray: The filtered data.
    """
    mask = build_pass_mask(data.shape, dx=dx, dy=dy,
                           mode=mode, cutoff=cutoff_freq)
    return filter_by_2d_fft_mask(data, mask)


def filter_by_2d_fft_mask(data, mask):
    """
    Keeps only the frequencies where `mask` is True and transforms back.
    This is the single code path every FFT filter goes through
    (lowpass/highpass via `build_pass_mask`, notches via
    `build_notch_mask`, bands via `build_band_mask` - masks combine
    with `&`).

    Args:
        data (np.ndarray): 2D numpy array.
        mask (np.ndarray): A 2D boolean or binary array of the same shape
                           as data, on the fftshift-ed frequency grid.
                           It must be point-symmetric about the origin so
                           the filtered image stays real (every mask
                           builder here is); the numerically tiny
                           imaginary residue is discarded.

    Returns:
        np.ndarray: The filtered data.
    """
    fshift = np.fft.fftshift(np.fft.fft2(data))
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
    return np.real(img_back)


def build_notch_mask(shape, dx=1.0, dy=1.0, notches=(), radius=0.1):
    """
    Builds a boolean frequency-domain mask with circular notches removed.

    For each notch position (fx, fy) the mask is set to False inside a
    circle of the given radius around BOTH (fx, fy) and (-fx, -fy), so the
    mask stays symmetric and the filtered image stays real.

    The mask is defined on the fftshift-ed frequency grid, matching
    `filter_by_2d_fft_mask`.

    Args:
        shape (tuple): (ny, nx) shape of the image.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        notches (iterable): Sequence of (fx, fy) notch center frequencies.
        radius (float): Notch radius in frequency units.

    Returns:
        np.ndarray: Boolean mask (True = keep frequency).
    """
    FX, FY = _fft_freq_grids(shape, dx, dy)

    mask = np.ones(shape, dtype=bool)
    for nfx, nfy in notches:
        for sx, sy in ((nfx, nfy), (-nfx, -nfy)):
            mask &= ((FX - sx) ** 2 + (FY - sy) ** 2) > radius**2
    return mask


def build_band_mask(shape, dx=1.0, dy=1.0, x_bands=(), y_bands=(), half_width=0.5):
    """
    Builds a boolean frequency-domain mask with straight bands removed
    (line notches).

    This targets noise that shows up as full stripes in the spectrum, e.g.
    single-frequency interference along the fast scan axis, which appears
    as a vertical line at some fx spanning all fy.

    Args:
        shape (tuple): (ny, nx) shape of the image.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        x_bands (iterable): fx center frequencies. For each center c the
                            vertical stripes |fx - c| < half_width and
                            |fx + c| < half_width are removed (all fy).
        y_bands (iterable): fy center frequencies, removing horizontal
                            stripes in the same symmetric way.
        half_width (float): Half-width of each band in frequency units.

    Returns:
        np.ndarray: Boolean mask (True = keep frequency), on the
                    fftshift-ed grid, matching `filter_by_2d_fft_mask`.
    """
    FX, FY = _fft_freq_grids(shape, dx, dy)

    mask = np.ones(shape, dtype=bool)
    for c in x_bands:
        mask &= (np.abs(FX - c) > half_width) & (np.abs(FX + c) > half_width)
    for c in y_bands:
        mask &= (np.abs(FY - c) > half_width) & (np.abs(FY + c) > half_width)
    return mask


def build_rect_mask(shape, dx=1.0, dy=1.0, rects=()):
    """
    Builds a boolean frequency-domain mask with axis-aligned rectangular
    patches removed.

    This targets noise that fills an extended rectangular region of the
    spectrum (e.g. a horizontal or vertical streak of excess power that
    is wider than a line but does not span the whole axis).

    Each rectangle is removed at BOTH (fx, fy) and (-fx, -fy), so the
    mask stays symmetric and the filtered image stays real.

    Args:
        shape (tuple): (ny, nx) shape of the image.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        rects (iterable): Sequence of (fx, fy, wx, wy) rectangles - the
                          center frequency and the FULL widths along fx
                          and fy, in frequency units.

    Returns:
        np.ndarray: Boolean mask (True = keep frequency), on the
                    fftshift-ed grid, matching `filter_by_2d_fft_mask`.
    """
    FX, FY = _fft_freq_grids(shape, dx, dy)

    mask = np.ones(shape, dtype=bool)
    for rfx, rfy, wx, wy in rects:
        for sx, sy in ((rfx, rfy), (-rfx, -rfy)):
            mask &= ~((np.abs(FX - sx) <= wx / 2.0) &
                      (np.abs(FY - sy) <= wy / 2.0))
    return mask


def smooth_fft_mask(mask, dx=1.0, dy=1.0, width=0.0):
    """
    Softens the edges of a binary frequency-domain mask with a Gaussian,
    so the filters roll off smoothly instead of cutting hard (a hard
    edge in the frequency domain rings in the image domain).

    Args:
        mask (np.ndarray): Boolean mask on the fftshift-ed grid.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        width (float): Gaussian sigma of the roll-off, in frequency
                       units. 0 returns the hard mask unchanged.

    Returns:
        np.ndarray: Float mask in [0, 1] (or the input mask if width<=0).
    """
    if width <= 0:
        return mask
    ny, nx = mask.shape
    # sigma in frequency BINS per axis: one bin is 1/(n*d) wide
    sigma = (width * ny * dy, width * nx * dx)
    return ndimage.gaussian_filter(mask.astype(float), sigma, mode="nearest")


def fft_excess_db(data, dx=1.0, dy=1.0, apodize=True):
    """
    dB magnitude of the spectrum ABOVE its local radial background.

    The magnitude spectrum of real topography falls off smoothly with
    |f|, so a fixed threshold on the raw spectrum flags the whole low-
    and mid-frequency image content. Instead the background is estimated
    as the median dB magnitude in annuli of constant |f| (the median is
    robust against the noise features themselves), lightly smoothed
    along the radius, and subtracted. The result is ~0 dB wherever the
    spectrum is ordinary, and positive where something sticks out of it.

    With `apodize` (default) the mean is removed and a Hann window is
    applied before the FFT, for THIS ANALYSIS ONLY: the image boundaries
    are not periodic, and the wrap-around jumps otherwise leak power
    into a wide cross along both frequency axes that buries real
    features near the axes. Nothing here ever windows the data being
    filtered - the excess map is used to FIND noise, not to remove it.

    Args:
        data (np.ndarray): 2D numpy array.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        apodize (bool): Suppress boundary leakage before the analysis.

    Returns:
        (np.ndarray, list): The excess map on the fftshift-ed grid and
        the matching imshow extent (as in `get_2d_fft_magnitude`).
    """
    ny, nx = data.shape
    if apodize:
        d = (data - data.mean()) * np.outer(np.hanning(ny), np.hanning(nx))
    else:
        d = data
    fshift = np.fft.fftshift(np.fft.fft2(d)) / (nx * ny)
    mag_db = 20 * np.log10(np.abs(fshift) + 1e-12)
    hx, hy = 0.5 / (nx * dx), 0.5 / (ny * dy)
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    extent = [freq_x[0] - hx, freq_x[-1] + hx,
              freq_y[-1] + hy, freq_y[0] - hy]

    FX, FY = _fft_freq_grids((ny, nx), dx, dy)
    r_phys = np.hypot(FX, FY)
    n_annuli = max(ny, nx) // 2
    r_idx = np.minimum((r_phys / r_phys.max() * n_annuli).astype(int),
                       n_annuli).ravel()

    order = np.argsort(r_idx, kind="stable")
    r_sorted = r_idx[order]
    v_sorted = mag_db.ravel()[order]
    edges = np.searchsorted(r_sorted, np.arange(n_annuli + 2))
    profile = np.full(n_annuli + 1, np.nan)
    for k in range(n_annuli + 1):
        if edges[k + 1] > edges[k]:
            profile[k] = np.median(v_sorted[edges[k]:edges[k + 1]])
    good = np.flatnonzero(~np.isnan(profile))
    profile = np.interp(np.arange(profile.size), good, profile[good])
    profile = ndimage.median_filter(profile, size=5, mode="nearest")

    background = profile[r_idx].reshape(ny, nx)
    return mag_db - background, extent


def detect_fft_noise(data, dx=1.0, dy=1.0, protect_radius=0.0,
                     peak_db=12.0, streak_db=None, max_items=50, pad=1):
    """
    Systematic detection of periodic noise in the 2D FFT spectrum.

    Everything is measured on the EXCESS spectrum (`fft_excess_db`): the
    dB magnitude above the local radial background. Real topography sits
    near zero excess whatever its shape, so only genuinely anomalous
    power is reported. Three kinds of noise are searched for, each with
    its own statistically matched test:

    1. STREAKS - noise with a fixed frequency along one scan axis but no
       phase coherence in the other direction fills a whole column (or
       row) of the spectrum. A column/row is a streak when the MEDIAN
       excess along it exceeds `streak_db`: the median over hundreds of
       bins is a very stable statistic, so a consistent streak only a
       few dB high is detected while no isolated feature can trigger
       it. Adjacent streak columns/rows are merged and reported as one
       full-height (full-width) rectangle.
    2. AXIS PEAKS - interference that IS phase-coherent from line to
       line sits exactly on the fy=0 axis (or fx=0). Those axes carry
       real directional image content, so each is compared against its
       own running median; sharp local peaks more than `peak_db` above
       it become circular notches.
    3. OFF-AXIS REGIONS - 8-connected patches of bins with excess above
       `peak_db`, with the axes, the protected center and the detected
       streaks excluded. Compact patches (up to ~4 bins across) become
       circular notches, extended ones bounding-box rectangles padded
       by `pad` bins.

    Anything centered within `protect_radius` of the origin, or whose
    box would cover the origin (the central cross), is dropped. Only one
    member of each conjugate +/- pair is returned.

    Args:
        data (np.ndarray): 2D image data.
        dx, dy (float): Pixel sizes.
        protect_radius (float): Protected low-frequency radius.
        peak_db (float): Excess threshold for axis peaks and regions.
        streak_db (float): Median-excess threshold for streaks. Defaults
                           to peak_db / 4, at least 2 dB.
        max_items (int): Maximum total number of items returned.
        pad (int): Bins added around each streak / region bounding box.

    Returns:
        (notches, rects): notches is a list of (fx, fy) circular-notch
        centers, rects a list of (fx, fy, wx, wy) rectangles (center
        and full widths); both sorted strongest first.
    """
    ny, nx = data.shape
    excess, _ = fft_excess_db(data, dx, dy)
    FX, FY = _fft_freq_grids((ny, nx), dx, dy)
    freq_x, freq_y = FX[0], FY[:, 0]
    dfx = 1.0 / (nx * dx)
    dfy = 1.0 / (ny * dy)
    mid_y, mid_x = ny // 2, nx // 2
    if streak_db is None:
        streak_db = max(2.0, peak_db / 4.0)

    def _runs(flags):
        idx = np.flatnonzero(flags)
        if idx.size == 0:
            return []
        splits = np.flatnonzero(np.diff(idx) > 1) + 1
        return [(int(g[0]), int(g[-1])) for g in np.split(idx, splits)]

    found = []                              # (strength, kind, item)
    blocked_cols = np.zeros(nx, dtype=bool)   # bins explained by a streak
    blocked_rows = np.zeros(ny, dtype=bool)

    # -- 1. streaks: columns / rows with elevated median excess ---------
    col_med = np.median(excess, axis=0)
    for i0, i1 in _runs(col_med > streak_db):
        blocked_cols[max(i0 - pad, 0):i1 + pad + 1] = True
        fx0, fx1 = freq_x[i0] - pad * dfx, freq_x[i1] + pad * dfx
        cx, wx = (fx0 + fx1) / 2.0, fx1 - fx0
        if abs(cx) <= max(protect_radius, wx / 2.0):
            continue        # the central cross is not removable noise
        found.append((float(col_med[i0:i1 + 1].max()), "rect",
                      (cx, 0.0, wx, 1.0 / dy)))
    row_med = np.median(excess, axis=1)
    for i0, i1 in _runs(row_med > streak_db):
        blocked_rows[max(i0 - pad, 0):i1 + pad + 1] = True
        fy0 = min(freq_y[i0], freq_y[i1]) - pad * dfy
        fy1 = max(freq_y[i0], freq_y[i1]) + pad * dfy
        cy, wy = (fy0 + fy1) / 2.0, fy1 - fy0
        if abs(cy) <= max(protect_radius, wy / 2.0):
            continue
        found.append((float(row_med[i0:i1 + 1].max()), "rect",
                      (0.0, cy, 1.0 / dx, wy)))

    # -- 2. axis peaks: sharp features on the fx=0 / fy=0 lines ---------
    def _axis_peaks(line, freqs):
        base = ndimage.median_filter(line, size=max(9, 4 * pad + 1),
                                     mode="nearest")
        rel = line - base
        out = []
        # +4 dB over the region threshold: a single axis bin has the whole
        # noise-floor tail to beat, like the lone-bin case below
        for i in np.flatnonzero((rel > peak_db + 4.0)
                                & (freqs > protect_radius)):
            lo, hi = max(i - 1, 0), min(i + 1, len(line) - 1)
            if line[i] >= line[lo] and line[i] >= line[hi]:
                out.append((float(rel[i]), int(i)))
        return out

    # a peak on the fy=0 axis is already covered by a VERTICAL streak at
    # the same fx (and vice versa) - but not by the perpendicular one
    for s, i in _axis_peaks(excess[mid_y], freq_x):
        if not blocked_cols[i]:
            found.append((s, "notch", (float(freq_x[i]), 0.0)))
    for s, i in _axis_peaks(excess[:, mid_x], freq_y):
        if not blocked_rows[i]:
            found.append((s, "notch", (0.0, float(freq_y[i]))))

    # -- 3. off-axis regions of excess power ----------------------------
    cand = (excess > peak_db) & (np.hypot(FX, FY) > protect_radius)
    cand[:, blocked_cols] = False
    cand[blocked_rows, :] = False
    cand[mid_y, :] = False
    cand[:, mid_x] = False
    labels, n_labels = ndimage.label(cand, structure=np.ones((3, 3)))
    if n_labels:
        slices = ndimage.find_objects(labels)
        strengths = ndimage.maximum(excess, labels,
                                    index=np.arange(1, n_labels + 1))
        sizes = ndimage.sum_labels(cand, labels,
                                   index=np.arange(1, n_labels + 1))
        for sl, strength, size in zip(slices, strengths, sizes):
            # small clusters are expected from the tail of the noise-floor
            # distribution (leakage correlates neighbouring bins, so even
            # pairs happen by chance); demand evidence scaled to the size
            need = peak_db + (8.0 if size < 2 else 4.0 if size < 4 else 0.0)
            if strength < need:
                continue
            ys, xs = sl
            fx0 = freq_x[xs.start] - pad * dfx
            fx1 = freq_x[xs.stop - 1] + pad * dfx
            fy0 = min(freq_y[ys.start], freq_y[ys.stop - 1]) - pad * dfy
            fy1 = max(freq_y[ys.start], freq_y[ys.stop - 1]) + pad * dfy
            cx, cy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
            wx, wy = fx1 - fx0, fy1 - fy0
            if abs(cx) <= wx / 2.0 and abs(cy) <= wy / 2.0:
                continue        # box covers the origin
            if (xs.stop - xs.start) <= 4 and (ys.stop - ys.start) <= 4:
                found.append((float(strength), "notch", (cx, cy)))
            else:
                found.append((float(strength), "rect", (cx, cy, wx, wy)))

    # -- keep one of each conjugate pair, strongest first ---------------
    found.sort(key=lambda t: -t[0])
    notches, rects, accepted = [], [], []
    tol = max(dfx, dfy)
    for strength, kind, item in found:
        if kind == "notch":
            cx, cy = item
            hx = hy = 2.0 * tol
        else:
            cx, cy, wx, wy = item
            hx, hy = wx / 2.0, wy / 2.0
        dup = any(
            (abs(cx + ax) <= hx + ahx + tol and abs(cy + ay) <= hy + ahy + tol)
            or (abs(cx - ax) <= hx + ahx + tol and abs(cy - ay) <= hy + ahy + tol)
            for ax, ay, ahx, ahy in accepted
        )
        if dup:
            continue
        accepted.append((cx, cy, hx, hy))
        if kind == "notch":
            notches.append(item)
        else:
            rects.append(item)
        if len(notches) + len(rects) >= max_items:
            break
    return notches, rects


# --- Utility and Loading Functions ---


def get_gwyddion_cmap():
    """
    The default Gwyddion colormap (black -> dark red -> yellow -> white).

    This used to be an approximation built here; it now comes from
    gwy_colormaps, which holds the stop tables of every Gwyddion gradient,
    so the colours are Gwyddion's own rather than a look-alike. Use
    gwy_colormaps.current() instead when the user's choice should be
    followed.

    Returns:
        matplotlib.colors.Colormap: The 'Gwyddion.net' gradient.
    """
    return gwy_colormaps.get(gwy_colormaps.DEFAULT)


def load_channel(filename, channel_name, fallback_to_height=False):
    """
    Loads a single channel from a Gwyddion (.gwy) file.

    Args:
        filename (str): Path to the .gwy file.
        channel_name (str): The exact name of the channel to load (e.g., "Height [Fwd]").
        fallback_to_height (bool): If True and channel_name is not found, it will
                                   try to load the first channel with "Height" in its name.

    Returns:
        GwyDataField or None: The loaded GwyDataField object, or None if not found.
    """
    print(f"Loading '{channel_name}' from {filename}...")
    try:
        all_channels = gwy_loader.load_gwy(filename)

        if channel_name in all_channels:
            return all_channels[channel_name]

        if fallback_to_height:
            height_keys = [k for k in all_channels.keys() if "Height" in k]
            if height_keys:
                print(
                    f"  > Channel '{channel_name}' not found. Using fallback '{height_keys[0]}'."
                )
                return all_channels[height_keys[0]]

        print(
            f"  > Error: Channel '{channel_name}' not found, and no suitable fallback was available."
        )
        return None

    except FileNotFoundError:
        print(f"  > Error: File not found - {filename}")
        return None
    except Exception as e:
        print(f"  > Error: Failed to process {filename}: {e}")
        return None


# --- Plotting Function ---


def plot_image(
    data,
    x_real,
    y_real,
    title="AFM Image",
    cmap="gray",
    cbar_label="units",
    spatial_units="units",
    vmin=None,
    vmax=None,
):
    """
    Displays 2D data as an image using matplotlib.

    Args:
        data (np.ndarray): The 2D numpy array to plot.
        x_real (float): The real-world width of the image (for axis scaling).
        y_real (float): The real-world height of the image (for axis scaling).
        title (str): The title for the plot.
        cmap (str or Colormap): The colormap to use for the image.
        cbar_label (str): The label for the colorbar.
        spatial_units (str): The units for the x and y axes (e.g., 'µm').
        vmin (float, optional): The minimum value for the color scale. Defaults to None.
        vmax (float, optional): The maximum value for the color scale. Defaults to None.
    """
    if data is None:
        print("Cannot plot, data is None.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    extent = (0, x_real, 0, y_real)

    im = ax.imshow(
        data,
        origin="upper",
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    ax.set_title(title)
    ax.set_xlabel(f"x ({spatial_units})")
    ax.set_ylabel(f"y ({spatial_units})")

    cbar = fig.colorbar(im, ax=ax, pad=0.05, fraction=0.046)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.show()


def plot_2d_fft(
    data,
    dx=1.0, 
    dy=1.0,
    title="2D FFT Magnitude",
    cmap="viridis",
    cbar_label="Magnitude (dB)",
    freq_units="1/units"
):
    """Plots the 2D FFT magnitude spectrum."""
    magnitude_spectrum, extent = get_2d_fft_magnitude(data, dx, dy)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        magnitude_spectrum,
        extent=extent,
        cmap=cmap,
        aspect="equal",
        origin="upper"
    )
    
    ax.set_title(title)
    ax.set_xlabel(f"Frequency X ({freq_units})")
    ax.set_ylabel(f"Frequency Y ({freq_units})")
    
    cbar = fig.colorbar(im, ax=ax, pad=0.05, fraction=0.046)
    cbar.set_label(cbar_label)
    
    plt.tight_layout()
    plt.show()


# --- Example Usage ---

if __name__ == "__main__":
    # This block demonstrates how to use the functions in this module.

    sample_file = "2023-12-01_16-05-17_G1_DDC_G2_DDC_6m_400m_0027_CALIBRATED.gwy"

    print("--- 1. Processing Height Channel ---")

    # Step 1: Load the desired channel
    height_field = load_channel(sample_file, "Height [Fwd]", fallback_to_height=True)

    if height_field:
        # Data is in meters. Calibration gain is 1.0.
        height_data = height_field.data.copy()

        # Step 2: Remove background tilt
        leveled_data = level_by_plane_fit(height_data)

        # Step 3: Remove scars (line defects)
        descarred_data = remove_scars(leveled_data, threshold=3.0, min_length=5)

        # Step 4: Set the baseline to zero
        final_height_data = set_baseline_to_zero(descarred_data)

        # Step 5: Convert units for plotting (e.g., to nanometers and micrometers)
        height_data_nm = final_height_data * 1e9
        x_real_um = height_field.xreal * 1e6
        y_real_um = height_field.yreal * 1e6

        # Step 6: Plot the processed height image
        print("\nPlotting processed Height data...")
        plot_image(
            data=height_data_nm,
            x_real=x_real_um,
            y_real=y_real_um,
            title="Processed Height (Leveled, Baseline at Zero)",
            cmap=get_gwyddion_cmap(),
            cbar_label="Height (nm)",
            spatial_units="µm",
        )
        
        # Step 7: Plot the 2D FFT
        dx_um = x_real_um / height_data.shape[1]
        dy_um = y_real_um / height_data.shape[0]
        
        print("\nPlotting 2D FFT of the Height data...")
        plot_2d_fft(
            data=final_height_data,
            dx=dx_um,
            dy=dy_um,
            title="2D FFT Magnitude (Height)",
            freq_units="1/µm"
        )

        # Step 8: Apply and plot a lowpass filter
        print("\nApplying lowpass FFT filter (cutoff = 10 1/µm)...")
        filtered_height = filter_by_2d_fft(final_height_data, cutoff_freq=10.0, mode='lowpass', dx=dx_um, dy=dy_um)
        
        plot_image(
            data=filtered_height * 1e9,
            x_real=x_real_um,
            y_real=y_real_um,
            title="Lowpass Filtered Height",
            cmap=get_gwyddion_cmap(),
            cbar_label="Height (nm)",
            spatial_units="µm",
        )

    print("\n" + "=" * 40 + "\n")

    print("--- 2. Processing Error Channel ---")

    # Step 1: Load the error channel
    error_field = load_channel(sample_file, "Error [Fwd]")

    if error_field:
        # Step 2: Extract data
        error_data = error_field.data.copy()

        # Step 3: Convert spatial units for plotting
        x_real_um = error_field.xreal * 1e6
        y_real_um = error_field.yreal * 1e6

        # Step 4: Plot the error image
        print("\nPlotting Error data...")
        plot_image(
            data=error_data,
            x_real=x_real_um,
            y_real=y_real_um,
            title="Error Signal",
            cmap="gray",
            cbar_label="Error",
            spatial_units="µm",
        )

    print("\nExample script finished.")