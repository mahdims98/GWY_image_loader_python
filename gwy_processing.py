import gwy_loader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


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


def _tukey_1d(n, alpha):
    """
    1D Tukey (tapered cosine) window without scipy.

    alpha is the fraction of the window inside the cosine taper:
    alpha=0 -> rectangular (no window), alpha=1 -> Hann window.
    """
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        return np.hanning(n)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return w
    t = np.arange(0, edge + 1)
    taper = 0.5 * (1 + np.cos(np.pi * (2.0 * t / (alpha * (n - 1)) - 1)))
    w[:edge + 1] = taper
    w[-(edge + 1):] = taper[::-1]
    return w


def make_fft_window(shape, window='hanning', alpha=0.5):
    """
    Builds a separable 2D window for FFT analysis.

    Args:
        shape (tuple): (ny, nx) shape of the image.
        window (str or None): 'hanning', 'tukey', or None/'none'.
        alpha (float): Taper fraction for the Tukey window (0..1).
                       alpha=0 is no tapering, alpha=1 equals Hann.
                       Ignored for 'hanning'.

    Returns:
        np.ndarray or None: The 2D window, or None if no windowing.
    """
    if window is None or window == 'none':
        return None
    ny, nx = shape
    if window == 'hanning':
        wy, wx = np.hanning(ny), np.hanning(nx)
    elif window == 'tukey':
        wy, wx = _tukey_1d(ny, alpha), _tukey_1d(nx, alpha)
    else:
        raise ValueError("Unknown window: '{}'".format(window))
    return np.sqrt(np.outer(wy, wx))


def get_2d_fft_magnitude(data, dx=1.0, dy=1.0, window=None, alpha=0.5):
    """
    Calculates the 2D FFT magnitude spectrum (in decibels) and frequency extents.

    Args:
        data (np.ndarray): 2D numpy array.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        window (str, optional): Windowing function: 'hanning', 'tukey' or None.
        alpha (float): Taper fraction for the Tukey window (0..1).
    """
    ny, nx = data.shape
    w2d = make_fft_window((ny, nx), window, alpha)
    if w2d is not None:
        data = data * w2d

    f = np.fft.fft2(data)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-8)

    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    extent = [freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]]
    return magnitude_spectrum, extent


def filter_by_2d_fft(data, cutoff_freq, mode='lowpass', dx=1.0, dy=1.0,
                     window=None, alpha=0.5):
    """
    Applies a basic 2D FFT lowpass or highpass filter.

    Args:
        data (np.ndarray): 2D numpy array.
        cutoff_freq (float): Cutoff frequency in the same inverse units as dx/dy.
        mode (str): 'lowpass' or 'highpass'.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        window (str, optional): Windowing function: 'hanning', 'tukey' or None.
        alpha (float): Taper fraction for the Tukey window (0..1).

    Returns:
        np.ndarray: The filtered data.
    """
    ny, nx = data.shape
    original_mean = data.mean()

    w2d = make_fft_window((ny, nx), window, alpha)
    if w2d is not None:
        data = data * w2d

    f = np.fft.fft2(data)
    fshift = np.fft.fftshift(f)

    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(freq_x, freq_y)
    F_dist = np.sqrt(FX**2 + FY**2)

    mask = F_dist <= cutoff_freq if mode == 'lowpass' else F_dist > cutoff_freq

    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))

    # Correct for amplitude loss from windowing and restore mean
    filtered_data = np.real(img_back)
    if w2d is not None:
        # The window reduces the total power. We need to compensate.
        # A simple approach for visualization is to rescale to the original mean.
        filtered_data = filtered_data * (1.0 / w2d.mean())
        filtered_data = filtered_data - filtered_data.mean() + original_mean

    return filtered_data


def filter_by_2d_fft_mask(data, mask, window=None):
    """
    Applies a user-defined mask in the frequency domain.
    This is useful for removing specific periodic noise (notch filtering).

    Args:
        data (np.ndarray): 2D numpy array.
        mask (np.ndarray): A 2D boolean or binary array of the same shape as data.
                           Frequencies where the mask is True (or 1) will be kept.
                           Frequencies where the mask is False (or 0) will be removed.
        window (str, optional): The windowing function to apply.
                                Currently supports 'hanning'. Defaults to None.
    Returns:
        np.ndarray: The filtered data.
    """
    # The logic for windowing, FFT, applying mask, and inverse FFT would be
    # very similar to `filter_by_2d_fft`. For brevity, the core step is shown.
    f = np.fft.fft2(data)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
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
    ny, nx = shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(freq_x, freq_y)

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
    ny, nx = shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(freq_x, freq_y)

    mask = np.ones(shape, dtype=bool)
    for c in x_bands:
        mask &= (np.abs(FX - c) > half_width) & (np.abs(FX + c) > half_width)
    for c in y_bands:
        mask &= (np.abs(FY - c) > half_width) & (np.abs(FY + c) > half_width)
    return mask


def detect_fft_peaks(data, dx=1.0, dy=1.0, protect_radius=0.0,
                     threshold_db=20.0, max_peaks=50, min_separation=None,
                     window='hanning', alpha=0.5):
    """
    Detects sharp peaks in the 2D FFT magnitude spectrum (periodic noise).

    A pixel is considered a peak candidate when its magnitude (in dB)
    exceeds the spectrum median by `threshold_db` AND it is a local maximum
    in its 3x3 neighbourhood AND it lies outside the central low-frequency
    region of radius `protect_radius`.

    Candidates are then accepted strongest-first, skipping any candidate
    closer than `min_separation` to an already accepted peak (or to its
    mirrored counterpart, since conjugate peaks come in +/- pairs).

    Args:
        data (np.ndarray): 2D image data.
        dx, dy (float): Pixel sizes.
        protect_radius (float): Frequencies within this radius of the origin
                                are never reported (protects the actual image
                                content around DC).
        threshold_db (float): Peak height above the spectrum median, in dB.
        max_peaks (int): Maximum number of peaks to return.
        min_separation (float): Minimum distance between reported peaks.
                                Defaults to ~3 frequency pixels.
        window (str, optional): Windowing before the FFT
                                ('hanning', 'tukey' or None).
        alpha (float): Taper fraction for the Tukey window (0..1).

    Returns:
        list of (fx, fy): Detected peak positions, strongest first. Only one
                          of each conjugate +/- pair is returned.
    """
    ny, nx = data.shape
    d = data
    w2d = make_fft_window((ny, nx), window, alpha)
    if w2d is not None:
        d = data * w2d

    fshift = np.fft.fftshift(np.fft.fft2(d))
    mag_db = 20 * np.log10(np.abs(fshift) + 1e-12)

    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(freq_x, freq_y)
    F_dist = np.sqrt(FX**2 + FY**2)

    background = np.median(mag_db)
    candidates = (mag_db > background + threshold_db) & (F_dist > protect_radius)

    # Local maximum in the 3x3 neighbourhood (edges wrap, which is harmless
    # here since the spectrum decays toward Nyquist).
    local_max = np.ones((ny, nx), dtype=bool)
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            if sx == 0 and sy == 0:
                continue
            local_max &= mag_db >= np.roll(np.roll(mag_db, sy, axis=0), sx, axis=1)

    peak_mask = candidates & local_max
    ys, xs = np.nonzero(peak_mask)
    if len(ys) == 0:
        return []

    if min_separation is None:
        dfx = freq_x[1] - freq_x[0] if nx > 1 else 1.0
        dfy = freq_y[1] - freq_y[0] if ny > 1 else 1.0
        min_separation = 3.0 * max(dfx, dfy)

    order = np.argsort(mag_db[ys, xs])[::-1]
    peaks = []
    for idx in order:
        fx_, fy_ = float(FX[ys[idx], xs[idx]]), float(FY[ys[idx], xs[idx]])
        too_close = any(
            (fx_ - px) ** 2 + (fy_ - py) ** 2 < min_separation**2
            or (fx_ + px) ** 2 + (fy_ + py) ** 2 < min_separation**2
            for px, py in peaks
        )
        if too_close:
            continue
        peaks.append((fx_, fy_))
        if len(peaks) >= max_peaks:
            break
    return peaks


# --- Utility and Loading Functions ---


def get_gwyddion_cmap():
    """
    Creates a custom colormap that approximates the default Gwyddion 'Gwy' style.
    (Black -> Red -> Yellow -> White)

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The custom colormap.
    """
    colors = (
        np.array(
            [
                (0, 0, 0),  # Black
                (168, 40, 15),  # Dark Red
                (243, 194, 93),  # Yellow
                (255, 255, 255),  # White
            ]
        )
        / 255
    )
    return mcolors.LinearSegmentedColormap.from_list("gwyddion_fake", colors)


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
    magnitude_spectrum, extent = get_2d_fft_magnitude(data, dx, dy, window='hanning')
    
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
        filtered_height = filter_by_2d_fft(final_height_data, cutoff_freq=10.0, mode='lowpass', dx=dx_um, dy=dy_um, window='hanning')
        
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