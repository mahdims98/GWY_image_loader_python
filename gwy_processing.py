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


def get_2d_fft_magnitude(data, dx=1.0, dy=1.0):
    """
    Calculates the 2D FFT magnitude spectrum (in decibels) and frequency extents.
    """
    f = np.fft.fft2(data)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-8)

    ny, nx = data.shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    extent = [freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]]
    return magnitude_spectrum, extent


def filter_by_2d_fft(data, cutoff_freq, mode='lowpass', dx=1.0, dy=1.0):
    """
    Applies a basic 2D FFT lowpass or highpass filter.
    
    Args:
        data (np.ndarray): 2D numpy array.
        cutoff_freq (float): Cutoff frequency in the same inverse units as dx/dy.
        mode (str): 'lowpass' or 'highpass'.
        dx (float): Pixel size in x.
        dy (float): Pixel size in y.
        
    Returns:
        np.ndarray: The filtered data.
    """
    ny, nx = data.shape
    f = np.fft.fft2(data)
    fshift = np.fft.fftshift(f)
    
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(freq_x, freq_y)
    F_dist = np.sqrt(FX**2 + FY**2)
    
    mask = F_dist <= cutoff_freq if mode == 'lowpass' else F_dist > cutoff_freq
        
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    return np.real(img_back)


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