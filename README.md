# Gwyddion Python Tools

This directory contains Python scripts for natively loading, processing, and visualizing Gwyddion (`.gwy`) files, which are commonly used for Atomic Force Microscopy (AFM) and Scanning Probe Microscopy (SPM) data. 

These tools allow you to work with Gwyddion files directly in Python without needing to install the complex Gwyddion C libraries.

## Files Included

* **`gwy_loader.py`**: A pure Python module for parsing and reading `.gwy` files. It extracts data fields, metadata, physical dimensions, and SI units.
* **`gwy_processing.py`**: A toolkit for common AFM image processing tasks (such as plane leveling and scar removal) and plotting, utilizing `numpy` and `matplotlib`.

## Requirements

* Python 3.x
* `numpy`
* `matplotlib`
* `six`

## Key Features

### Loading & Metadata (`gwy_loader.py`)
* **Load Channels**: Directly access data fields like `"Height [Fwd]"` or `"Error [Fwd]"`.
* **Extract Metadata**: Read physical properties, resolutions, offsets, and extract embedded metadata dictionaries.

### Image Processing (`gwy_processing.py`)
* **Plane Leveling** (`level_by_plane_fit`): Subtracts a fitted background plane to remove large-scale sample tilt.
* **Scar Removal** (`remove_scars`): Detects and interpolates horizontal line defects (strokes) introduced during scanning.
* **Baseline Adjustment** (`set_baseline_to_zero`): Shifts the minimum data point to a base of zero.
* **Outlier Filtering** (`filter_by_percentile`): Clips extreme values (spikes) based on a designated percentile range.
* **FFT Analysis & Filtering** (`get_2d_fft_magnitude`, `filter_by_2d_fft`): Performs 2D Fast Fourier Transform analysis and applies lowpass or highpass frequency filters.

### Visualization
* Offers a custom colormap approximating the default Gwyddion "Gwy" palette (`get_gwyddion_cmap`).
* Convenient wrapper functions for drawing plots with real-world scaled dimensions (`plot_image`, `plot_2d_fft`).

## Quick Start Example

```python
from gwy_loader import load_gwy
from gwy_processing import load_channel, level_by_plane_fit, remove_scars, set_baseline_to_zero, plot_image, get_gwyddion_cmap

filename = 'sample_scan.gwy'

# 1. Load the Height channel
channel = load_channel(filename, "Height [Fwd]", fallback_to_height=True)

if channel:
    data = channel.data.copy()
    
    # 2. Process the data
    leveled = level_by_plane_fit(data)
    descarred = remove_scars(leveled, threshold=3.0, min_length=5)
    final_data = set_baseline_to_zero(descarred)
    
    # 3. Convert units (e.g., meters to nanometers/micrometers)
    data_nm = final_data * 1e9
    x_um = channel.xreal * 1e6
    y_um = channel.yreal * 1e6
    
    # 4. Plot
    plot_image(
        data=data_nm, x_real=x_um, y_real=y_um,
        title="Processed AFM Height", cmap=get_gwyddion_cmap(),
        cbar_label="Height (nm)", spatial_units="µm"
    )
```