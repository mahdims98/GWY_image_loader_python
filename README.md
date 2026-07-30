# Gwyddion Python Tools

This directory contains Python scripts for natively loading, processing, and visualizing Gwyddion (`.gwy`) files, which are commonly used for Atomic Force Microscopy (AFM) and Scanning Probe Microscopy (SPM) data. 

These tools allow you to work with Gwyddion files directly in Python without needing to install the complex Gwyddion C libraries.

## Files Included

* **`gwy_loader.py`**: A pure Python module for parsing and reading `.gwy` files. It extracts data fields, metadata, physical dimensions, and SI units.
* **`gwy_processing.py`**: A toolkit for common AFM image processing tasks (such as plane leveling and scar removal) and plotting, utilizing `numpy` and `matplotlib`.
* **`gwy_twoway.py`**: Forward/backward (two-way) scan processing — scanner lag and hysteresis alignment, parachuting-artifact detection, and soft-min merging of the two scan directions.
* **`gwy_processor_gui.py`**: An interactive Tkinter front-end that exposes every processing step in its own dialog with live previews, an undo stack, a processing log, and batch folder processing.

## Requirements

* Python 3.x
* `numpy`
* `scipy`
* `matplotlib`
* `six`

`gwy_twoway.py` also needs `hysteresis_compensation.py` (the power-law hysteresis
fit) when the `model` or `model_scaled` shift models are used. It is located
automatically in the same folder, in a `hysteresis/` subfolder, or in a sibling
`Hysteresis compensation Python/` folder; set the `GWY_HYSTERESIS_PATH`
environment variable to point elsewhere.

## Key Features

### Loading & Metadata (`gwy_loader.py`)
* **Load Channels**: Directly access data fields like `"Height [Fwd]"` or `"Error [Fwd]"`.
* **Extract Metadata**: Read physical properties, resolutions, offsets, and extract embedded metadata dictionaries.

### Image Processing (`gwy_processing.py`)
* **Plane Leveling** (`level_by_plane_fit`): Subtracts a fitted background plane to remove large-scale sample tilt.
* **Scar Removal** (`remove_scars`): Detects and interpolates horizontal line defects (strokes) introduced during scanning.
* **Baseline Adjustment** (`set_baseline_to_zero`): Shifts the minimum data point to a base of zero.
* **Outlier Filtering** (`filter_by_percentile`): Clips extreme values (spikes) based on a designated percentile range.
* **FFT Analysis & Filtering** (`get_2d_fft_magnitude`, `filter_by_2d_fft_mask`): 2D FFT analysis (no windowing - the displayed spectrum is exactly the one being filtered, normalized so the DC bin is the image mean) and frequency-domain filtering through a single boolean mask that can combine a radial lowpass/highpass (`build_pass_mask`), circular notches (`build_notch_mask`, auto-detectable with `detect_fft_peaks`), and straight bands (`build_band_mask`). The GUI exposes all of these in one FFT-filter dialog.

### Two-Way Scan Processing (`gwy_twoway.py`)

Implements the method of Kubo, Umeda, Kodera & Takada, *Biophysics and
Physicobiology* **20**, e200006 (2023), "Removing the parachuting artifact using
two-way scanning data", in three independently tunable stages:

* **Alignment** (`align_two_way`): the forward and backward images do not sample
  the same x positions, because of piezo hysteresis and of the feedback lag of
  the scanner. The shift is measured directly by block-wise cross-correlation on
  leveled data (`measure_shift_profile`; `match_level` chooses a plane fit
  or a per-row polynomial of degree `match_poly_order` - a matching aid only,
  never applied to the output data) and then regularized. Choose
  the model with `mapping`:
  * `xcorr` (default) — polynomial fit to the measured shift. Degree 0 is a pure
    constant lag, degree 2 adds the hysteresis bow.
  * `model_scaled` — the power-law hysteresis model supplies the bow *shape*,
    the data supply the lag and amplitude.
  * `model` — the power-law hysteresis model exactly as fitted. It is pinned to
    zero shift at both scan ends, so it describes pure hysteresis and **cannot**
    represent a constant feedback lag; use it only when the lag is known to be
    negligible.
  * `measured`, `none`.
* **Parachuting detection** (`difference_histogram`, `detect_parachuting`): flags
  pixels where the surface fell away faster along the scan line than the tip can
  follow. The decision line `dz = -(slope*delta + offset)` is chosen from the
  `H(delta, dz)` histogram, as in the paper. A symmetric histogram with no sharp
  lower border means there is no parachuting to remove.
* **Background correction** (`pre_plane`, `pre_rows` + `pre_rows_order`, both
  off by default): optionally remove each scan's own fitted plane and/or
  per-row polynomial background *before* the hysteresis is found and the scans
  are merged. Unlike `match_level` (a matching aid that never touches the
  output) this changes the data: it puts both scans on one common zero
  background, so a pixel replaced from the opposite scan sits flush with its
  neighbours instead of jumping by the background difference.
* **Merge** (`combine_scans`, `merge_two_way`): flagged pixels are replaced by
  the opposite scan; elsewhere the two scans are combined according to
  `combine`:
  * `average` (default) — weighted average `weight*fwd + (1-weight)*bwd`.
    `0.5` is the plain mean (lowest noise, no bias), `1` keeps the forward scan
    and `0` the backward one.
  * `slope` — slope-directional blend: the tip tracks rising edges well and
    lags on falling ones, and the two scans move in opposite directions, so at
    each pixel the scan that was climbing there is trusted more
    (`slope_gain` sets how sharply the weight switches). Sharpens edges
    compared to a plain average.
  * `consensus` — outlier rejection: keep whichever scan is closer to the
    local (`consensus_size` box) mean of the two. Rejects single-scan
    glitches.
  * `correlation` (`correlation_select`) — correlation-gated merge: the local
    windowed correlation between the two scans (`corr_window`) marks each
    pixel as shared (`corr >= corr_margin` → combined with `corr_combine`,
    which can be any of the modes above, e.g. `average` or `softmin`) or
    disputed.
    Disputed pixels are refereed by the phase and error channels of the same
    file (warped with the same alignment as the heights): each direction's
    height is locally correlated against the direction-averaged phase/error
    pattern and the direction with the highest |correlation| keeps the pixel.
    The dialog's Overlay selector gains `corr map` and `decision` views for
    tuning the margin.
  * `stripes` (`stripe_select`, `detect_line_artifacts`) — stripe/scratch-gated
    merge: line artifacts are detected independently in each scan from the
    vertical neighbours (a pixel that juts out from BOTH the row above and the
    row below by more than `stripe_thresh` robust sigmas, in runs of at least
    `stripe_min_len` px along the line — partial-line segments count). Clean
    pixels are combined with `corr_combine`; where exactly one scan is striped
    the clean scan gets weight `stripe_pref` (1 = taken outright). The Overlay
    selector's `stripes` view shows the detected artifacts per scan and
    `decision` shows the outcome.
  * `softmin` — the paper's corrected soft-minimum with parameter `beta`;
    `beta = 0` degenerates to the plain mean, large `beta` to a hard minimum.
  * `min`, `max`, `forward`, `backward`.
* **Crop** (`crop=True`, default): the lag/hysteresis shift means the first or
  last few columns were only ever imaged in one direction; those columns are
  trimmed off, so the merged image is slightly narrower than the input and its
  physical extent shrinks accordingly.

The GUI exposes this in two windows:

* **Two-way merge (Fwd/Bwd)** — alignment and merging. Shows the forward and
  backward images, an overlay of the two (opacity blend with an adjustable
  backward opacity, or a red/cyan anaglyph in which residual misalignment
  appears as color fringes), the hysteresis/lag curves in both directions
  (fwd→bwd and bwd→fwd, with the cropped edge region shaded), and the final
  merged, cropped image.
* **Parachuting removal (Fwd/Bwd)** — detection and repair. Shows the
  `H(delta, dz)` histograms of *both* scan directions with the decision line
  drawn on top, the flagged-pixel map, and the repaired result.

Both windows offer two ways to commit the result:

* **Merge to new channel** adds the merged image as a new channel (e.g.
  `Height [Merged]`), leaves the forward and backward channels untouched, and
  switches editing to it, so the rest of the processing chain runs on the merged
  data. The merge is recorded as the first pipeline step, so *Batch process
  folder* reproduces it on every file (re-measuring the shift for each one).
* **Replace current image** overwrites the image being edited.

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

### Merging a forward/backward pair

```python
import numpy as np
from gwy_loader import load_gwy
import gwy_twoway as tw

channels = load_gwy('sample_scan.gwy')
fwd = channels['Height [Fwd]'].data * 1e9      # nm
bwd = channels['Height [Bwd]'].data * 1e9

result = tw.process_two_way(fwd, bwd, mapping='xcorr', poly_order=2,
                            detect=False, combine='average', weight=0.5)

a = result.alignment
print(f"lag {a.lag_px:+.2f} px, bow {a.bow_px:.2f} px")
print(f"forward/backward rms difference {a.rms_before:.2f} -> {a.rms_after:.2f} nm")

merged = result.merged     # feed this into the normal processing chain
```

Inspect `tw.difference_histogram(fwd)` before enabling `detect=True`: pick the
decision slope from where the histogram shows a sharp lower border, and leave
detection off if it does not.

### Running the GUI

```
python gwy_processor_gui.py
```