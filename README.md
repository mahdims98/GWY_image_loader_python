# Gwyddion Python Tools

This directory contains Python scripts for natively loading, processing, and visualizing Gwyddion (`.gwy`) files, which are commonly used for Atomic Force Microscopy (AFM) and Scanning Probe Microscopy (SPM) data. 

These tools allow you to work with Gwyddion files directly in Python without needing to install the complex Gwyddion C libraries.

## Files Included

* **`gwy_loader.py`**: A pure Python module for parsing and reading `.gwy` files. It extracts data fields, metadata, physical dimensions, and SI units.
* **`gwy_processing.py`**: A toolkit for common AFM image processing tasks (such as plane leveling and scar removal) and plotting, utilizing `numpy` and `matplotlib`.
* **`gwy_twoway.py`**: Forward/backward (two-way) scan processing — scanner lag and hysteresis alignment, parachuting-artifact detection, and soft-min merging of the two scan directions.
* **`gwy_destripe.py`**: Multidirectional stripe removal (MDSR) — the contourlet-domain destriping method of Liang et al. (2016).
* **`gwy_processor_gui.py`**: An interactive Tkinter front-end that exposes every processing step in its own dialog with live previews, undo/redo, a processing log, and batch folder processing.

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
* **Scar Removal** (`remove_scars`): Detects and interpolates horizontal line defects (strokes) introduced during scanning. For stripes that run the whole width of the scan rather than isolated strokes, see the MDSR destriper below.
* **Baseline Adjustment** (`set_baseline_to_zero`): Shifts the minimum data point to a base of zero.
* **Outlier Filtering** (`filter_by_percentile`): Clips extreme values (spikes) based on a designated percentile range. In the GUI the clip is *re-editable*: reopening the dialog right after a clip edits that same step, so the histogram still shows the full unclipped distribution and the limits can be widened again instead of only narrowed.
* **FFT Analysis & Filtering** (`get_2d_fft_magnitude`, `filter_by_2d_fft_mask`): 2D FFT analysis (no windowing - the displayed spectrum is exactly the one being filtered, normalized so the DC bin is the image mean) and frequency-domain filtering through a single mask that can combine a radial lowpass/highpass (`build_pass_mask`), circular notches (`build_notch_mask`), rectangular patches (`build_rect_mask`), and straight bands (`build_band_mask`). Noise is auto-detected systematically (`detect_fft_noise`) on the *excess* spectrum - the dB magnitude above the local radial background (`fft_excess_db`), so the falloff of the real topography never triggers it: streak columns/rows (median excess along the axis), coherent interference peaks sitting on the fx/fy axes, and off-axis regions each get their own statistically matched test. The whole mask can be given a smooth Gaussian roll-off (`smooth_fft_mask`) instead of hard edges. The GUI exposes all of these in one FFT-filter dialog with a large interactive spectrum (click to place cutoff/notches/bands, drag to notch a rectangle) and a *Zoom window* that shows the image before and after the filter side by side on one color scale, cropped to an area dragged on the result panel - so it can be checked close up that the filter took out the noise and not the topography.

### Stripe Removal (`gwy_destripe.py`)

`mdsr` implements the **multidirectional stripe remover** of X. Liang et al.,
*"Stripe artifact elimination based on nonsubsampled contourlet transform for
light sheet fluorescence microscopy"*, J. Biomed. Opt. **21**(10), 106005
(2016), following the reference implementation in
[General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)
(`Matlab-Stripe-Removal/Algorithms/MDSR.m`). The image is decomposed into
shift-invariant subbands of different scale and direction (a nonsubsampled
contourlet transform), the frequencies carrying stripes of the given
direction are damped in every high-pass subband — with a groove that narrows
as the subband's own orientation moves away from the stripes — and the image
is reconstructed.

Both stages of the NSCT are nonsubsampled, i.e. plain linear shift-invariant
filters, and the damping is a frequency-domain multiplication, so the whole
method is a single linear filter. It is implemented as one: the analysis
filters, dampings and synthesis filters are accumulated into one frequency
mask (`mdsr_mask`) applied in a single FFT round trip. That is exact, not an
approximation of the subband loop, and it makes the filter fast enough for a
live preview. The filter bank itself is built directly in the frequency
domain — raised-cosine rings and angular wedges that sum to exactly one, so
reconstruction is perfect — instead of the `maxflat`/`dmaxflat7` filter banks
of Cunha's NSCT toolbox that the reference implementation calls; the method,
the damping equations and the parameters are the paper's, the filter-bank
transfer functions are not bit-identical.

Parameters (`angle`, `sigma`, `directions`, `levels`, `sigma_a`, `max_angle`):

* `angle` — direction of the stripes, 0° = horizontal scan lines (the usual
  AFM artifact), 90° = vertical.
* `sigma` — width of the damping groove **in frequency bins**, as in the
  reference implementation. It trades stripe removal against real structure
  that is itself elongated along the scan lines: the groove reaches about
  2.5·σ bins, so it removes stripe-parallel features longer than roughly
  `nx/(2.5σ)` pixels. The reference recommends 5–25 for ~1000 px light-sheet
  images; on a 512 px AFM scan that is aggressive, hence the lower default.
* `levels` — number of scales. The low-pass residual is never filtered, so
  raise this until no stripes are left in it.
* `directions`, `sigma_a`, `max_angle` — subbands per scale, how fast the
  damping narrows with angular deviation, and the deviation beyond which a
  subband is left alone.

Two things follow from the method itself and are worth knowing: the exact
zero-frequency line along the stripes is damped to zero at any σ, so MDSR
always removes the per-line offsets (the overall mean height is kept); and a
plane tilt across the slow axis is itself a set of line offsets, so **level
the image before destriping** or the filter will eat the tilt.

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

### Saving from the GUI

* **Save processed image...** writes an annotated PNG (axes, colorbar, scale bar)
  plus a bare one-pixel-per-datapoint copy in a `pure/` subfolder, or a `.npy`
  array.
* **Save channel to .gwy...** writes the processed channel *together with every
  other channel of the loaded image*, so the output is a complete copy of the
  measurement plus the result. It defaults to the source file name with a
  `_processed` suffix, in the source folder. Saving into an existing file
  appends: the originals already in it are not written twice, and a repeated
  save of the same channel gets a numbered title (`... (processed) 2`).

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