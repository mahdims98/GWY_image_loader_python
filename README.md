# Gwyddion Python Tools

This directory contains Python scripts for natively loading, processing, and visualizing Gwyddion (`.gwy`) files, which are commonly used for Atomic Force Microscopy (AFM) and Scanning Probe Microscopy (SPM) data. 

These tools allow you to work with Gwyddion files directly in Python without needing to install the complex Gwyddion C libraries.

## Files Included

* **`gwy_loader.py`**: A pure Python module for parsing and reading `.gwy` files. It extracts data fields, metadata, physical dimensions, and SI units.
* **`gwy_processing.py`**: A toolkit for common AFM image processing tasks (such as plane leveling and scar removal) and plotting, utilizing `numpy` and `matplotlib`.
* **`gwy_twoway.py`**: Forward/backward (two-way) scan processing — scanner lag and hysteresis alignment, parachuting-artifact detection, and soft-min merging of the two scan directions.
* **`gwy_destripe.py`**: Stripe removal — the contourlet-domain Fourier method of Liang et al. (2016) and the variational method of Rottmayer et al. (2025).
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
* **Scar Removal** (`remove_scars`): Detects and interpolates horizontal line defects (strokes) introduced during scanning. For stripes that run the whole width of the scan rather than isolated strokes, see [Stripe Removal](#stripe-removal-gwy_destripepy) below.
* **Baseline Adjustment** (`set_baseline_to_zero`): Shifts the minimum data point to a base of zero.
* **Outlier Filtering** (`filter_by_percentile`): Clips extreme values (spikes) based on a designated percentile range. In the GUI the clip is *re-editable*: reopening the dialog right after a clip edits that same step, so the histogram still shows the full unclipped distribution and the limits can be widened again instead of only narrowed.
* **FFT Analysis & Filtering** (`get_2d_fft_magnitude`, `filter_by_2d_fft_mask`): 2D FFT analysis (no windowing - the displayed spectrum is exactly the one being filtered, normalized so the DC bin is the image mean) and frequency-domain filtering through a single mask that can combine a radial lowpass/highpass (`build_pass_mask`), circular notches (`build_notch_mask`), rectangular patches (`build_rect_mask`), and straight bands (`build_band_mask`). Noise is auto-detected systematically (`detect_fft_noise`) on the *excess* spectrum - the dB magnitude above the local radial background (`fft_excess_db`), so the falloff of the real topography never triggers it: streak columns/rows (median excess along the axis), coherent interference peaks sitting on the fx/fy axes, and off-axis regions each get their own statistically matched test. The whole mask can be given a smooth Gaussian roll-off (`smooth_fft_mask`) instead of hard edges. The GUI exposes all of these in one FFT-filter dialog with a large interactive spectrum (click to place cutoff/notches/bands, drag to notch a rectangle) and a *Zoom window* that shows the image before and after the filter side by side on one color scale, cropped to an area dragged on the result panel - so it can be checked close up that the filter took out the noise and not the topography.

### Stripe Removal (`gwy_destripe.py`)

Stripe artifacts are elongated, roughly parallel corruptions that share one
direction. In AFM they appear as scan-line offsets: a line, or a run of
lines, sits at the wrong height because the feedback settled badly, the tip
picked something up, or the drift jumped. The same artifact class is called
*curtaining* in FIB-SEM, *striping* in light-sheet microscopy and remote
sensing, and the literature is largely shared.

Two methods are implemented, both taken from the
[General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)
project [[5]](#stripe-references) and selected from the **Method** dropdown of
the GUI's *Stripe removal* window, which shows only the chosen method's
parameters. Both assume the same decomposition of the recorded image,

```
u0 = u + s          u = the clean image,  s = the stripes
```

and both take the stripe direction in degrees, `angle`, with **0° = horizontal
scan lines** (the usual AFM case) and 90° = vertical. What separates them is
*how they decide which part of the image is `s`*: **MDSR** answers in the
frequency domain — stripes are a narrow band of frequencies — while **GSR**
answers by optimization — stripes are whatever is sparse, elongated along the
given direction, and leaves a clean image behind.

#### MDSR — multidirectional stripe remover (Fourier filtering)

`mdsr` implements the method of Liang et al. [[1]](#stripe-references),
following the reference implementation
`Matlab-Stripe-Removal/Algorithms/MDSR.m` in [[5]](#stripe-references). The
paper's three steps are:

**1. Decompose.** The image is split by a *nonsubsampled contourlet
transform* (NSCT, Cunha et al. [[4]](#stripe-references)) into subbands of
different scale and direction. The NSCT has two stages: a nonsubsampled
pyramid, which separates octave-wide bands of spatial frequency (scale), and
a nonsubsampled directional filter bank, which cuts each of those into
angular wedges (direction). "Nonsubsampled" means nothing is decimated, so
the transform is shift-invariant — an artifact does not change character
depending on where it happens to sit, and the pseudo-Gibbs ringing of
decimated wavelet transforms is largely avoided. The paper uses 5 scales × 8
directions.

**2. Damp.** Stripes running in direction θ₀ are nearly constant along
themselves and vary across, so their energy in the Fourier plane collapses
onto the line through the origin *perpendicular* to θ₀. Every high-pass
subband is multiplied by a groove that is zero on that line and rises to one
away from it:

```
w(f) = 1 − exp( −t² / (2·σ̄ᵢ²) ),      t = f·(cos θ₀, sin θ₀)
σ̄ᵢ = σ · exp( −θᵢ² / (2·σ_a²) )
```

`t` is the frequency component *along* the stripes — the coordinate that
crosses the ridge of stripe energy — and `θᵢ` is the angle between subband
*i*'s own orientation and the stripes. So the subband aligned with the
stripes gets the full groove `σ`, and subbands pointing elsewhere get a
groove that narrows exponentially and barely touches them. Following
Rottmayer et al.'s **MDSR+** variant [[2]](#stripe-references), subbands
deviating by more than `max_angle` are skipped entirely, which they show
reduces artifacting. The low-pass residual is never filtered.

**3. Reconstruct** by summing the subbands back together.

**How it is implemented here.** Both NSCT stages are nonsubsampled, i.e.
plain linear shift-invariant filters, and the damping is a multiplication in
the frequency domain. The whole method is therefore *one linear filter*, and
it is implemented as one: the analysis filters, the dampings and the
synthesis filters are accumulated into a single frequency mask (`mdsr_mask`)
applied in one FFT round trip. This is exact — not an approximation of the
subband loop — and it is why the preview is instant where the paper reports
seconds per image. The GUI shows that mask, which is the most direct picture
of what the method does.

The filter bank itself is built directly in the frequency domain, as
raised-cosine rings and angular wedges that sum to exactly one (so
reconstruction is perfect), rather than with the `maxflat`/`dmaxflat7` filter
banks of Cunha's NSCT toolbox that the reference implementation calls. The
structure of the method, the damping equations and the parameters are the
paper's; the transfer functions of the filter bank are not bit-identical.

One correction: Eq. (1) of [[1]](#stripe-references) writes the damping
coordinate as `u·cos(π/2+θ₀) + v·sin(π/2+θ₀)`, which places the zero of the
groove on the line *perpendicular* to the stripe frequencies — the wrong
axis. The surrounding text ("the bottom is the line with angle π/2 + θ₀ …
where the value of w is 0") and the reference code both place it on the
stripe frequencies. The latter is what is implemented.

Parameters:

| Parameter | Meaning |
|---|---|
| `angle` | Stripe direction, 0° = horizontal scan lines. |
| `sigma` | Width of the damping groove, **in frequency bins** (as in the reference). The groove reaches about 2.5·σ bins, so it removes stripe-parallel structure longer than roughly `nx/(2.5σ)` pixels. |
| `levels` | Number of scales. The low-pass residual is never filtered, so raise this until no stripes are left in it. |
| `directions` | Angular wedges per scale (a power of two; 8 is the paper's choice). |
| `sigma_a` | How fast the groove narrows with angular deviation, in radians. 0.3 in the reference, 0.8 in the paper; raise it only if the stripe direction is uncertain. |
| `max_angle` | Deviation beyond which a subband is left alone (the MDSR+ restriction). |
| `pad` | Mirror the image before filtering, to keep the FFT's periodic wrap-around from ringing along the notch. Off by default, which is the reference behaviour. |

`sigma` is the only parameter that normally needs tuning, and it trades
stripe removal against real structure that is itself elongated along the scan
lines. The reference recommends 5–25 for the ~1000 px light-sheet images it
was tuned on; the same bin count is a 2–4× wider notch on a 512 px AFM scan,
hence the lower default here. The dialog prints the equivalent length scale
next to the mask.

#### GSR — general stripe remover (variational)

`gsr` implements the method of Rottmayer, Redenbach and Fahrbach
[[2]](#stripe-references), ported from that project's
`Python-Stripe-Removal/GeneralStripeRemover.py` (PyTorch) to numpy for the 2D
case. Instead of deciding in advance which frequencies are stripes, it states
what a clean image and a stripe image *look like* and lets an optimizer find
the split. It minimizes, over all splits with `u + s = u0`,

```
μ1·‖∇u‖₂,₁  +  ι[0,1](u)  +  ‖∇_θ s‖₁  +  μ2·‖s‖₁
```

Term by term:

* `μ1·‖∇u‖₂,₁` — the total variation of the clean image. It says a clean
  image is made of smooth regions separated by few strong edges, so anything
  that adds gradient without adding structure is pushed out of `u`.
* `ι[0,1](u)` — an indicator that is 0 inside the value range and ∞ outside,
  keeping the clean image within the range of the input. No brightness
  correction is needed afterwards.
* `‖∇_θ s‖₁` — the derivative of the stripe image *along* the stripe
  direction. Penalizing it forces `s` to be nearly constant along θ, i.e.
  actually stripe-shaped.
* `μ2·‖s‖₁` — sparsity of the stripe image: only a small part of the image is
  struck by artifacts, so `s` should be zero nearly everywhere.

The objective builds on the directional-difference model of Fitschen et al.
[[6]](#stripe-references); a similar functional was explored by Liu et al.
[[7]](#stripe-references) for oblique stripes in remote sensing.

**How it is solved.** With the primal–dual hybrid gradient method with
extrapolation of the dual variable (PDHGMp; Chambolle & Pock
[[8]](#stripe-references), Burger et al. [[9]](#stripe-references)). Each of
the three non-smooth penalties gets a dual variable, and one iteration is:

```
u ← u − τσ·divergence(b₁)                 descend using the current duals
s ← s − τσ·(D_θᵀ b₂ + b₃)
t ← ½(u0 − u − s);  u ← u + t;  s ← s + t   back onto the constraint u + s = u0
clip u into [0,1], moving what was clipped into s
b₁ ← project(b₁ + ∇u,   onto the disc of radius μ1/σ)     dual of the TV term
b₂ ← clip(b₂ + D_θ s,   ±1/σ)                             dual of ‖∇_θ s‖₁
b₃ ← clip(b₃ + s,       ±μ2/σ)                            dual of μ2‖s‖₁
b̄ ← 2b − b_old                            extrapolation of the duals
```

with step sizes τ = σ = 0.35 (this σ is the optimizer's step size, unrelated
to MDSR's damping width). The sequence provably converges to the minimizer
[[8]](#stripe-references); in practice the result stops moving well before
the recommended iteration count.

`D_θ` is a forward difference along a whole-pixel step. Steps of 0°, 26.6°
(2:1) and 45° (1:1) are supported, and every flip and transpose of them, so
an arbitrary `angle` snaps to the nearest — that is the reference
implementation's limitation, not the port's.

Parameters:

| Parameter | Meaning |
|---|---|
| `angle` | Stripe direction, 0° = horizontal scan lines. |
| `mu1` | Strength of the removal. Larger removes more, but starts to smooth and to eat structures that look like stripes. |
| `mu2` | Caution about touching real structure: larger keeps `s` sparser, so less is removed. |
| `iterations` | Primal–dual steps. |

The paper's defaults are `μ1 = 1/3` and `μ2 = 1/300`, with the intervals
`μ1 ∈ [0.1, 0.5]` and `μ2 ∈ [0.0016, 0.017]` never exceeded across all their
data; the *ratio* matters more than either value alone, and scaling both
together adjusts how far the method strays from ideal stripes. Their
supplement recommends 10 000 iterations for a fully converged result and
notes 5 000 usually suffice; this port runs at about 1.8 ms/iteration on a
512×256 scan (10 000 ≈ 18 s), so the default is lower to keep the preview
interactive — raise it before applying.

Because the indicator constrains `u` to [0, 1], the image is normalized to
that range internally and mapped back afterwards. AFM heights in nanometres
would otherwise be clipped away entirely, and this way the published
parameters transfer unchanged.

**Parameter sweep.** Since the two gains are best judged by eye and against
each other, the GSR mode has a *Parameter sweep...* button that runs a grid
of pairs and shows the results side by side — 3×3 by default, rows stepping
`mu1` and columns stepping `mu2`. Both axes use the *same* step factor around
the values currently in the dialog, so with factor 2 the rows are `mu1/2,
mu1, 2·mu1` and the columns `mu2/2, mu2, 2·mu2`; the diagonal is then the
"scale both together" direction the paper describes. Every cell is computed
on the whole image and only then cropped, so the zoom area picked in the
dialog (drag on the result panel, then *Zoom area only*) changes what you see
and never what is computed. All cells share one color scale, the panels are
linked for toolbar zooming, and clicking a cell copies its pair back into the
dialog.

#### Choosing between them, and two shared caveats

MDSR is a single linear filter: fast, predictable, and easy to reason about
because the mask is visible. It suits regular stripes of roughly constant
width. Its cost is intrinsic — it removes *everything* inside a frequency
band, including real structure elongated along the scan lines.

GSR is an optimization and adapts to the image, which is why
[[2]](#stripe-references) reports it outperforming both MDSR+ and the
variational VSNR [[3]](#stripe-references) on light-sheet, FIB-SEM and remote
sensing data (PSNR, MS-SSIM, curtaining metric, line profiles), particularly
on irregular stripes of varying width and on short trails. It is the slower
of the two here, and its own documented limitation is that image structures
aligned with the stripe direction get reduced along with the stripes.

Two things apply to both:

* **Level the image first.** A plane tilt across the slow axis *is* a set of
  line offsets, and neither method can tell it from an artifact — it will be
  removed along with the stripes.
* **Per-line offsets always go.** For MDSR this is exact: the zero-frequency
  line along the stripes is damped to zero at any σ (the overall mean height
  is preserved separately). This is the part of the topography that is
  genuinely indistinguishable from a stripe, so if your surface really has
  height information that is constant along each scan line, destriping will
  take some of it.

<a name="stripe-references"></a>

**References**

1. X. Liang, Y. Zang, D. Dong, L. Zhang, M. Fang, X. Yang, A. Arranz,
   J. Ripoll, H. Hui and J. Tian, "Stripe artifact elimination based on
   nonsubsampled contourlet transform for light sheet fluorescence
   microscopy", *J. Biomed. Opt.* **21**(10), 106005 (2016).
   [doi:10.1117/1.JBO.21.10.106005](https://doi.org/10.1117/1.JBO.21.10.106005)
2. N. Rottmayer, C. Redenbach and F. O. Fahrbach, "A universal and effective
   variational method for destriping: application to light-sheet microscopy,
   FIB-SEM, and remote sensing images", *Opt. Express* **33**(3), 5800
   (2025). [doi:10.1364/OE.542868](https://doi.org/10.1364/OE.542868)
   (with Supplement 1, [doi:10.6084/m9.figshare.28022984](https://doi.org/10.6084/m9.figshare.28022984),
   which contains the parameter guidance and the MDSR+ definition)
3. J. Fehrenbach, P. Weiss and C. Lorenzo, "Variational algorithms to remove
   stationary noise: applications to microscopy imaging", *IEEE Trans. Image
   Process.* **21**(10), 4420–4430 (2012). — VSNR, the variational method
   both papers above compare against.
4. A. L. da Cunha, J. Zhou and M. N. Do, "The nonsubsampled contourlet
   transform: theory, design, and applications", *IEEE Trans. Image Process.*
   **15**(10), 3089–3101 (2006). — the transform MDSR decomposes with.
5. N. Rottmayer, "General stripe removal",
   <https://github.com/NiklasRottmayer/General-Stripe-Removal> (2024). — the
   reference implementations of both methods.
6. J. H. Fitschen, J. Ma and S. Schuff, "Removal of curtaining effects by a
   variational model with directional forward differences", *Computer Vision
   and Image Understanding* **155**, 24–32 (2017). — the model GSR builds on.
7. X. Liu, X. Lu, H. Shen et al., "Oblique stripe removal in remote sensing
   images via oriented variation", arXiv (2018).
8. A. Chambolle and T. Pock, "A first-order primal-dual algorithm for convex
   problems with applications to imaging", *J. Math. Imaging Vis.* **40**(1),
   120–145 (2011). — the optimizer.
9. M. Burger, A. Sawatzky and G. Steidl, "First order algorithms in
   variational image processing", in *Splitting Methods in Communication,
   Imaging, Science, and Engineering* (Springer, 2016), pp. 345–407.
10. B. Münch, P. Trtik, F. Marone and M. Stampanoni, "Stripe and ring artifact
    removal with combined wavelet — Fourier filtering", *Opt. Express*
    **17**(10), 8567–8591 (2009). — the wavelet-Fourier ancestor of MDSR's
    damping step.

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