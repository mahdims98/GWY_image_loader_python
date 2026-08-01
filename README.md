# Gwyddion Python Tools

This directory contains Python scripts for natively loading, processing, and visualizing Gwyddion (`.gwy`) files, which are commonly used for Atomic Force Microscopy (AFM) and Scanning Probe Microscopy (SPM) data. 

These tools allow you to work with Gwyddion files directly in Python without needing to install the complex Gwyddion C libraries.

## Files Included

* **`gwy_loader.py`**: A pure Python module for parsing and reading `.gwy` files. It extracts data fields, metadata, physical dimensions, and SI units.
* **`gwy_processing.py`**: A toolkit for common AFM image processing tasks (such as plane leveling and scar removal) and plotting, utilizing `numpy` and `matplotlib`.
* **`gwy_flatten.py`**: Background subtraction that leaves the sample out of the fit — the cells, bubbles or pits are segmented and excluded first, so the polynomial cannot bend itself around them and leave trenches and uneven surfaces behind. After Wang et al. (2018), with the paper's sliding-window fit in closed form; the fitting direction can be measured from the scan and areas can be excluded by hand, after Zhang et al. (2026).
* **`gwy_twoway.py`**: Forward/backward (two-way) scan processing — scanner lag and hysteresis alignment, parachuting-artifact detection, and soft-min merging of the two scan directions.
* **`gwy_destripe.py`**: Stripe removal — the contourlet-domain Fourier method of Liang et al. (2016), the variational method of Rottmayer et al. (2025) and the spectrum-denoising method of Chen & Pellequer (2011).
* **`gwy_colormaps.py`**: Gwyddion's false-colour gradients (`Gray`, `Sky`, `Body`, `Rainbow2`, `Viridis`, ... 60 in all) as matplotlib colormaps, plus the application-wide selection used by the GUI.
* **`gwy_balance.py`**: A colour range a whole folder can share — segments each image into cells and substrate, measures both, and anchors the range on those two places instead of on the image as a whole, so the substrate reads the same colour in every image without flattening real height differences between them. Offsets only; the z scale is never touched.
* **`gwy_processor_gui.py`**: An interactive Tkinter front-end that exposes every processing step in its own dialog with live previews, undo/redo, a processing log, a folder quick view, a balanced folder view, and batch folder processing.

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

### Smart background (`gwy_flatten.py`)

Levelling means fitting the artifact — tilt, bow, drift, the slow wander of the
z scanner — and subtracting it. The fit is the whole problem. Fit it to *every*
pixel, as `level_by_plane_fit` or `align_rows` does, and the sample is fitted
too: a scan line that crosses a cell has its baseline pulled up by the cell, the
fit compensates by pushing that line down, and what comes out is a dark trench
along each side of every raised object and a cell whose surface is no longer
flat. On the yeast scans in `Data to test`, the ordinary row-by-row cubic pushes
the cells all the way down onto the substrate: a cross section shows the cell
and the substrate between the cells at *the same height*, with the 20–47 nm the
cells actually stand proud of it removed as though it were an artifact.

The fix is not a better polynomial. It is to fit the background only to the
background, which means finding the background first. This module follows

> Y. Wang, T. Lu, X. Li and H. Wang, *Automated image segmentation-assisted
> flattening of atomic force microscopy images*, Beilstein J. Nanotechnol.
> **2018**, 9, 975–985. [doi:10.3762/bjnano.9.91](https://doi.org/10.3762/bjnano.9.91)

in two steps: segment the features and exclude them, then fit what is left.

**1. Find the features** (`segment_foreground`). A threshold gives a first
outline, which is always too small — a threshold necessarily cuts a bump partway
up its flank. Two thresholds are offered and they are for different samples:

| `threshold` | How it splits | Use it when |
| --- | --- | --- |
| `otsu` | one threshold for the whole image, on a heavily smoothed copy | features are **large** — cells, anything a good fraction of the frame across. This is the default, and it is what the `Data to test` folders need. |
| `adaptive` | each pixel against the median of a `neighbourhood` around it | features are **small and many** — the paper's nanobubbles and nanopits. The neighbourhood must be wider than any one feature, or the window sees the middle of a cell as its own background. |

`detect` chooses convex features (bumps, cells), concave ones (pits, holes — the
paper complements the image; so does this) or both. `both` is refused with
`otsu`, which splits the image in two and would therefore mask all of it.

**2. Take the outline out to the foot** (`expand_contour`). What the threshold
leaves outside the mask is the foot of the feature, and the foot is the worst
possible thing to feed a background fit — it is the steepest part of the error.
The mask grows a pixel at a time and a pixel joins only while the gradient says
the flank is still falling, so growth stops on its own at flat ground; `expand`
is a limit, not a target, and `edge` sets how much slope still counts. `grow`
adds a plain margin afterwards and `min_area` drops specks.

**3. Fit what is left** (`fit_background`). `rows` fits a polynomial along each
scan line (the paper's curve fitting), `columns` does the same down each column,
`both` does one and then the other, `surface` fits one polynomial over the whole
image, and `auto` measures which way the scan lines run and follows them. Masked
pixels are simply not in the least-squares problem. `rows` is the default and is
usually right: drift is slow compared with a scan line, so it lands between
lines rather than along them, and only a per-line fit can take it out — the
paper reaches the same conclusion by rotating an image 90° and watching the
flattening get worse.

Setting `window` above 0 switches on the paper's sliding-window fit (SWCF/SWSF)
for backgrounds too complicated for one polynomial. A window of that many pixels
steps across the image one pixel at a time; at each position a polynomial is
fitted to the background inside it and recorded at every pixel it covers, and
each pixel averages the values from every window that reached it. Raising the
order instead is the obvious alternative and the wrong one — a high-order fit
oscillates between the points that constrain it. Smaller windows follow the
background more closely; larger ones start leaving a corrugation behind. This is
computed in closed form rather than by looping over positions (the normal
equations of every window at once are a handful of `correlate1d` passes, and the
averaging is a convolution of the coefficient images), which is what makes it a
second or two on a 512×1024 scan instead of a quarter of a million least-squares
fits. It is checked against a literal position-by-position implementation to
1e-7.

**Two departures from the paper**, both about images its nanobubbles never
produced. A fit is **rejected outright when it would be extrapolating** rather
than interpolating — when the background it has left is bunched at one end of
the line or one corner of the window, so the polynomial it supports says nothing
about the rest of it. A whole-line fit then drops its order until one is
supportable, down to a constant; a window drops out and the whole-image fit
stands in. This is not a refinement: without it, on a scan two thirds covered by
cells, the fitted background swung over 700 nm on an image whose features are
40 nm tall. And the **contour expansion is not the paper's active contour** —
same purpose, one parameter instead of two, no dependency beyond scipy.

**Finding the features is kept separate from removing the background**, and
that turned out to matter more than anything else here. Segmenting needs a
reasonably flat image: on a raw scan the drift between one line and the next is
routinely larger than the cells sitting on it, and segmenting *that* marks the
bright scan lines rather than the sample. So the features are always looked for
on a `seed_background` copy — a plane off and a **second-order polynomial off
every scan line**, which is the standard way of making a raw scan readable —
whatever fit is going to be used afterwards. That copy is a bad image (it is
exactly the one this module exists to avoid: it dents the features) but a good
image to *segment*, and it is thrown away. Only then is the background fitted,
once, the way you asked.

Both halves of that are measured rather than assumed. Scored against
`gwy_balance.segment_cells` on a properly flattened image — an independent
segmentation, already checked against these same scans — over **18 scans from
six sessions**:

| seed order | 1 pass | 2 passes | 3 passes |
| --- | --- | --- | --- |
| 1 | 0.477 | 0.337 | 0.306 |
| **2** | **0.912** | 0.465 | 0.324 |
| 3 | 0.572 | 0.372 | 0.253 |

(mean intersection-over-union with the reference mask). Second order at one pass
was the best of the nine on *every single scan*: first order leaves the drift
in, third order starts following the features.

That one look is already the paper's two-step segmentation — its last section
segments, flattens with the resulting mask and segments the flattened image, and
the seed here *is* that flattening. `passes` above 1 repeats it again, and the
table shows what that does. The reason is worth knowing before raising it: once
the features are restored to their full height they have a wide spread of their
own, and Otsu's threshold, which splits a histogram wherever that separates it
best, starts splitting *inside* the features instead of between them and the
substrate.

The other consequence of seeding every pass is that **the mask no longer depends
on the fit**. Segmenting the fit-flattened image instead — the obvious reading of
the paper — makes it depend badly: with `fit="surface"` the flattened image still
has every line-to-line step in it, because no surface of any order can remove
drift that lands *between* lines, so the mask that comes back disagreed with the
`fit="rows"` one on about three quarters of its area. Seeding brings that
disagreement to exactly nothing across all 18 scans. Which features are on the
sample is a fact about the sample, and should not change with how you intend to
remove the background.

#### The direction, and areas you exclude by hand

A second paper describes a whole pipeline — classify the scan, segment the
artifacts with a network, inpaint them — whose flattening stage is the same idea
as Wang's with three additions:

> J. Zhang, A. Biswas, J. Rade, C. Shukla, J. Ren, A. Sarkar, A. Krishnamurthy
> and A. Balu, *Artifact Removal and Image Restoration in AFM: A Structured
> Mask-Guided Directional Inpainting Approach*,
> [arXiv:2602.04051](https://arxiv.org/abs/2602.04051) (2026).

Those three are here; the rest of it is not (its exclusion mask is a global
`|z − mean| > kσ` threshold plus a dilation, its fallback for a line with too
few background pixels is that line's median, and its classifier, segmentation
network and Telea inpainting are about repairing artifacts rather than
levelling).

**`fit="auto"` measures which way the scan lines run** (`choose_direction`). The
paper picks the direction from "the dominant slope direction"; taken literally
that is the wrong statistic, because a smooth tilt is removed just as well
either way, so the slope cannot decide between them. What only a line-by-line
fit can remove is the part of the background that is *incoherent* between
neighbouring lines — the offset the scanner has drifted to by the time it starts
the next one. So that is what is measured: the spread of the step from one line
to the next, with the part of it the noise inside the lines already explains
subtracted back out, compared both ways. It picks `rows` on all 18 test scans
and `columns` on every one of them transposed.

Worth having, because the wrong direction is not a small loss. Over 17 scans
(the 18th is 99 % cells and has no background left for anything to work with),
medians:

| `fit` | background rms | feature height above the substrate beside it |
| --- | --- | --- |
| `rows` | 5.74 nm | 3.19 nm |
| `columns` | 42.74 nm | 0.14 nm |
| `both` | 5.31 nm | 2.95 nm |

Fitting these scans down columns leaves seven times the background roughness
*and* wipes 96 % of the feature height, because a column crossing a cell has to
interpolate across it and the polynomial that does so passes through the cell.

**`fit="both"`** is the paper's two-step: a polynomial off every row, then one
off every column of what is left. The second fit sees no row-to-row drift, so
what it removes is the column-to-column part, which a single line fit cannot
reach. It costs what the table shows — 5 % less background roughness for 3 % of
the feature height on median, but as much as half of it on one scan — so it is
not the default. Use it when there is genuine drift both ways, and watch the
third panel of the dialog while you do.

**`exclude`** is a mask, or in the GUI a rectangle dragged on the image, that is
kept out of the fit whatever the threshold thinks of it: a step edge, a piece of
debris, the corner where the tip crashed — anything a threshold has no way of
recognising. It is fed to the seed as well as to the final fit, so the excluded
area cannot bend the image the features are looked for on either. On a synthetic
scan with a trench across every line and `detect="convex"` (which cannot see a
pit), the fitted background follows the trench 26 nm away from the truth;
marking it by hand brings that to 0.26 nm.

**Nothing here rescales z.** The result is `data - background`, a subtraction,
so a height measured on the result is the height that was measured on the
sample. Multiplying the input by three multiplies the output by exactly three,
which is checked for every fit and window setting.

**What it does not do, by construction.** A per-line fit corrects each line by
what its own background says, so drift that happened while the tip was crossing
a cell — where there is no background to see it in — stays in the image. The
ordinary levelling appears to remove it, but only because it is re-levelling
each line on the cell itself, which is the same operation that flattens the
cell. So a scan whose lines are individually noisy will still look banded
*inside* the features after this step, and the honest order is to level with
**Smart background** and then take the line noise out with
[Stripe Removal](#stripe-removal-gwy_destripepy), which is built to tell a
stripe from topography. Levelling harder is not the answer to a stripe.

In the GUI, **Smart background...** sits directly under *Plane level* and *Poly
background*. Its dialog shows three panels: the result with the excluded area
outlined, the background that was removed, and **what the mask changed** — the
difference against the same fit run over every pixel, i.e. against the ordinary
levelling two buttons up. Where that panel is flat the mask made no difference;
where it is bright, that is how much of the sample the ordinary fit was
subtracting from itself. Underneath, the mask's share of the frame is reported,
with a warning if almost nothing is left to fit, if nothing was masked at all,
or if any scan lines had no background left and had to be interpolated from
their neighbours. A fourth row of controls covers the manual exclusion: **drag
on the result panel** to keep that rectangle out of the fit, right-click one to
take it back. The rectangles are carried in physical units, so they survive into
the pipeline, the log and a batch replay.

```python
import numpy as np
import gwy_flatten as gf

result = gf.flatten(image, threshold="otsu", detect="convex",
                    fit="rows", order=3, window=0)
flat = result["data"]          # the levelled image
result["mask"]                 # what was kept out of the fit
result["coverage"]             # how much of the frame that was

# let the scan choose the direction, and exclude a corner by hand
corner = np.zeros(image.shape, dtype=bool)
corner[:64, :64] = True
result = gf.flatten(image, fit="auto", exclude=corner)
result["fit"]                  # "rows" or "columns", whichever it measured
```

### Stripe Removal (`gwy_destripe.py`)

Stripe artifacts are elongated, roughly parallel corruptions that share one
direction. In AFM they appear as scan-line offsets: a line, or a run of
lines, sits at the wrong height because the feedback settled badly, the tip
picked something up, or the drift jumped. The same artifact class is called
*curtaining* in FIB-SEM, *striping* in light-sheet microscopy and remote
sensing, and the literature is largely shared.

Three methods are implemented, selected from the **Method** dropdown of the
GUI's *Stripe removal* window, which shows only the chosen method's
parameters. Two of them, **MDSR** and **GSR**, come from the
[General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)
project [[5]](#stripe-references); the third, **DeStripe**, was written for
AFM specifically. All three assume the same decomposition of the recorded
image,

```
u0 = u + s          u = the clean image,  s = the stripes
```

and the first two take the stripe direction in degrees, `angle`, with **0° =
horizontal scan lines** (the usual AFM case) and 90° = vertical. What
separates them is *how they decide which part of the image is `s`*: **MDSR**
answers in the frequency domain — stripes are a narrow band of frequencies —
**GSR** answers by optimization — stripes are whatever is sparse, elongated
along the given direction, and leaves a clean image behind — and **DeStripe**
answers statistically, from the image's own spectrum, which is why it needs
no direction and no parameters at all.

> The sections below describe how each method works. For **which one to use
> and how to set its parameters**, with measured comparisons on data whose
> answer is known, see [STRIPE_REMOVAL_GUIDE.md](STRIPE_REMOVAL_GUIDE.md).

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

#### DeStripe — noisy pixels of the spectrum (Chen & Pellequer)

`destripe_chen` implements the method of Chen and Pellequer
[[11]](#stripe-references), the only one of the three written for AFM, and
the only one that needs nothing from you at all — not even the stripe
direction. Where MDSR decides what a stripe is from a model of the frequency
plane and GSR from an energy, DeStripe decides it from the image's own
spectrum: in `LogF = log|FFT(image)|` the stripes appear as a few abnormally
bright pixels arranged in lines, and the method finds those pixels
statistically and pulls them down to the level of their neighbours. The
phase is never touched.

**Step 1 — heterogeneity.** Every pixel of LogF gets

```
H = (L − Lmin)/(Lmax − Lmin) · (I − Imin)/(Imax − Imin)  ∈ [0, 1]
```

where `L` is the discrete Laplacian (the paper's Table 1: −1 around a center
of 8) and `I` the log-amplitude. `H` is large only where a pixel is *both*
bright and abruptly brighter than its surroundings — one of the two alone is
not enough, which is why real structure, bright but smooth, mostly survives.

**Step 2 — global sampling.** The threshold on `H` is read off its own
20-bin histogram: take the longest run of populated bins and walk from its
peak towards higher `H` until a bin holds at most half of the peak's count;
`Href` is the middle of that bin. Over the quiet pixels (`H ≤ Href`) the
intensity threshold `Iref = (max + mean)/2` is formed, and the first
candidate set is

```
Pn1 = { H > Href  and  I > Iref }
```

**Step 3 — divide and conquer.** The neighbourhood of the origin is where
the amplitude changes by orders of magnitude, so it is treated separately.
The intensity-weighted moment-of-inertia tensor of `Pn1` gives a center and
an initial radius `√(σx + σy)`; a disk grows around the center in tenths of
that radius while more than `density` of the pixels inside it are candidates.
Inside (`C0`) an anisotropic Gaussian is least-squares fitted to the
intensities and only pixels *above* the fit stay candidates — a peak that the
smooth model already explains is not noise. Outside (`Pn2`) a pixel stays a
candidate if it stands more than `cvar_k` standard deviations above its own
neighbourhood.

**Step 4 — lines.** Both sets are thinned once more by the same histogram
rule (10 bins this time) and then kept only where they look like a line: a
row or column of the region that is more than two thirds candidates, or a run
of `min_run` consecutive candidates. This is the step that separates a stripe
from a bright speck, and it is why the method needs no direction — a
horizontal line in the spectrum and a vertical one are equally acceptable.

**Step 5 — the CVAR test and the filter.** For each surviving pixel the mean
and standard deviation of the *non-candidate* pixels in its
`(2·window+1)²` neighbourhood are taken (that constraint is the C in CVAR),
and the pixel is pulled down to that mean if it exceeds it by more than
`cvar_k` standard deviations. Clusters are worked from their boundary
inwards, so a pixel in the middle of one still has restored values to average
over. The filter is then

```
Φ = exp(restored LogF) / exp(LogF)   ∈ (0, 1]
```

and the result is the inverse FFT of the spectrum times `Φ`. Since `Φ ≤ 1`
the method can only ever *take energy out* of the image — the paper insists
on this, on the grounds that a denoiser which adds height to an AFM
measurement is inventing topography. The dialog's bottom right panel is `Φ`
(the paper's F-image): yellow is kept, dark is removed, and the title counts
how many frequencies were touched.

Parameters:

| Parameter | Meaning |
|---|---|
| `cvar_k` | How far above its neighbours a frequency must sit to count as noise, in standard deviations. The main knob: lower removes more. |
| `window` | `NS` of the `(2·NS+1)²` neighbourhood. The paper uses 1. |
| `density` | Candidate density at which the central disk stops growing (0.85 in the paper). |
| `min_run` | How many candidates in a row make a line (4 in the paper). Larger is more conservative. |
| `keep_mean` | Leave the amplitude at the origin — the mean height — alone. |

The paper fixes all of these internally and takes the raw image as its only
input; they are exposed here because AFM scans vary more than the set the
values were tuned on, and because watching them is the only way to see what
the method is doing.

Three places where the paper leaves a step underdetermined are marked in the
source: the direction of the histogram walk, the normalization of the inertia
tensor (its radius is only a radius if the tensor is divided by the total
mass), and the "VAR test" that the flow chart names but the text never
defines — the constrained version of it, the CVAR test, *is* defined, so the
unconstrained one is used. `keep_mean` is a deliberate deviation: the paper
lets the origin be restored like any other pixel and reports that for one SEM
image most of the striping was in fact the amplitude at the origin, but for
an AFM height map that amplitude is the mean height, and scaling it moves the
whole surface up or down without touching a single stripe.

**What it does on AFM data.** Per-line offsets live on the `fx = 0` column of
the spectrum, and that is exactly where the method finds them: on a synthetic
scan (grainy topography plus random per-line offsets) every noisy frequency
it identifies is on that column, and the error against the true topography
falls by a quarter. On a real scan with strong topography it is more
selective — it removes a handful of periodic bands rather than the broad
stripe content — and because it acts on individual frequencies, what it takes
out is a set of clean sinusoids across the whole image. Check the removed
panel: if it shows a regular ripple rather than the stripes you meant to
remove, raise `cvar_k` or use one of the other two methods.

#### Parameter sweep (all three methods)

Parameters like these are best judged by eye and against
each other, so *Parameter sweep...* runs a grid over **two parameters you
pick** in that window and shows the results side by side — 3×3 by default,
one parameter down the rows and one across the columns. Nothing is computed
until *Run* is pressed; the cells then fill in one at a time so the window
stays responsive.

The window offers the parameters of whichever method the dialog is set to,
and each axis steps around the value currently in the dialog in the way that
suits it: gains and widths are *multiplied* (`mu1`, `mu2`, `sigma`,
`directions`, iterations — they span orders of magnitude), while counts,
angles and thresholds are *incremented* (`levels`, `angle`, `max_angle`,
`cvar_k`, `min_run`, `density`, `window`). GSR opens on `mu1` × `mu2` with
*Same rate* ticked, which keeps both axes on one step factor: with factor 2
the rows are `mu1/2, mu1, 2·mu1` and the columns `mu2/2, mu2, 2·mu2`, and the
diagonal is then the "scale both together" direction the paper describes.
MDSR opens on `sigma` × `levels` and DeStripe on `cvar_k` × `min_run`; there
the two steps are independent — *Same rate* is only ticked by default for two
multiplied axes, since two incremented parameters do not share a unit.
Values that would leave a parameter's allowed range are clamped to it, and
`directions` stays on powers of two.

Every cell is computed on the whole image and only then cropped, so the zoom
area picked in the dialog (drag on the result panel, then *Zoom area only*)
changes what you see and never what is computed. All cells share one color
scale, the panels are linked for toolbar zooming, and clicking a cell copies
both of its values back into the dialog.

#### Choosing between them, and two shared caveats

MDSR is a single linear filter: fast, predictable, and easy to reason about
because the mask is visible. It suits regular stripes of roughly constant
width. Its cost is intrinsic — it removes *everything* inside a frequency
band, including real structure elongated along the scan lines.

DeStripe is the conservative one, and the one to try first when you do not
know the stripe direction or do not want to choose parameters: it touches
only the frequencies it can argue are noise, typically a fraction of a
percent of the spectrum, and it can only remove energy. The same caution is
its limit — on an image whose stripes are broadband it will under-remove, and
the paper says so itself.

GSR is an optimization and adapts to the image, which is why
[[2]](#stripe-references) reports it outperforming both MDSR+ and the
variational VSNR [[3]](#stripe-references) on light-sheet, FIB-SEM and remote
sensing data (PSNR, MS-SSIM, curtaining metric, line profiles), particularly
on irregular stripes of varying width and on short trails. It is by far the
slowest of the three here, and its own documented limitation is that image
structures aligned with the stripe direction get reduced along with the
stripes.

Two things apply to all three:

* **Level the image first.** A plane tilt across the slow axis *is* a set of
  line offsets, and no method can tell it from an artifact — it will be
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
11. S.-w. W. Chen and J.-L. Pellequer, "DeStripe: frequency-based algorithm
    for removing stripe noises from AFM images", *BMC Struct. Biol.* **11**,
    7 (2011).
    [doi:10.1186/1472-6807-11-7](https://doi.org/10.1186/1472-6807-11-7)

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

### Quick view (folder browser)

The GUI has three tabs. *Processing* is the workbench described above; **Quick
view** is for looking through a measurement session without deciding anything,
and **Balanced view** puts a whole folder on one colour scale.

* **Select folder...** lists every `.gwy` file in it, in natural order
  (`_2` before `_10`).
* Each image is shown with the same two steps applied — a fitted plane
  subtracted, then rows aligned with a **second-order polynomial** — which is
  what makes a raw scan readable. Nothing is written, and nothing carries over
  to the processing tab.
* **Next** / **Back** step through the folder, with the position (`7 / 17`) and
  the file name above the image. Click the image once and the left/right arrow
  keys do the same.
* The **Channel** box works like the one in the main window and keeps your
  choice as you step; a file that does not have that channel falls back to one
  whose name starts the same way, then to a Height channel.
* Results are cached per file and channel, so stepping back is instant. The
  cache holds the last 32 images and then drops the oldest.

### Balanced view (one colour range for a folder)

A stack of scans of the same sample — successive layers of a cell, or a time
series — almost never shares a false-colour range. Each image is levelled on its
own, each has a different fraction of its frame covered by cells, and each has a
different amount of drift left in it. Set the range from the whole image
(percentiles, or min to max) and the answer moves from image to image: the same
physical height comes out a different colour, and a layer whose cells happen to
be taller washes out while its neighbour stays flat. Comparing layers by eye is
then comparing two different colour scales.

**Balanced view** stops measuring the whole image and measures two *places*
instead, both of which mean the same thing in every image: the substrate between
the cells, and the inside of the cells. How much of the frame each one covers no
longer matters, which is the point — that is what a plain percentile gets wrong.

**How each image is measured** (`gwy_balance.measure`):

1. The image is levelled like the quick view levels it (plane, then rows with a
   second-order polynomial) unless **Level first** is unticked.
2. It is segmented into cells and substrate. Segmenting on the height histogram
   alone does *not* work on levelled AFM data: row alignment gives every row its
   own baseline, so a row that is mostly cell is pushed down towards the
   substrate and the two populations smear into one broad peak. The shape
   survives though — cells are large, smooth and connected, the substrate
   texture is fine — so the split is made on a heavily smoothed copy, where the
   histogram is bimodal again and Otsu's threshold lands in the valley. **Cell
   size (% of frame)** is that smoothing width; raise it if the mask follows the
   texture, lower it if neighbouring cells merge.
3. Both masks are shrunk before any statistic is read, so the bright rim at the
   edge of a cell counts as neither substrate nor interior. Including the rim
   would drag the top of the range up and flatten exactly the structure inside
   the cell that the range is supposed to reveal.
4. The anchors are the median of the substrate and the **cell percentiles** of
   the interior (2 % and 98 % by default).

A split is only believed when both classes are at least 2 % of the frame and
still big enough to take a median of after shrinking. That test earns its keep:
in a frame packed edge to edge with cells, the only thing left below the
threshold is often a scan artifact — a hole a hundred nanometres deep covering
one percent of the image — and measuring "the substrate" there puts the anchor
far below anything real and squashes the picture. When the split fails, the
image is treated as all cell, its lower quartile stands in for the substrate,
and the status line says how many images that happened to.

**The three modes:**

| Mode | Bottom of the range | Top of the range |
| --- | --- | --- |
| Per image (no sharing) | that image's own cell-interior low percentile | that image's own cells |
| **Substrate anchored** (default) | the folder's median, so the substrate reads the same colour everywhere | that image's own cells |
| Common range (shared) | the folder's median | the folder's median — one range for everything |

`Substrate anchored` is the default because of what does and does not vary
between scans of the same sample. Two things move: **where the substrate sits**
— drift, thermal motion and the plane fit, an artifact with no physical content,
which should go — and **how far the cells stand above it**, which is the sample,
and should not.

`Common range` removes both, so real height differences between layers get
flattened or clipped. `Per image` removes neither: its bottom is that image's own
cell-interior percentile, which wanders with the segmentation, so on one folder
here the substrate between the cells lands anywhere from **5 % to 52 %** up the
colour map — visible as the substrate changing colour from image to image for no
physical reason, which is the original complaint. `Substrate anchored` pins the
bottom to the folder and lets the top follow each image's own cells, so a taller
layer is drawn taller instead of clipped or squeezed, and the range boxes let you
type over the bottom (the shared end) while each top stays that image's own.

**Only offsets are ever applied — the z scale of the data is never touched.**
That is a hard rule, not a default, and `gwy_balance.apply_levels` is an
addition and nothing else. An earlier version had a third mode that stretched
each image onto the folder's median span with a per-image gain. It made a set
look more uniform, and it was wrong: on the yeast data it rewrote a cell
measured 46.7 nm above the substrate as 27.8 nm — a 40 % error — and that number
went into the exported PNGs and `.gwy` files. Uniform appearance is not worth a
false height, so the gain was removed rather than merely defaulted off.

The consequence to expect: under `Common range`, a layer whose cells are
genuinely taller than the folder's median will clip at the bright end. That is
real information, not a defect — the diagnostics view reports the clipped
fraction per image, and the range can always be widened by hand. `Substrate
anchored` does not clip that way, because each image's top is its own.

**Looking at the result.** **View** switches between a single image (with
**Next**/**Back**, or the arrow keys after one click on the image), a **contact
sheet** of the whole folder at once with the current image outlined, and
**Diagnostics** — the mask over the image, where this image sits inside the
range, all four anchors across the folder on the balanced scale, and the gain
and cell coverage per image. The contact sheet is the quickest way to see
whether the balance worked; the diagnostics say why it did not.

**The range is a starting point, not a verdict.** Type over it and press
**Apply**, or press **Auto** to go back. The default deliberately favours the
inside of the cells: the bottom of the range comes from the *cell interior*, so
the substrate between the cells sits at the dark end and its texture is partly
clipped. Lowering the bottom of the range brings the substrate texture back, at
the cost of contrast inside the cells — the diagnostics histogram shows exactly
what is being cut.

**Baseline to zero** (on by default) shifts the whole set — every image *and*
the range, by the same amount — so the bottom of the range reads 0 and every
colour bar runs `0 … span` instead of starting at a negative number. The
substrate then sits a little above zero, by the same amount in every image.
Note this is deliberately *not* the usual per-image "set baseline to zero"
(subtract each image's own minimum): that would give every image a different
offset again, which is the thing this tab exists to undo.

**Export all...** asks what to write and then for a folder. For every image,
any of:

* **Annotated PNG** — axes, colour bar and scale bar, as the main window's
  *Save processed image...* writes them — `<name>_balanced.png`.
* **Pure image** — one pixel per data point, no labels, square pixels — in a
  `pure/` subfolder, same file name.
* **Gwyddion `.gwy`** — the balanced channel, titled
  `<name> - <channel> (balanced)`.

All of them use the range the tab is showing and the current colour map, so they
can be compared side by side. Because balancing only shifts, the `.gwy` carries the
**measured heights**: its data differs from the source channel by one constant
and by nothing else. It also holds the **full data** — only the display is
clipped to the range — and an existing file is replaced rather than appended to,
so re-exporting is repeatable. Existing files are listed and confirmed before
anything is overwritten.

Outside the GUI:

```python
import gwy_balance as gb

measures = [gb.measure(image) for image in levelled_images]
result = gb.zero_baseline(gb.balance(measures, "substrate"))
shown = [gb.apply_levels(image, offset)
         for image, offset in zip(levelled_images, result["offsets"])]
# image i is drawn with vmin, vmax = result["ranges"][i]
```

### Colour maps

The **Display** box in the main window selects the false-colour gradient, with a
strip underneath showing what it looks like. The choice applies to the main
image, to every dialog preview drawn from then on, and to saved images, so what
you export is what you saw.

The list holds all 60 of Gwyddion's gradients. They are not look-alikes: each
one is the stop table from Gwyddion's own `share/gwyddion/gradients` resource
files, transcribed into `gwy_colormaps.GRADIENTS` and rebuilt with
`LinearSegmentedColormap.from_list`, which interpolates between stops exactly as
Gwyddion does. The dozen most used ones (`Gray`, `Gwyddion.net`, `Gold`, `Body`,
`Sky`, `Spring`, `Cold`, `Warm`, `Rainbow1`, `Rainbow2`, `Viridis`, `Spectral`,
`BW1`, `Gray-inverted`) are listed first, then the rest alphabetically. The
default is `Gwyddion.net`, the black → dark red → yellow → white palette this
project has always drawn with — it was previously an approximation built in
`gwy_processing.get_gwyddion_cmap()` and is now the exact gradient (the colours
moved by less than 0.5 %).

Outside the GUI:

```python
import gwy_colormaps as gcm

gcm.names()             # every gradient, common ones first
cmap = gcm.get("Sky")   # a matplotlib colormap
gcm.set_current("Sky")  # what gcm.current() returns from now on
```

`get()` also accepts matplotlib names (`"viridis"`) and falls back to the
default instead of raising, so a stale name cannot stop a window from drawing.

### Visualization
* `get_gwyddion_cmap()` returns Gwyddion's default palette; `gwy_colormaps.current()` returns whichever gradient the user has selected.
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

### Levelling a scan without flattening the sample into the background

```python
from gwy_loader import load_gwy
import gwy_processing as gp
import gwy_flatten as gf

channels = load_gwy('sample_scan.gwy')
image = channels['Height [Fwd]'].data * 1e9          # nm

# Take the tilt out first, then fit the rest to the background only.
result = gf.flatten(gp.level_by_plane_fit(image), fit="rows", order=3)

flat = result["data"]
inside = flat[result["mask"]]
outside = flat[~result["mask"]]
print(f"{100 * result['coverage']:.0f} % of the frame is sample, "
      f"standing {inside.mean() - outside.mean():.1f} nm above the substrate")
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