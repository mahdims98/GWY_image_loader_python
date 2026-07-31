# Destriping in practice

Three stripe removers are implemented in `gwy_destripe.py` and share one
dialog in the GUI. This is a practical guide to them: what each one actually
decides, what it is good at, where it fails, and what every parameter does to
the result. The formal descriptions and the references are in the
[README](README.md#stripe-removal-gwy_destripepy); this text is about
choosing.

All numbers below are measured, not estimated. The synthetic test image is
grainy topography (Gaussian-filtered noise, rms 7.67 nm, 256×256) plus a
known stripe pattern, so the error against the *true* surface can be
computed. The real scan is `..._0009.gwy`, plane-leveled, 256×512.

---

## 1. What they have in common, and the one thing none of them can do

All three assume the recorded image is `u0 = u + s` — a clean surface plus
stripes — and all three try to identify `s`. They differ only in what
evidence they use: MDSR uses *frequency* (stripes are a narrow band), GSR
uses *energy* (stripes are sparse and constant along their direction),
DeStripe uses *statistics of the spectrum* (stripes are abnormally bright,
line-shaped groups of frequencies).

**The hard limit is the same for all of them.** A per-line height offset and
a real surface feature that happens to be constant along a scan line are the
same signal. Nothing in the image distinguishes them. In the synthetic test
above, the true topography's own row means have an rms of **1.07 nm** while
the injected stripes have an rms of **0.99 nm** — so a filter that removes
the whole `fx = 0` line of the spectrum, which is what "remove per-line
offsets" means, necessarily destroys about as much real signal as noise.
This is not a defect of any implementation; it is what the information
allows.

Two consequences worth internalising:

* **Level first, always.** A tilt across the slow axis *is* a ramp of line
  offsets. Every one of these methods will eat it, and then you have removed
  the very tilt you could have measured properly with a plane fit.
* **Judge with two numbers, not one.** "Line-to-line jump" (the mean
  absolute difference between neighbouring row means) measures how striped
  the image *looks*. Error against the truth measures whether the surface is
  still the surface. They disagree, and the disagreement is the whole
  trade-off. On the synthetic image MDSR drives the line-to-line metric from
  1.12 to **0.01** — visually perfect — while raising the error from 0.99 to
  **2.53**. It bought a clean-looking image by deleting real topography.

---

## 2. MDSR — a fixed frequency mask

*Liang et al. 2016. In this implementation the whole contourlet decomposition
collapses to a single frequency mask, which is exact and makes it fast.*

The mask is a groove along the stripe frequencies, tapered by direction and
by scale. Everything inside the groove goes; everything outside is untouched.
The dialog shows the mask, which is the method's best feature: you can see
precisely what will be removed before you apply it.

**Advantages.** Fast (15 ms on 256², 60 ms on 256×512) and completely
predictable — it is a linear filter, so the same settings do the same thing
to every image, which is what you want when batch-processing a series. Its
effect is visible in advance. It is also the most thorough: per-line offsets
are removed *completely*, not reduced.

**Problems.** It cannot distinguish. Everything in the groove is removed
whether it is a stripe or a real ridge running along the scan direction. It
has no concept of "this line looks fine" — a scan with three bad lines out of
256 is filtered exactly as hard as one that is striped throughout. And
because a hard band is removed from the spectrum, what comes out can carry
faint periodic ripples.

### Parameters

| Parameter | What it does | How to set it |
|---|---|---|
| `angle` | Stripe direction, 0° = horizontal scan lines. | Almost always 0 for AFM. Get this wrong and the method removes a band of real structure and no stripes. |
| `sigma` | Width of the groove **across** the stripe frequencies, in frequency bins. | The main knob. See below. |
| `levels` | How many octave scales are filtered. Sets which stripe *periods* are reachable. | See below. |
| `directions` | Angular resolution of the filter bank. | Leave at 8. Measured effect across 4/8/16/32: error 2.65 / 2.53 / 2.66 / 2.68 — noise. |
| `sigma_a` | How fast the damping fades for directions away from the stripes. | Leave at 0.3. Raising it towards the paper's 0.8 filters more directions and removes more unrelated structure. |
| `max_angle` | Directions beyond this angle from the stripes are not filtered at all. | Leave at 45°. |
| `pad` | Mirror the edges before the FFT. | Turn on if you see ringing at the top and bottom edges; it costs a factor of four in time. |

**`sigma` does not control whether per-line offsets are removed.** They are
removed at any σ — the `fx = 0` line is zeroed regardless. What σ controls is
how far the groove reaches *along* the stripes, i.e. how much real structure
elongated in the scan direction is sacrificed. Measured on the mask (256 px
wide, at `fy` = 64 bins), the transmission at 4 bins from the center is 0.87
for σ=1, 0.40 for σ=2, 0.08 for σ=5 and 0.005 for σ=20. As a rule of thumb the
groove takes out horizontal structure longer than about `width / 2.5σ`
pixels — 100 px at σ=1, 20 px at σ=5, 5 px at σ=20 on a 256-px scan. Since
that is a *length in pixels*, **σ must be rescaled when the scan size
changes**; the value 5–25 recommended in the literature was tuned on ~1000 px
images and is aggressive on a 512 px AFM scan.

Effect on the synthetic image (error against the truth, and how much was
removed):

| σ | 0.5 | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| error (nm) | 1.07 | 1.36 | 1.82 | 2.53 | 3.10 | 3.68 | 4.27 |
| removed rms (nm) | 1.39 | 1.62 | 2.03 | 2.68 | 3.22 | 3.79 | 4.36 |

Every increase in σ removes more and costs more real signal, while the
visible striping is already gone at σ=0.5. **Start low, raise only while the
preview still shows stripes.**

**`levels` sets the coarsest stripe you can reach.** The center of the
spectrum is a low-pass residual that is never filtered, and each extra level
halves it. Measured on a 256-px image, the mask along the stripe line is 1
(kept) below `width / 2^(levels+1)` bins and 0 above it:

| levels | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| stripe patterns left untouched | everything | slower than 8 rows | slower than 16 rows | slower than 64 rows | nothing |
| removed rms (nm) | 0.87 | 1.61 | 2.25 | 2.68 | 2.72 |

The threshold is `2^(levels+1)` pixels, and — unlike σ — it is a fixed number
of *pixels*, so it does not have to be rescaled when the scan size changes.
`levels` answers "does my striping repeat every few lines, or does it drift
over a hundred?" Slow drift needs more levels, and more levels also means
slow *real* topography goes. At `levels = 5` anything that repeats over more
than 64 rows survives; that is usually the right compromise, and the default.

---

## 3. GSR — a variational split

*Rottmayer et al. 2025. Minimizes `μ1‖∇u‖ + ι[0,1](u) + ‖∇θ s‖₁ + μ2‖s‖₁`
subject to `u + s = u0`, by primal-dual iteration.*

Instead of declaring a band of frequencies guilty, GSR asks for the *pair*
`(u, s)` that best satisfies four wishes at once: the clean image should have
few strong edges, the stripe image should vary little along the stripe
direction, the stripe image should be mostly zero, and the two should add up
to the input.

**Advantages.** It adapts. Because sparsity of `s` is part of the objective,
it can leave most of the image alone and concentrate on the lines that are
actually bad — something neither of the other two can do. This is why the
paper reports it beating MDSR+ and VSNR on irregular stripes of varying
width and on short trails. It also produces no periodic ripple: the stripe
image is a spatial object, not a frequency band.

**Problems.** It is by far the slowest — 313 ms for 600 iterations on 256²,
1.1 s on 256×512, and a fully converged result wants several thousand — so
the live preview lags and a parameter sweep costs real time. The result
depends on a convergence state as well as on the parameters, so "the same
settings" do not strictly mean "the same filter" unless the iteration count
is fixed. Its own documented weakness is the shared one, sharpened: real
structure aligned with the stripe direction is reduced along with the
stripes. And the direction is *snapped* — only 0°, 26.6° and 45° and their
flips are supported by the difference operator.

### Parameters

| Parameter | What it does | How to set it |
|---|---|---|
| `mu1` | Strength of the removal — the weight on the clean image's edges. | The main knob. Paper's interval: 0.1–0.5, default 1/3. |
| `mu2` | Retention — the weight on the sparsity of the stripe image. Larger keeps `s` sparser, so **less** is removed. | Paper's interval: 0.0016–0.017, default 1/300. |
| `iterations` | Primal-dual steps. Not a quality knob but a convergence knob. | 600 for previewing, 2000+ before applying. |
| `angle` | Stripe direction; snapped to the nearest supported step. | 0 for AFM. |

Measured on the synthetic image (`mu2` fixed at 1/300):

| `mu1` | 1/12 | 1/6 | 1/3 | 2/3 | 4/3 |
|---|---|---|---|---|---|
| error (nm) | 1.15 | 1.52 | 2.43 | 4.90 | 7.67 |
| line-to-line (nm) | 0.177 | 0.128 | 0.096 | 0.055 | 0.011 |

and with `mu1` fixed at 1/3:

| `mu2` | 1/1200 | 1/600 | 1/300 | 1/150 | 1/60 |
|---|---|---|---|---|---|
| error (nm) | 2.64 | 2.55 | 2.43 | 2.27 | 1.92 |
| removed rms (nm) | 2.75 | 2.68 | 2.56 | 2.40 | 2.05 |

The two knobs pull against each other and it is their **ratio** that sets the
behaviour; scaling both together changes how strictly the model insists the
stripes be ideal. That is why the sweep window ties them to one step factor
by default — the diagonal of that grid is the meaningful direction. Practical
procedure: sweep `mu1` first over a factor of 4 with `mu2` at its default,
pick the smallest `mu1` that clears the stripes, then adjust `mu2` up if real
structure is being eaten.

Convergence, measured as the mean deviation from a 5000-iteration result:

| iterations | 50 | 100 | 300 | 600 | 1200 | 2500 |
|---|---|---|---|---|---|---|
| deviation | 50 % | 26 % | 8.3 % | 4.4 % | 2.0 % | 0.7 % |

600 is honest for previewing and comparing parameters; it is not a converged
result. Raise it before the final apply — and note that a parameter chosen at
600 iterations is still roughly right at 5000, since the iteration converges
towards the same solution.

---

## 4. DeStripe — statistics of the spectrum

*Chen & Pellequer 2011. The only one of the three written for AFM, and the
only one that needs no direction and, in the paper, no parameters at all.*

It looks at `log|FFT(image)|` and asks which pixels *there* are anomalous:
both bright and abruptly brighter than their surroundings, and arranged in
lines. Those it pulls down to the level of their neighbours; everything else
it leaves exactly alone. The filter Φ it builds is ≤ 1 everywhere, so the
method can only take energy out of an image — it can never invent height.

**Advantages.** Selectivity. It typically touches a fraction of a percent of
the spectrum, so what it does not identify as noise is bit-for-bit untouched.
On every synthetic case tested it recovered the true surface better than the
other two, precisely because it removes *some* frequencies of the `fx = 0`
line rather than all of them. It needs no stripe direction — it finds
whichever lines exist in the spectrum, so oblique or vertical striping is
handled without being told. And it is fast (15–30 ms).

**Problems.** It under-removes, and the paper says so itself. Broadband
striping — every row offset independent — is spread over the whole `fx = 0`
line, and DeStripe only takes the peaks: on the synthetic random-offset case
it removed 113 of 256 frequencies on that line, leaving line-to-line at 0.57
where MDSR reached 0.01. Because it removes individual Fourier coefficients,
what it takes out is a set of clean sinusoids spanning the whole image, so
the removed panel often looks like a regular ripple rather than like stripes
— on the real scan it removed 32 frequencies out of 131 072 and the removed
image is periodic banding. Finally, its detection is threshold-based and can
fall off a cliff (see `cvar_k` below) rather than degrading smoothly.

### Parameters

| Parameter | What it does | How to set it |
|---|---|---|
| `cvar_k` | How many standard deviations above its neighbours a frequency must sit to count as noise. | The main knob, but not a smooth one. See below. |
| `min_run` | How many candidates in a row make a "line". The guard that separates a stripe from a bright speck. | Leave at 4. This is the parameter that keeps real spectral content safe. |
| `density` | Candidate density at which the central disk stops growing, i.e. how much of the spectrum center is treated as the special central region. | Leave at 0.85. Measured error at 0.5 / 0.7 / 0.85 / 0.95: 0.95 / 0.78 / **0.74** / 0.89 nm — the paper's value was best, and the response is not monotone, so there is nothing to tune towards. |
| `window` | Half-width of the neighbourhood the statistics are taken over. | 1 (the paper) or 2. Barely matters: error 0.735 / 0.728 / 0.728 / 0.731 for 1 / 2 / 3 / 5. |
| `keep_mean` | Protects the amplitude at the origin — the mean height. | Leave on. Off, and an image with a large mean offset can lose all of it (measured on a scan sitting at 99.9 nm: it came back at 0.2 nm). |

**`cvar_k` is a detector threshold, so it has a cliff.** Measured on the
synthetic image:

| `cvar_k` | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| frequencies removed | 137 | 137 | 113 | **0** | 0 | 0 |

Between 1.0 and 1.5 the method stops finding anything at all *on this image*
— where the cliff sits depends on how far the stripes stand out. On the
periodic-banding case the same method still found its frequencies at k = 2.
So: if nothing happens, lower it in steps of 0.25 rather than concluding the
image is clean; if too much happens, raise it; and do not expect a
proportional response — sweep it.

**`min_run` is the safety catch, and it works:**

| `min_run` | 1 | 2 | 4 | 8+ |
|---|---|---|---|---|
| frequencies removed | 821 | 209 | 113 | 113 |
| of those, *off* the stripe line | 702 | 96 | 0 | 0 |
| error (nm) | 2.40 | 1.17 | 0.74 | 0.74 |

At `min_run = 1` the "is it a line?" test is disabled and the method removes
702 frequencies that have nothing to do with stripes, tripling the error.
At 4 and above every removed frequency lies on the stripe line. There is no
reason to lower it, and little to gain from raising it.

---

## 5. The measured comparison

Error against the true surface (nm), synthetic image, four kinds of striping.
Lower is better; the raw image's error is the stripe rms itself.

| | random offsets | 12 bad lines | periodic banding | slow drift |
|---|---|---|---|---|
| raw | 0.99 | 1.38 | 1.41 | 1.99 |
| `align_rows` (median of differences) | 0.78 | **1.01** | 1.09 | 1.99 |
| MDSR σ=2 | 1.82 | 1.81 | 1.81 | 2.11 |
| MDSR σ=5 (default) | 2.53 | 2.52 | 2.52 | 2.75 |
| GSR μ1=1/6 | 1.52 | 1.45 | 1.53 | 2.20 |
| GSR default | 2.43 | 2.40 | 2.42 | 2.75 |
| DeStripe (default) | **0.74** | 1.15 | **0.53** | **1.04** |

Line-to-line jump (nm) — how striped it *looks* — same runs:

| | random offsets | 12 bad lines | periodic banding | slow drift |
|---|---|---|---|---|
| raw | 1.12 | 0.86 | 0.98 | 0.47 |
| MDSR (any σ) | **0.01** | **0.02** | **0.01** | **0.04** |
| GSR default | 0.10 | 0.10 | 0.10 | 0.13 |
| DeStripe (default) | 0.57 | 1.05 | 0.48 | 0.43 |

Read these two tables together. **MDSR and GSR win the appearance metric by
two orders of magnitude and lose the fidelity metric**; DeStripe does the
reverse. On this test image the row means carry as much real signal as
stripes (1.07 nm vs 0.99 nm), which is the hardest possible case for the
aggressive methods — on a genuinely flat sample with heavy striping the
ranking would move towards MDSR and GSR.

Note also the `align_rows` row. The plain median-of-differences already in
the GUI beats both aggressive methods on all four cases and wins outright on
the isolated bad lines: **if your stripes really are per-line offsets and
nothing else, use it** and keep the sophistication for what it cannot
handle — which is visible in the last column, where a smooth drift leaves it
with exactly the error it started with (1.99 → 1.99), because a slow drift
between neighbouring rows is indistinguishable from a real slope and the
median of differences correctly refuses to touch it.

On the real scan (line-to-line 9.49 nm before):

| | line-to-line after | removed rms | time |
|---|---|---|---|
| MDSR σ=5 | 0.82 | 33.6 nm | 63 ms |
| GSR default | 0.40 | 35.9 nm | 1.1 s |
| GSR μ1×2 | 0.32 | 43.1 nm | 1.1 s |
| DeStripe k=1 | 9.31 | 15.2 nm | 26 ms |
| DeStripe k=0.5 | 4.02 | 24.3 nm | 31 ms |

That image has strong real structure in its row means (a milled step), which
is why the aggressive methods remove 30–43 nm rms and DeStripe at the default
threshold barely moves the line-to-line number. There is no "right" row in
that table — it depends entirely on whether those 30 nm are artifact or
surface, which only you can decide.

---

## 6. Choosing

1. **Level the image first.** Plane fit at minimum. Everything below assumes
   it.
2. **Are the stripes pure per-line offsets?** Look at the image: if each row
   is uniformly shifted, `align_rows` with the median of differences is the
   cheapest and most faithful answer. Try it before any of the three.
3. **Do you know the direction, and is the striping heavy and broadband?**
   → **MDSR**. Start at σ=1–2 on a 512-px scan, `levels` 5, and raise σ only
   while stripes are still visible. Watch the mask panel.
4. **Are the stripes irregular — a few bad lines, varying width, short
   trails?** → **GSR**, which is the only one that can treat parts of the
   image differently. Preview at 600 iterations, sweep `mu1` over a factor
   of 4, raise iterations to 2000+ for the final apply.
5. **Do you not know the direction, or do you want the most conservative
   result?** → **DeStripe**. Run it at the default; if nothing changes, lower
   `cvar_k` in steps of 0.25. Never lower `min_run`.
6. **Always check the removed panel.** It is the honest output. If it
   contains a shadow of your surface features, the filter is eating
   topography — back off. If it contains a regular sinusoidal ripple across
   the whole frame (typical of DeStripe and of narrow MDSR grooves), you are
   removing individual frequencies rather than stripes, which may or may not
   be what you want.
7. **Use the parameter sweep** rather than typing values one at a time. It
   runs a grid over two parameters of the current method and shows the
   results side by side, computed on the whole image and cropped to the area
   you dragged.

## 7. What none of this fixes

Stripes that come from a *changing* tip — the tip picks something up halfway
through the scan and the surface is genuinely measured differently before and
after — are not additive noise, and no `u0 = u + s` model can undo them. The
same holds for parachuting, which is a feedback failure and is handled by the
two-way merge instead. If the forward and backward scans disagree in a
region, that is the signal to reach for `gwy_twoway`, not for a destriper.
