# The 3D view, and how to learn it

`gwy_3d_viewer.py` is a separate program from the processing GUI. It draws
one AFM channel as a surface on the GPU, and it can hand that same surface to
Blender for a path-traced still. This is a guide to what the pieces are, why
they were chosen, and where to read more.

```bash
python gwy_3d_viewer.py
```

It takes a file and a channel on the command line too, so it can be launched
from somewhere else later:

```bash
python gwy_3d_viewer.py "Data to test/2026-07-21_exp00/scan_0001.gwy" --channel "Height [Fwd]"
```

Requires `pyvista`, `pyvistaqt` and a Qt binding (PySide6 is what it was
written against). The processing GUI does not need any of them.

---

## 1. The three files, and what each one is responsible for

| File | Runs in | Depends on | Does |
|---|---|---|---|
| `gwy_surface.py` | your Python | numpy, pyvista | Reads a channel, holds it, builds the mesh |
| `gwy_3d_viewer.py` | your Python | Qt, pyvista | The window and every control |
| `gwy_segment.py` | your Python | numpy, scipy (scikit-image optional) | Finds the objects, removes the rest |
| `gwy_segment_view.py` | your Python | Qt, numpy | The flat window where the mask is corrected |
| `gwy_blender_export.py` | your Python | Qt, numpy | Writes the scene file, launches Blender |
| `gwy_blender_render.py` | **Blender's Python** | bpy, numpy | Builds the scene, renders it |

The split down the middle matters. The last file never runs in your
interpreter and never imports pyvista; the others never import `bpy`.
They meet at a `.npz` file on disk. Section 4 explains why.

`gwy_segment.py` is deliberately free of both Qt and pyvista, so the same
segmentation can be run from a script or picked up by the processing GUI
later without dragging a renderer along.

---

## 2. Why PyVista and not something else

The requirement was GPU rendering with quality close to Blender's, and a
path to stacking layers and slicing through them later.

**PyVista** is a Python interface to **VTK**, the C++ visualization library
ParaView is built on. It was chosen for three reasons:

- It has the ingredients that make a render look like a render: physically
  based materials (metallic/roughness, the Disney principled model Blender
  also uses), real shadow maps, screen-space ambient occlusion, environment
  reflections and anti-aliasing. Roughly EEVEE-class, in a live viewport.
- It also has the *un*photorealistic shading that scientific images need.
  Eye-dome lighting darkens each pixel by how much closer its neighbours are
  in screen space. On a rough surface where a lit render becomes noise, this
  is what makes the depth ordering readable. Turn it on with the material
  set to Flat.
- Volume rendering, clipping planes and interactive slice widgets have been
  in VTK for over a decade. Stacking layers and cutting through the stack is
  the plan for this tool, and that is the part that is painful to bolt on
  afterwards.

The alternatives that were considered and why they lost:

- **pygfx** — a three.js-style engine on WebGPU. Better architecture, cleaner
  API, PBR and shadows, and genuinely the future. But it is pre-1.0 and its
  own roadmap says the API is still changing on purpose. Worth revisiting.
- **VisPy** — very fast, but a drawing library rather than a rendering one:
  no PBR, no shadow maps, no AO. You would write the shaders yourself.
- **napari** — Qt + VisPy, and its whole data model is *layers you stack with
  sliders that slice*, which is exactly the deferred requirement. But its
  rendering quality is VisPy-level. The opposite trade.
- **Blender alone** — highest quality by definition, but `bpy` can be
  imported only once per process and has no viewport to embed. It cannot be
  the interactive view. It is used here as the second half, not the first.
- **matplotlib `plot_surface`** — CPU only, no real depth sorting. Fine for a
  quick look, not for this.

---

## 3. The two ideas in `gwy_surface.py` worth understanding

**The mesh is not in metres.** An AFM frame is microns across and nanometres
tall. Rendered at true proportion it is a flat sheet, and coordinates around
1e-6 waste the depth buffer's precision on nothing. So the mesh is built in
*frame units*: the longer side of the scan is 1.0. The physical values ride
along as the `height` point array in whatever SI prefix reads best, and that
array — never the geometry — is what the colour map and the scale bar use.
Geometry normalised, numbers real.

**Exaggeration is a number on the record.** Every 3D SPM view stretches Z.
This one shows the factor it is using, `1.0` always means true proportion,
and the `Auto` button picks the factor that makes the relief a quarter of the
frame width. The Z axis is labelled `z [nm] x15.87` so a reader of the figure
knows.

Changing the exaggeration moves the Z coordinate of the existing points
rather than rebuilding the mesh, which is what keeps the slider smooth on a
megapixel scan:

```python
points = self.mesh.points
points[:, 2] = self._relief * exaggeration
self.mesh.points = points
```

---

## 4. How the Blender half actually works

This is the part that is not obvious, so here it is in full.

**Blender contains a complete Python interpreter**, and inside it the module
`bpy` *is* Blender — every object, material, light and render setting is
reachable from Python. You do not install Blender into your Python; you hand
a script to Blender's Python:

```bash
blender --background --python gwy_blender_render.py -- scene.npz out.png
```

- `--background` means no window: build, render, write the file, exit.
- `--python FILE` runs that file in Blender's interpreter.
- Everything after the bare `--` is ignored by Blender and left in
  `sys.argv` for the script. That is the only way to pass it arguments.

So the two programs never share an interpreter. `gwy_blender_export.py`
writes `scene.npz` — the height grid, the vertex colours as RGB, and the
render settings as a JSON string — and runs the command above as a
subprocess, streaming Blender's output into the dialog's log.

**Why not `pip install bpy`?** It exists on PyPI, but each release is built
against one exact CPython version (Blender 4.x → Python 3.11). This project
runs on 3.13, so it would not import at all, and pinning a working scientific
environment to a specific minor Python to get a renderer is a bad trade. The
subprocess approach also works with whatever Blender is already installed,
and needs nothing installed if you only ever use "Save scene only".

**The useful trick for learning:** press **Open in Blender** instead of
Render. That runs the same script *without* `--background`, so Blender's own
window opens with the mesh, the material node tree, the three area lights and
the camera already built. Everything the script did is sitting there in the
interface to be inspected, tweaked and copied. This is by far the fastest way
to understand `gwy_blender_render.py`.

Two details in that script that are easy to get wrong:

- **Colour space.** Blender shades in linear light. Handing it sRGB values
  from a matplotlib colour map washes the gradient out, so `srgb_to_linear`
  converts first.
- **Winding.** A quad's corners must go counter-clockwise seen from above or
  the face normal points down and the surface renders black. The order is
  `[i, i+1, i+nx+1, i+nx]`.

Blender does not have to be installed for anything else in the viewer to
work. If none is found, the dialog still writes the `.npz` and prints the
exact command to run against it later. Set `GWY_BLENDER` to override the
search for the executable.

---

## 5. What the controls do

**Height** — the exaggeration factor. `True 1x` is physical proportion and
will usually look almost flat, which is the honest answer. `Auto` makes the
relief a quarter of the frame width.

**Colour** — a Gwyddion gradient from `gwy_colormaps`, over a range set by
dragging on the histogram. Drag a handle to move one end, drag between them
to slide the window, double-click to reset. The strip under the histogram
shows the gradient *as it will be applied*, including the flat bands at each
end where values clip — which is the part that is easy to set too tight
without noticing. The `Keep` buttons set the range by percentile.

**Material** — metallic and roughness. Not decoration: the material decides
how much fine texture survives. Matte hides it, Satin picks it out, Polished
metal turns the surface into a mirror of the synthetic environment.

**Light** — on a flat sample this matters more than the material does.

The rig is always the same three lights: a *key* that does the shading, a
*fill* from the other side that keeps its shadows from going black, and a
*rim* from behind that separates the surface from the background. You place
the key and the other two follow it.

- *Direction* (0–359°) — where the key stands going round the sample. This
  decides which slopes are lit and which way the shadows fall. Sweep it when
  a feature is hard to see; something usually pops out.
- *Height* (0–90°) — how high the key stands. **This is the important one.**
  Low is a raking light: long shadows, every bump visible, which is how you
  photograph a surface you want to show relief on. High flattens it.
- *Key / Fill / Rim* — how hard each burns, in percent. Fill at 0 gives hard
  dramatic shadows; turn it up to see into them. Setting a light to 0 removes
  it from the scene rather than leaving a dark one there.
- *Cast shadows* — real shadow maps from these lights. Costs frame rate, and
  needs some relief and a low enough *Height* to show at all.

The preset list is not a separate mode — each preset is just those five
numbers, and it fills the sliders in. Move any slider and the list says
`Custom`, because it no longer describes the scene. Two presets are
different in kind and grey out the angles: *Soft (light kit)* is VTK's own
five-light rig, and *From the camera* is a headlight that always points where
you are looking, which gives even illumination and no visible shadows.

**Segment** — see section 5a. This panel only chooses how the answer is
shown; the answer itself is found and corrected in a separate flat window.

**Quality**

- *Ambient occlusion (SSAO)* — darkens crevices by how enclosed they are.
  The cheapest thing that makes a rough surface read as three-dimensional.
  Leave it on.
- *Eye-dome lighting* — non-photorealistic depth shading. Ugly over a lit
  material, excellent over Flat when you want to read fine relief.
- *Anti-alias* — `fxaa` is cheap and slightly soft; `ssaa` renders larger and
  shrinks, which is much cleaner. Switch to `ssaa` before a screenshot.
- *Environment reflections* — the synthetic studio cube map. Without it, a
  metallic material has nothing to reflect and goes black.

**File → Save screenshot** renders at up to 8× the window size, which is how
you get a figure-resolution image without a path tracer.

**File → Export mesh** writes `.ply` with vertex colours (Blender, MeshLab,
CloudCompare all read it), or `.stl` if you want to 3D-print the scan.

---

## 5a. Keeping only the objects

`Segment → Find objects...`, or Ctrl+E. This exists to answer one question:
*there are things on this surface, show me the things and take the rest
away.*

### Regions have no type

Everything that comes out of this window is just a region. There is no
"cell", no "particle", no "fibre" anywhere in the panel, and that is
deliberate: what a region *is* is a fact about your specimen, and the program
has never seen your specimen. Naming the categories after one sample would be
wrong on the next one.

What the panel offers instead is three **ways of looking**. Any mixture of
them can run, and everything they find lands in one undifferentiated list
that you then keep, drop, erase or correct. The regions are drawn in cycling
colours by number, which say nothing about what they are — they only make one
region readable against the one next to it.

### Outlines — the one that ignores height

Reach for this first, because **height is a bad way to find an object**.

Threshold the heights and anything whose parts sit at different levels loses
a bite out of itself. An object tilted in the scan. An object with a dip in
the middle. Two objects of the same kind at different heights on an uneven
field. Levelling makes it worse rather than better: row alignment gives every
scan line its own baseline, so a row that is mostly object is pushed down
towards the field it sits on.

So this detector never looks at a height. It looks at where the height
*changes*. The gradient magnitude is large along the rim of an object and
small anywhere the surface is merely smooth — whatever level that smooth
surface happens to sit at — and thresholding it gives a set of walls. What
falls out is the picture your eye already sees: the frame is divided into
patches by those walls, and each patch is a candidate.

The numbers, measured on a yeast scan with the top half of one object shifted
upwards by up to four times the object's own height range:

| | kept, no shift | kept, shifted by 4× |
|---|---|---|
| Outlines | 98 % of the object | 98 % |
| Otsu threshold on blurred heights | 69 % | **49 %** — it lost exactly the half that moved |

Two filters then say which patches are objects.

**Size** is the obvious one — `Smallest` and `Largest`, each as the side of a
square.

**Smoothness** is the useful one. A patch that is smooth inside is a thing;
the field it sits on is textured, and its texture is what broke the field
into patches in the first place. The test compares the median edge strength
inside a patch against the median over the whole frame, so it is the image
compared with itself and carries no absolute number. On the yeast scan the
objects came out at 0.5–0.6 and the field at 1.1–1.3 — a wide enough gap that
the setting is not delicate. `0` turns the test off.

One thing to know about it: the comparison is against the median over the
whole frame, so where the field covers most of the frame *the field is the
median* and reads about 1.0. That is why the default sits at **0.8** and not
at 1.0 — checked both ways round, on a frame where the objects cover 55 % and
one where they cover 7 %, 0.7–0.8 worked for both.

Finally the walls are shared out. A wall is the rim of the object it bounds
and belongs to it, so the patches are grown back into the wall by a watershed
on the edge strength; where two objects share a rim, the two floods meet in
the middle of it and each gets its half.

### Ridges — for the things that have no inside

A thin line has no interior for the outline detector to find, and it cannot
be found by height either: it is often no taller than the texture around it,
so any threshold that catches it catches the texture too. On the yeast scans,
thresholding the residual marked 5–9 % of the frame and the real lines were a
small part of that.

What separates them is **curvature**. A line is curved sharply across itself
and not at all along itself; a grain is curved both ways. That is exactly the
difference the Hessian — the matrix of second derivatives — sees. The
detector blurs at the width of a line, takes the more negative of the two
eigenvalues at every pixel, and thresholds *that* instead of the height.

Then **continuity** finishes the job, which is the part that matches what you
can see by eye. Every marked region is measured along its own principal axis
and any that does not run at least `Length` is dropped. A grain that survived
the curvature test is a few tens of nanometres across and goes; a line runs
for a micron and stays. The measurement is a real principal-axis span and not
an area or a bounding box, because a compact blob of the same area would pass
an area test and a diagonal line and a square patch have nearly the same
bounding box.

*The single most useful control here is `Width`.* At half a line's width the
response fills with texture; at twice it neighbouring lines smear together.
Measured on the synthetic case: a 5-pixel line was 12 % recovered at a
quarter of its width and 98 % at the right one. Reach for that one first when
the result is wrong.

### Raised areas — the local comparison

The one detector that does look at height, but only ever at a *local*
difference: the heights minus a blurred copy of themselves, thresholded in
robust sigmas of that residual. Because the comparison is with the immediate
surroundings and not with a level, it survives a tilted or uneven field,
which a plain threshold does not.

### Separating things that have grown together

Objects packed on a surface touch, and no threshold sees the seam — you would
then have to keep all of them or none. `Separate` cuts them apart on shape:
the distance to the nearest edge peaks in the middle of each object and
collapses at the neck between two, so a watershed run downhill from those
peaks puts the boundary at the neck. On the yeast scan this is the difference
between **one region and eleven**.

This is done one region at a time rather than over the whole map at once, and
that is not an implementation detail. The flood quantises to 256 levels; over
a whole frame the deepest basin anywhere sets that scale, so on a map holding
one large region and a hundred small ones every neck between the small ones
collapses into a single level and one basin takes the lot — measured: ten
seeds found, one region returned.

For the same reason `scikit-image` is **optional but recommended**. Its
watershed works on the values themselves rather than on 256 grey levels; the
`scipy` fallback left the largest six objects on this scan fused into one
region of 26.6 µm². The window says so in its status line when it is running
without it.

### Sizes are percentages of the frame

Every length in that panel is a percentage of the frame width, and the number
beside it says what that currently works out to — `0.90 % = 63 nm`. Two
reasons. These scans are 1024 × 512 pixels over a square 7 × 7 µm frame, so a
pixel is twice as tall as it is wide, and a filter given one radius in pixels
would be measuring something different along x than along y; the conversion
is done per axis. And a default in pixels stops meaning anything when the
scan size changes, while "one percent of the frame" survives it.

### Correcting it

The detector proposes; you dispose. **Pick** clicks a region on or off,
**Brush** paints in something that was missed and **Erase** paints out
something that should not have been found. The brush is a circle in
*physical* units, so it stays round on a scan whose pixels are not square.

**Box** drags a rectangle, and what that does is the combo next to the tool
buttons:

| Box action | What it does |
|---|---|
| **Find objects inside** | runs the detectors over that rectangle *alone* |
| Keep what is inside | keeps every region the rectangle touches |
| Drop what is inside | drops them — they stay on the map, greyed out |
| Erase what is inside | takes the rectangle off the map entirely, regions and painting both |

**Find objects inside** is the way out of a bad automatic result, and it is
worth understanding why it works. Every threshold in every detector is a
robust statistic of what it is looking at; run over the whole frame, those
statistics are dominated by whatever covers most of it. Measure them inside a
rectangle instead and they describe that rectangle. So an object that the
frame-wide settings drowned usually comes straight out once the box is drawn
round it — no setting has to change. Whatever was in the box before is
replaced, so a second attempt over the same place corrects its own last
answer instead of piling a new set of regions on top of it, and a region that
only reaches into the box is trimmed rather than removed.

A useful pattern on a difficult scan: **Erase all**, then draw a box round
each thing you actually want. You end up with exactly the objects you meant
and nothing else.

**Dropping and erasing are different, on purpose.** Dropped says "found it,
do not want it": the region stays on the screen greyed out and one more click
changes its mind. Erased says "that is not a thing" and takes it off the map.
Use drop while you are still deciding and erase once you are not. **Erase
dropped** clears out everything you have already said no to.

Painting is kept separately from the regions, so changing a setting and
pressing **Detect** again replaces the regions and leaves the hand
corrections in place. That is the order the work actually happens in: detect,
look, fix the obvious mistakes, adjust a number, look again.

**Grey image** is worth knowing about: an AFM gradient is warm and light
nearly everywhere, which is the hardest possible background for a coloured
overlay. In grey the region colours are unmistakable.

### What the 3D window then does with it

- **Fade the rest** — the discarded part gets an opacity. `Rest 0` leaves the
  objects floating on their own. Heights are not touched at all; this only
  changes what you can see through.
- **Flatten the rest to the background** — the discarded part is replaced by
  a background surface estimated *from those same discarded pixels*, so the
  substrate keeps its large-scale shape and loses its texture. `Smooth 0`
  gives one flat level instead, which is the honest answer when the substrate
  really is flat.

In both cases the kept pixels are untouched — not smoothed, not re-levelled,
not shifted. What was measured on a cell is still what is drawn on the cell.
A pixel that was never measured stays NaN rather than being filled in with a
plausible-looking background, which is the one thing a tool that removes
things must not do.

Flattening changes the mesh, so `File → Export mesh` and the Blender render
both carry it. Fading is a property of the screen and does not.

Two things worth knowing:

- Fading is drawn by handing VTK explicit per-point RGBA rather than an
  opacity array. The documented opacity-array route was tried and does not
  work here — the surface came out flat grey and fully opaque, because the
  actor was still being sorted into the opaque pass. The cost of the working
  route is that the colour map is no longer live, so changing the gradient or
  the range rebuilds the actor.
- **The flat window draws row 0 at the bottom, to match the 3D view.** The
  processing GUI draws its images the other way up, so the same scan is a
  vertical mirror of itself between the two programs. That is a property the
  3D viewer already had; this window matches the view it exists to control.

### From a script

```python
import gwy_segment as gs, gwy_surface

s = gwy_surface.load('scan.gwy', 'Height [Fwd]')

# the whole frame, outlines only - nothing here depends on a height
seg = gs.segment(s.z, s.x_real, s.y_real, methods=('outline',), smoothness=0.8)
seg.erase([i for i, row in enumerate(seg.measure()) if i and row['area'] < 1e-12])
clean = gs.flatten(s.z, seg.mask(), seg.scale)

# or: start from nothing and search inside one rectangle at a time
seg = gs.empty(s.z, s.x_real, s.y_real)
for box in [(60, 260, 60, 360), (300, 480, 400, 700)]:
    found = gs.segment(s.z, s.x_real, s.y_real, window=box)
    seg.absorb(found, (slice(box[0], box[1] + 1), slice(box[2], box[3] + 1)))
```

---

## 6. Where to read more

**PyVista / VTK**

- PyVista examples gallery — <https://docs.pyvista.org/examples/> — the
  fastest way in; nearly every control in this viewer came from one page.
- Physically based rendering — <https://docs.pyvista.org/examples/02-plot/pbr>
- Ambient occlusion — <https://docs.pyvista.org/examples/02-plot/ssao>
- Eye-dome lighting — <https://docs.pyvista.org/examples/02-plot/edl>
- Shadows and lights — <https://docs.pyvista.org/examples/04-lights/>
- `pyvistaqt` (embedding in Qt) — <https://qt.pyvista.org/>
- The PyVista tutorial (a full course) — <https://tutorial.pyvista.org/>
- VTK's own book, for the model underneath — <https://book.vtk.org/>

**For the layers-and-slicing work later**

- Volume rendering — <https://docs.pyvista.org/examples/02-plot/volume_rendering>
- Clipping with a plane widget — <https://docs.pyvista.org/examples/03-widgets/plane_widget>
- Orthogonal slices — <https://docs.pyvista.org/examples/01-filter/slice>
- napari, if the layer model turns out to matter more than the shading —
  <https://napari.org/stable/getting_started/layers.html>

**Blender scripting**

- Blender Python API — <https://docs.blender.org/api/current/>
- Command-line arguments (`--background`, `--python`, the `--` separator) —
  <https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html>
- `bpy.types.Mesh.from_pydata` — <https://docs.blender.org/api/current/bpy.types.Mesh.html>
- Principled BSDF, so the material node tree makes sense —
  <https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html>
- Cycles settings from Python — <https://docs.blender.org/api/current/bpy.types.CyclesRenderSettings.html>
- Blender as a Python module, if you ever want the other direction —
  <https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html>

**The segmentation, if you want to go further**

- `scipy.ndimage`, which is nearly all of `gwy_segment.py` —
  <https://docs.scipy.org/doc/scipy/reference/ndimage.html>
- Ridge (Hessian) filters, the idea the ridge detector is built on — Frangi
  et al., *Multiscale vessel enhancement filtering*, and scikit-image's
  version of it: <https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html>
- Watershed, which appears twice here — sharing a wall out between the
  objects either side of it, and cutting apart two things grown together:
  <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html>
- Watershed *segmentation by edges* specifically, which is what the outline
  detector is a simple form of —
  <https://scikit-image.org/docs/stable/auto_examples/applications/plot_coins_segmentation.html>
- Gwyddion's own grain analysis, for what a mature version of this looks
  like — <https://gwyddion.net/documentation/user-guide-en/grain-analysis.html>

**Qt**

- Qt for Python (PySide6) — <https://doc.qt.io/qtforpython-6/>
- `QPainter`, which is all the histogram widget is —
  <https://doc.qt.io/qtforpython-6/PySide6/QtGui/QPainter.html>
- `QProcess`, which is how Blender is run without freezing the window —
  <https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html>

**The alternative engines, if the choice is ever revisited**

- pygfx — <https://pygfx.org/> and its roadmap posts
- VisPy — <https://vispy.org/>
- fastplotlib — <https://fastplotlib.org/>

---

## 7. Not implemented yet, on purpose

Stacking several channels or several scans into a volume and slicing through
it. The pieces are arranged for it — `Surface` is one field with its own
extent and its own Z offset, which is what a stack would be built from, and
VTK already has the volume mapper and the plane widgets — but none of it is
written.
