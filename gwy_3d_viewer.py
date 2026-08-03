"""
A standalone 3D view of one AFM channel, drawn on the GPU.

Run with:  python gwy_3d_viewer.py [file.gwy] [--channel "Height [Fwd]"]

This is a separate program from `gwy_processor_gui.py`. It shares the file
reader and the colour gradients and nothing else: no pipeline, no undo, no
processing. Open a .gwy, pick a channel, and look at it as a surface.

Why a second window and not a tab in the first one. The main GUI is Tkinter,
and Tkinter has no way to host a GPU drawing surface - there is no OpenGL
widget in it, and the third-party ones are not something to build a tool on.
Anything that renders on the GPU has to live in Qt, so it lives in its own
process. That is a smaller loss than it sounds: a 3D view is a different
task from a processing pipeline, and it is genuinely useful to have both on
screen at once.

What it is built on. PyVista over VTK, which is the same renderer ParaView
uses. The reasons it was picked over the alternatives:

  * The look is not fixed. VTK does physically based rendering with a
    metallic/roughness material, real shadow maps, screen-space ambient
    occlusion and an environment to reflect - the same ingredients Blender's
    EEVEE uses, and enough to make a surface read as a surface rather than
    as a coloured sheet.
  * It also does the *un*photorealistic shading that scientific images
    actually need. Eye-dome lighting darkens a pixel by how much nearer its
    neighbours are, which makes depth ordering readable on a rough surface
    where a lit render turns to noise.
  * Volumes, clipping planes and interactive slice widgets are a decade-old
    part of it rather than a roadmap item. Stacking layers and cutting
    through the stack is the plan for this tool later, and that is the piece
    that is hard to add afterwards.

Live rendering can only go so far, so `gwy_blender_export` hands the same
surface to Blender for a path-traced still. The viewer is for looking; that
is for the figure that goes in the paper.

The controls, and why each one is there:

  Height        The exaggeration of Z, as an explicit factor, with 1.0 the
                true physical proportion. AFM frames are microns across and
                nanometres tall, so every 3D SPM view exaggerates; this one
                shows the number it is using and lets it be set.

  Colour        The Gwyddion gradients from `gwy_colormaps`, over a range
                chosen by dragging on the histogram - the same way the clip
                dialog in the main GUI works, for the same reason: the
                distribution is where you can see what a limit will cost.
                The colour range and the height exaggeration are separate
                controls and moving one never moves the other.

  Material      Metallic and roughness, plus presets. Worth having because
                the material decides how much of the fine texture survives:
                a matte surface hides it, a satin one picks it out.

  Light         Where the key light stands, as an angle round the sample and
                an angle above it, plus how hard it and its two companions
                burn. On a flat sample this matters more than the material
                does: a low raking light throws a long shadow off every bump
                and makes relief legible that an overhead light erases. The
                presets are only sets of those same numbers, so moving a
                slider is never fighting a mode.

  Segment       Which parts of the scan are objects and which are the field
                they sit on, and what to do with the field: fade it, hide
                it, or flatten it away and leave the objects standing. The
                finding and the correcting happen in a flat 2D window
                (`gwy_segment_view`), because deciding whether a mark is one
                object or two is a question about the map and not about the
                surface; this panel only chooses how the answer is shown.

  Quality       Ambient occlusion, eye-dome lighting and anti-aliasing, each
                on its own switch, because the right combination depends on
                the sample and on whether the image is for looking at or for
                printing.
"""

import os
import sys

# qtpy takes the first Qt binding it can import and PyQt5 usually wins that
# race. Say so explicitly when PySide6 is installed, which is what this was
# written against, but leave an existing choice alone.
if 'QT_API' not in os.environ:
    try:
        import PySide6  # noqa: F401
        os.environ['QT_API'] = 'pyside6'
    except ImportError:
        pass

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import pyvista as pv
from pyvistaqt import QtInteractor

import gwy_colormaps
import gwy_surface


APP_NAME = 'GWY 3D view'


# ------------------------------------------------------------------ scene

def studio_cubemap(size=64):
    """A synthetic studio environment for the PBR shader to reflect.

    Physically based materials need something around them: with no
    environment, metal has nothing to be metallic *about* and comes out
    black. A real HDRI would be better and can be dropped in later, but it
    is a file to ship and a file to lose, so the default is built here - a
    soft overhead source, walls that fall off towards the floor.
    """
    faces = []
    # +X, -X, +Y, -Y, +Z, -Z, in VTK's order.
    ramps = [
        (0.55, 0.80), (0.45, 0.72),     # side walls
        (0.95, 1.00),                   # the bright one: overhead softbox
        (0.18, 0.26),                   # the dark one: floor
        (0.50, 0.78), (0.42, 0.70),     # front and back
    ]
    for lo, hi in ramps:
        ramp = np.linspace(hi, lo, size, dtype=np.float32)[:, None]
        # A trace of blue at the top and warmth at the bottom keeps a grey
        # surface from looking dead without tinting it noticeably.
        rgb = np.empty((size, size, 3), dtype=np.float32)
        rgb[..., 0] = ramp * 0.98
        rgb[..., 1] = ramp * 0.99
        rgb[..., 2] = ramp * 1.00
        img = pv.ImageData(dimensions=(size, size, 1))
        img.point_data['env'] = np.clip(
            rgb.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
        img.set_active_scalars('env')
        faces.append(img)
    return pv.Texture(faces)


#: name -> (metallic, roughness, physically based, diffuse, specular)
MATERIALS = {
    'Matte':            (0.00, 0.90, True,  1.00, 0.00),
    'Satin':            (0.10, 0.55, True,  1.00, 0.10),
    'Polished metal':   (0.85, 0.25, True,  1.00, 0.20),
    'Ceramic':          (0.00, 0.35, True,  1.00, 0.30),
    'Flat (no shading)': (0.00, 1.00, False, 1.00, 0.00),
}

#: name -> (background colour, top colour or None for flat)
BACKGROUNDS = {
    'Dark studio':  ('#1b1f27', '#39414f'),
    'Charcoal':     ('#2b2b2b', None),
    'Black':        ('#000000', None),
    'Paper white':  ('#ffffff', None),
    'Light grey':   ('#d9dde3', '#f4f6f9'),
}

#: The rig is always the same three lights - a key that does the shading, a
#: fill that keeps its shadows from going black, and a rim that separates the
#: surface from the background. Only where they point and how hard they burn
#: changes, so a preset is nothing more than five numbers and the sliders and
#: the presets can be the same control. `None` means "leave the sliders where
#: the user put them".
LIGHT_PRESETS = {
    'Custom': None,
    'Studio (3 point)': dict(mode='rig', azimuth=135, elevation=40,
                             key=95, fill=35, rim=45),
    'Raking (grazing)': dict(mode='rig', azimuth=110, elevation=12,
                             key=120, fill=15, rim=25),
    'Overhead': dict(mode='rig', azimuth=90, elevation=78,
                     key=100, fill=30, rim=20),
    'Soft (light kit)': dict(mode='lightkit'),
    'From the camera': dict(mode='headlight', key=100),
}

CAMERAS = {
    'Isometric': None,          # handled by view_isometric()
    'Top': 'xy',
    'Front': 'xz',
    'Side': 'yz',
}

#: What the segmentation does to the scene. The three answers to "and now
#: remove the rest" that are worth having: leave it there, fade or hide it,
#: or replace it with the background it is sitting on.
SEGMENT_MODES = (
    'Off',
    'Fade the rest',
    'Flatten the rest to the background',
)


# -------------------------------------------------------------- histogram

def _vtk_text(text):
    """VTK's built-in fonts have no micro sign and drop it silently."""
    return text.replace('µ', 'u')


def cmap_colors(cmap, n=256):
    """`n` QColors sampled along a matplotlib colour map."""
    rgba = np.asarray(cmap(np.linspace(0.0, 1.0, n)))
    rgb = np.clip(rgba[:, :3] * 255.0, 0, 255).astype(int)
    return [QtGui.QColor(int(r), int(g), int(b)) for r, g, b in rgb]


class HistogramWidget(QtWidgets.QWidget):
    """The colour range, set by dragging on the distribution of heights.

    The two limits are drawn as handles on the histogram and the colour
    strip underneath shows what the gradient will do with them - including
    the flat bands at each end where values are clipped, which is the part
    that is easy to set too tight without noticing.

    Emits `rangeChanged(vmin, vmax)` continuously while dragging, so the
    render follows the mouse. Double-click resets to the full data range.
    """

    rangeChanged = QtCore.Signal(float, float)

    MARGIN = 8
    STRIP = 14          # height of the colour strip
    GRAB = 6            # how close the mouse must be to catch a handle
    TEXT = 15           # room for the two value labels

    def __init__(self, parent=None):
        super(HistogramWidget, self).__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self._counts = np.zeros(1)
        self._edges = np.array([0.0, 1.0])
        self._lo, self._hi = 0.0, 1.0
        self._vmin, self._vmax = 0.0, 1.0
        self._colors = cmap_colors(gwy_colormaps.get())
        self._unit = ''
        self._drag = None
        self._drag_ref = None

    def sizeHint(self):
        return QtCore.QSize(300, 140)

    # ---- what to show ----

    def set_distribution(self, counts, edges):
        self._counts = np.asarray(counts, dtype=np.float64)
        self._edges = np.asarray(edges, dtype=np.float64)
        self._lo, self._hi = float(edges[0]), float(edges[-1])
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0
        self.update()

    def set_limits(self, vmin, vmax, emit=False):
        vmin, vmax = float(vmin), float(vmax)
        if vmax <= vmin:
            vmax = vmin + abs(vmin) * 1e-6 + 1e-12
        self._vmin, self._vmax = vmin, vmax
        self.update()
        if emit:
            self.rangeChanged.emit(vmin, vmax)

    def limits(self):
        return self._vmin, self._vmax

    def set_colormap(self, cmap):
        self._colors = cmap_colors(cmap)
        self.update()

    def set_unit(self, unit):
        self._unit = unit or ''
        self.update()

    # ---- geometry ----

    def _plot_rect(self):
        m = self.MARGIN
        h = self.height() - 2 * m - self.STRIP - self.TEXT
        return QtCore.QRect(m, m, max(1, self.width() - 2 * m), max(1, h))

    def _strip_rect(self):
        p = self._plot_rect()
        return QtCore.QRect(p.left(), p.bottom() + 2, p.width(), self.STRIP)

    def _x_of(self, value):
        p = self._plot_rect()
        t = (value - self._lo) / (self._hi - self._lo)
        return p.left() + t * p.width()

    def _value_of(self, x):
        p = self._plot_rect()
        t = (x - p.left()) / float(max(1, p.width()))
        return self._lo + t * (self._hi - self._lo)

    # ---- painting ----

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        plot = self._plot_rect()
        strip = self._strip_rect()
        pal = self.palette()

        painter.fillRect(self.rect(), pal.window())
        painter.fillRect(plot, pal.base())

        self._paint_bars(painter, plot)
        self._paint_strip(painter, strip)
        self._paint_handles(painter, plot, strip)
        self._paint_labels(painter, strip)

        # An outline, not a fill: drawRect paints with the current brush and
        # the handles left a solid one behind.
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(pal.mid().color(), 1))
        painter.drawRect(plot.adjusted(0, 0, -1, -1))
        painter.end()

    def _paint_bars(self, painter, plot):
        counts = self._counts
        if counts.size == 0 or counts.max() <= 0:
            return
        # Square root rather than linear: an AFM height histogram is usually
        # one tall peak and a long thin tail, and the tail is the part that
        # matters when choosing a limit.
        scaled = np.sqrt(counts / counts.max())
        n = len(scaled)
        w = plot.width() / float(n)

        inside = QtGui.QColor(94, 140, 200)
        outside = QtGui.QColor(120, 124, 132)
        for i, v in enumerate(scaled):
            if v <= 0:
                continue
            x0 = plot.left() + i * w
            centre = 0.5 * (self._edges[i] + self._edges[i + 1])
            colour = inside if self._vmin <= centre <= self._vmax else outside
            h = v * plot.height()
            painter.fillRect(
                QtCore.QRectF(x0, plot.bottom() - h, max(1.0, w), h), colour)

    def _paint_strip(self, painter, strip):
        """The gradient as the render will apply it, clipping included."""
        if strip.width() <= 0:
            return
        n = len(self._colors)
        span = self._vmax - self._vmin
        for px in range(strip.width()):
            value = self._value_of(strip.left() + px)
            t = 0.0 if span <= 0 else (value - self._vmin) / span
            idx = int(np.clip(t, 0.0, 1.0) * (n - 1))
            painter.fillRect(strip.left() + px, strip.top(), 1, strip.height(),
                             self._colors[idx])
        painter.setPen(QtGui.QPen(self.palette().mid().color(), 1))
        painter.drawRect(strip.adjusted(0, 0, -1, -1))

    def _paint_handles(self, painter, plot, strip):
        shade = QtGui.QColor(0, 0, 0, 60)
        xmin, xmax = self._x_of(self._vmin), self._x_of(self._vmax)
        # Dim what falls outside the range.
        if xmin > plot.left():
            painter.fillRect(QtCore.QRectF(plot.left(), plot.top(),
                                           xmin - plot.left(), plot.height()),
                             shade)
        if xmax < plot.right():
            painter.fillRect(QtCore.QRectF(xmax, plot.top(),
                                           plot.right() - xmax, plot.height()),
                             shade)

        pen = QtGui.QPen(QtGui.QColor(220, 60, 60), 2)
        for x in (xmin, xmax):
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(x, plot.top()),
                             QtCore.QPointF(x, strip.bottom()))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(220, 60, 60)))
            painter.setPen(QtCore.Qt.NoPen)
            tri = QtGui.QPolygonF([
                QtCore.QPointF(x - 4, plot.top()),
                QtCore.QPointF(x + 4, plot.top()),
                QtCore.QPointF(x, plot.top() + 6),
            ])
            painter.drawPolygon(tri)
        painter.setBrush(QtCore.Qt.NoBrush)

    def _paint_labels(self, painter, strip):
        painter.setPen(self.palette().windowText().color())
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        y = strip.bottom() + 1
        box = QtCore.QRect(strip.left(), y, strip.width(), self.TEXT)
        painter.drawText(box, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                         self._format(self._vmin))
        painter.drawText(box, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                         self._format(self._vmax))
        painter.drawText(box, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                         'span %s' % self._format(self._vmax - self._vmin))

    def _format(self, value):
        span = abs(self._hi - self._lo)
        if span >= 100:
            text = '%.0f' % value
        elif span >= 1:
            text = '%.2f' % value
        else:
            text = '%.3g' % value
        return '%s %s' % (text, self._unit) if self._unit else text

    # ---- mouse ----

    def _hit(self, x):
        if abs(x - self._x_of(self._vmin)) <= self.GRAB:
            return 'min'
        if abs(x - self._x_of(self._vmax)) <= self.GRAB:
            return 'max'
        if self._x_of(self._vmin) < x < self._x_of(self._vmax):
            return 'span'
        return None

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        x = event.position().x() if hasattr(event, 'position') else event.x()
        self._drag = self._hit(x)
        if self._drag == 'span':
            self._drag_ref = (self._value_of(x), self._vmin, self._vmax)

    def mouseMoveEvent(self, event):
        x = event.position().x() if hasattr(event, 'position') else event.x()
        if self._drag is None:
            hit = self._hit(x)
            self.setCursor(QtCore.Qt.SizeHorCursor if hit in ('min', 'max')
                           else (QtCore.Qt.OpenHandCursor if hit == 'span'
                                 else QtCore.Qt.ArrowCursor))
            return
        value = float(np.clip(self._value_of(x), self._lo, self._hi))
        span = self._hi - self._lo
        if self._drag == 'min':
            self.set_limits(min(value, self._vmax - span * 1e-3), self._vmax,
                            emit=True)
        elif self._drag == 'max':
            self.set_limits(self._vmin, max(value, self._vmin + span * 1e-3),
                            emit=True)
        else:
            ref, vmin0, vmax0 = self._drag_ref
            shift = value - ref
            width = vmax0 - vmin0
            new_min = float(np.clip(vmin0 + shift, self._lo, self._hi - width))
            self.set_limits(new_min, new_min + width, emit=True)

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        self.set_limits(self._lo, self._hi, emit=True)


# ----------------------------------------------------------------- window

class Viewer3DWindow(QtWidgets.QMainWindow):
    """The 3D window: a render view, and the controls that drive it."""

    def __init__(self, path=None, channel=None):
        super(Viewer3DWindow, self).__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 880)

        self.surface = None
        self.mesh = None            # gwy_surface.SurfaceMesh
        self.segmentation = None    # gwy_segment.Segmentation, or None
        self.actor = None
        self.scalar_bar = None
        self.path = None
        self._environment = None
        self._loading = False       # guard against control echo while loading

        self._build_ui()
        self._build_menu()

        if path:
            self.open_file(path, channel)
        else:
            self.statusBar().showMessage('Open a .gwy file to begin')

    # ------------------------------------------------------------ layout

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plotter = QtInteractor(central)
        self._set_background('Dark studio')
        self.plotter.add_axes()

        panel = self._build_panel()
        panel.setFixedWidth(370)

        layout.addWidget(self.plotter.interactor, 1)
        layout.addWidget(panel, 0)
        self.setCentralWidget(central)
        self.statusBar()

    def _build_panel(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        inner = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(inner)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(8)

        box.addWidget(self._group_data())
        box.addWidget(self._group_height())
        box.addWidget(self._group_colour())
        box.addWidget(self._group_segment())
        box.addWidget(self._group_material())
        box.addWidget(self._group_light())
        box.addWidget(self._group_quality())
        box.addWidget(self._group_scene())
        box.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    # ---- groups ----

    def _group_data(self):
        group = QtWidgets.QGroupBox('Data')
        form = QtWidgets.QVBoxLayout(group)

        button = QtWidgets.QPushButton('Open file...')
        button.clicked.connect(self.on_open)
        form.addWidget(button)

        self.file_label = QtWidgets.QLabel('(no file)')
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet('color: gray;')
        form.addWidget(self.file_label)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Channel'))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.currentTextChanged.connect(self.on_channel)
        row.addWidget(self.channel_combo, 1)
        form.addLayout(row)
        return group

    def _group_height(self):
        group = QtWidgets.QGroupBox('Height')
        form = QtWidgets.QVBoxLayout(group)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Exaggeration'))
        self.exag_spin = QtWidgets.QDoubleSpinBox()
        self.exag_spin.setDecimals(3)
        self.exag_spin.setRange(1e-3, 1e6)
        self.exag_spin.setKeyboardTracking(False)
        self.exag_spin.valueChanged.connect(self.on_exaggeration_typed)
        row.addWidget(self.exag_spin, 1)
        row.addWidget(QtWidgets.QLabel('x'))
        form.addLayout(row)

        self.exag_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exag_slider.setRange(0, 1000)
        self.exag_slider.valueChanged.connect(self.on_exaggeration_slid)
        form.addWidget(self.exag_slider)

        self.relief_label = QtWidgets.QLabel('-')
        self.relief_label.setStyleSheet('color: gray;')
        form.addWidget(self.relief_label)

        row = QtWidgets.QHBoxLayout()
        for text, tip, slot in (
                ('True 1x', 'True physical proportion - usually very flat',
                 lambda: self.set_exaggeration(1.0)),
                ('Auto', 'Relief a quarter of the frame width',
                 self.auto_exaggeration)):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        form.addLayout(row)
        return group

    def _group_colour(self):
        group = QtWidgets.QGroupBox('Colour')
        form = QtWidgets.QVBoxLayout(group)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(gwy_colormaps.names())
        self.cmap_combo.setCurrentText(gwy_colormaps.current_name())
        self.cmap_combo.currentTextChanged.connect(self.on_colormap)
        form.addWidget(self.cmap_combo)

        self.histogram = HistogramWidget()
        self.histogram.rangeChanged.connect(self.on_range)
        form.addWidget(self.histogram)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(2)
        row.addWidget(QtWidgets.QLabel('Keep'))
        for label, (lo, hi) in (('all', (0.0, 100.0)),
                                ('99.9', (0.05, 99.95)),
                                ('99', (0.5, 99.5)),
                                ('98', (1.0, 99.0)),
                                ('95', (2.5, 97.5))):
            b = QtWidgets.QPushButton(label)
            b.setMaximumWidth(52)
            b.setToolTip('Keep the middle %s%% of the values' % label
                         if label != 'all' else 'The whole range')
            b.clicked.connect(
                lambda _=False, a=lo, b_=hi: self.set_percentiles(a, b_))
            row.addWidget(b)
        form.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.invert_check = QtWidgets.QCheckBox('Invert')
        self.invert_check.toggled.connect(self.on_colormap)
        row.addWidget(self.invert_check)
        self.bar_check = QtWidgets.QCheckBox('Show colour scale')
        self.bar_check.setChecked(True)
        self.bar_check.toggled.connect(self.on_scalar_bar)
        row.addWidget(self.bar_check)
        row.addStretch(1)
        form.addLayout(row)
        return group

    def _group_segment(self):
        group = QtWidgets.QGroupBox('Segment')
        form = QtWidgets.QVBoxLayout(group)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        button = QtWidgets.QPushButton('Find objects...')
        button.setToolTip(
            'Open the flat view: find the objects on the scan, correct the '
            'result by hand, and pick which of them to keep.')
        button.clicked.connect(self.on_segment)
        row.addWidget(button, 1)
        self.seg_clear = QtWidgets.QPushButton('Clear')
        self.seg_clear.setMaximumWidth(60)
        self.seg_clear.setToolTip('Forget the segmentation entirely.')
        self.seg_clear.clicked.connect(self.clear_segmentation)
        row.addWidget(self.seg_clear)
        form.addLayout(row)

        self.seg_label = QtWidgets.QLabel('nothing found yet')
        self.seg_label.setWordWrap(True)
        self.seg_label.setStyleSheet('color: gray;')
        form.addWidget(self.seg_label)

        self.seg_mode = QtWidgets.QComboBox()
        self.seg_mode.addItems(SEGMENT_MODES)
        self.seg_mode.setToolTip(
            'What happens to everything outside the kept regions. Fading '
            'leaves the heights alone and only changes what you see through; '
            'flattening replaces them with the background the objects sit '
            'on, so the substrate keeps its shape and loses its texture.')
        self.seg_mode.currentTextChanged.connect(self.on_segment_mode)
        form.addWidget(self.seg_mode)

        self.rest_slider = self._value_slider(
            form, 'Rest', 0, 100, 15, '%',
            'How much of the discarded part is still visible. 0 leaves the '
            'kept objects floating on their own.', slot=self.on_segment_view)
        self.bg_smooth = self._value_slider(
            form, 'Smooth', 0, 100, 34, '',
            'Over what distance the replacement background is allowed to '
            'change, as a percentage of the frame. 0 gives a single flat '
            'level, which is the honest answer when the substrate really is '
            'flat; larger lets it follow the real substrate.',
            slot=self.on_segment_apply,
            fmt=lambda v: '%.1f%%' % (v / 10.0))
        return group

    def _group_material(self):
        group = QtWidgets.QGroupBox('Material')
        form = QtWidgets.QVBoxLayout(group)

        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems(list(MATERIALS))
        self.material_combo.setCurrentText('Satin')
        self.material_combo.currentTextChanged.connect(self.on_material_preset)
        form.addWidget(self.material_combo)

        self.metallic = self._slider_row(form, 'Metallic', 10, self.on_material)
        self.roughness = self._slider_row(form, 'Roughness', 55, self.on_material)

        self.smooth_check = QtWidgets.QCheckBox('Smooth shading')
        self.smooth_check.setChecked(True)
        self.smooth_check.setToolTip(
            'Interpolate normals across each cell. Off shows the pixel grid, '
            'which is honest about the sampling but noisy on a large scan.')
        self.smooth_check.toggled.connect(self.rebuild_actor)
        form.addWidget(self.smooth_check)
        return group

    def _group_light(self):
        group = QtWidgets.QGroupBox('Light')
        form = QtWidgets.QVBoxLayout(group)

        self.light_combo = QtWidgets.QComboBox()
        self.light_combo.addItems(list(LIGHT_PRESETS))
        self.light_combo.setCurrentText('Studio (3 point)')
        self.light_combo.currentTextChanged.connect(self.on_light_preset)
        form.addWidget(self.light_combo)

        preset = LIGHT_PRESETS['Studio (3 point)']
        self.azimuth = self._value_slider(
            form, 'Direction', 0, 359, preset['azimuth'], 'deg',
            'Where the key light stands, going round the sample. This is the '
            'control that decides which slopes are lit and which way the '
            'shadows fall.')
        self.elevation = self._value_slider(
            form, 'Height', 0, 90, preset['elevation'], 'deg',
            'How high the key light stands. Low is a raking light: long '
            'shadows and every bump visible. High flattens the relief.')
        self.key_slider = self._value_slider(
            form, 'Key', 0, 200, preset['key'], '%',
            'Brightness of the main light.')
        self.fill_slider = self._value_slider(
            form, 'Fill', 0, 100, preset['fill'], '%',
            'A soft light from the other side. Turn it down for hard, '
            'dramatic shadows; up to see into them.')
        self.rim_slider = self._value_slider(
            form, 'Rim', 0, 100, preset['rim'], '%',
            'A light from behind, to separate the surface from the '
            'background.')

        self.shadow_check = QtWidgets.QCheckBox('Cast shadows')
        self.shadow_check.setToolTip(
            'Real shadow maps from these lights. Costs frame rate, and needs '
            'some relief and a low enough light to show at all.')
        self.shadow_check.toggled.connect(self.on_effects)
        form.addWidget(self.shadow_check)
        return group

    def _group_quality(self):
        group = QtWidgets.QGroupBox('Quality')
        form = QtWidgets.QVBoxLayout(group)

        self.ssao_check = QtWidgets.QCheckBox('Ambient occlusion (SSAO)')
        self.ssao_check.setToolTip(
            'Darkens crevices by how enclosed they are. The cheapest way to '
            'make a rough surface read as three-dimensional.')
        self.ssao_check.setChecked(True)
        self.ssao_check.toggled.connect(self.on_effects)
        form.addWidget(self.ssao_check)

        self.edl_check = QtWidgets.QCheckBox('Eye-dome lighting')
        self.edl_check.setToolTip(
            'Non-photorealistic depth shading. Ugly with a lit material, '
            'excellent for reading fine relief - try it with Flat.')
        self.edl_check.toggled.connect(self.on_effects)
        form.addWidget(self.edl_check)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Anti-alias'))
        self.aa_combo = QtWidgets.QComboBox()
        self.aa_combo.addItems(['off', 'fxaa', 'ssaa'])
        self.aa_combo.setCurrentText('fxaa')
        self.aa_combo.setToolTip(
            'ssaa renders larger and shrinks - much cleaner edges, much '
            'slower. Worth switching on just before a screenshot.')
        self.aa_combo.currentTextChanged.connect(self.on_effects)
        row.addWidget(self.aa_combo, 1)
        form.addLayout(row)

        self.env_check = QtWidgets.QCheckBox('Environment reflections')
        self.env_check.setChecked(True)
        self.env_check.toggled.connect(self.on_environment)
        form.addWidget(self.env_check)
        return group

    def _group_scene(self):
        group = QtWidgets.QGroupBox('Scene')
        form = QtWidgets.QVBoxLayout(group)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Background'))
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(list(BACKGROUNDS))
        self.bg_combo.currentTextChanged.connect(self.on_background)
        row.addWidget(self.bg_combo, 1)
        form.addLayout(row)

        self.axes_check = QtWidgets.QCheckBox('Show axes and box')
        self.axes_check.setChecked(True)
        self.axes_check.toggled.connect(self.on_bounds)
        form.addWidget(self.axes_check)

        row = QtWidgets.QHBoxLayout()
        for name in CAMERAS:
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self.set_camera(n))
            row.addWidget(b)
        form.addLayout(row)
        return group

    def _slider_row(self, form, label, value, slot):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.valueChanged.connect(slot)
        row.addWidget(slider, 1)
        form.addLayout(row)
        return slider

    def _value_slider(self, form, label, low, high, value, suffix, tip='',
                      slot=None, fmt=None):
        """A labelled slider: name, slider, and the number it is currently on.

        The light sliders all default to `on_light_slider`, which is what
        makes the preset list and the sliders one control rather than two
        that can disagree; anything else passes its own `slot`.
        """
        row = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel(label)
        name.setFixedWidth(58)
        row.addWidget(name)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        row.addWidget(slider, 1)
        readout = QtWidgets.QLabel()
        readout.setFixedWidth(40)
        readout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        readout.setStyleSheet('color: gray;')
        row.addWidget(readout)

        text = fmt or (lambda v: '%d%s' % (v, suffix))
        slider.valueChanged.connect(lambda v: readout.setText(text(v)))
        slider.valueChanged.connect(slot or self.on_light_slider)
        readout.setText(text(value))
        if tip:
            name.setToolTip(tip)
            slider.setToolTip(tip)
        form.addLayout(row)
        return slider

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu('&File')
        file_menu.addAction('&Open...', self.on_open, 'Ctrl+O')
        file_menu.addSeparator()
        file_menu.addAction('Save &screenshot...', self.on_screenshot, 'Ctrl+S')
        file_menu.addAction('Export &mesh...', self.on_export_mesh)
        file_menu.addAction('Render with &Blender...', self.on_blender)
        file_menu.addSeparator()
        file_menu.addAction('&Quit', self.close, 'Ctrl+Q')

        view_menu = menu.addMenu('&View')
        for name in CAMERAS:
            view_menu.addAction(name, lambda n=name: self.set_camera(n))
        view_menu.addSeparator()
        view_menu.addAction('Reset camera', self.reset_camera)
        view_menu.addSeparator()
        view_menu.addAction('Find &objects...', self.on_segment, 'Ctrl+E')

    # -------------------------------------------------------------- data

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open a scan', os.path.dirname(self.path or ''),
            'Scans (*.gwy *.npy);;Gwyddion files (*.gwy);;NumPy (*.npy);;All files (*)')
        if path:
            self.open_file(path)

    def open_file(self, path, channel=None):
        try:
            names = (gwy_surface.channels(path)
                     if path.lower().endswith('.gwy') else [])
        except Exception as exc:
            self._error('Could not read %s' % os.path.basename(path), exc)
            return

        self.path = path
        self.file_label.setText(os.path.basename(path))

        self._loading = True
        self.channel_combo.clear()
        self.channel_combo.addItems(names)
        if channel and channel in names:
            self.channel_combo.setCurrentText(channel)
        self._loading = False

        self.load_channel(channel or (names[0] if names else None))

    def on_channel(self, name):
        if not self._loading and name:
            self.load_channel(name)

    def load_channel(self, channel):
        if not self.path:
            return
        try:
            surface = gwy_surface.load(self.path, channel)
        except Exception as exc:
            self._error('Could not read that channel', exc)
            return

        trimmed = surface.subsampled()
        if trimmed is not surface:
            self.statusBar().showMessage(
                '%d x %d is large - showing %d x %d'
                % (surface.nx, surface.ny, trimmed.nx, trimmed.ny), 8000)
        self.surface = trimmed
        self.mesh = gwy_surface.SurfaceMesh(trimmed)
        # A mask belongs to the channel it was drawn on. Carrying it to the
        # next one would be silently wrong even when the shapes agree.
        self.segmentation = None

        self._loading = True
        self._sync_height_controls()
        self._sync_colour_controls()
        self.seg_mode.setCurrentText('Off')
        self._update_segment_label()
        self._loading = False

        self.rebuild_actor(reset_camera=True)
        self.setWindowTitle('%s - %s - %s'
                            % (APP_NAME, os.path.basename(self.path),
                               trimmed.name))

    # ------------------------------------------------------------ height

    def _sync_height_controls(self):
        natural = gwy_surface.natural_exaggeration(self.surface)
        # Wide enough to reach true proportion at one end and a caricature at
        # the other, whatever the scan happens to be.
        self._exag_lo = min(natural / 200.0, 0.5)
        self._exag_hi = max(natural * 200.0, 5.0)
        self.exag_spin.setRange(self._exag_lo, self._exag_hi)
        self.exag_spin.setValue(self.mesh.exaggeration)
        self.exag_slider.setValue(self._exag_to_slider(self.mesh.exaggeration))
        self._update_relief_label()

    def _exag_to_slider(self, value):
        t = ((np.log10(max(value, 1e-9)) - np.log10(self._exag_lo))
             / (np.log10(self._exag_hi) - np.log10(self._exag_lo)))
        return int(round(float(np.clip(t, 0.0, 1.0)) * 1000))

    def _slider_to_exag(self, pos):
        t = pos / 1000.0
        log = (np.log10(self._exag_lo)
               + t * (np.log10(self._exag_hi) - np.log10(self._exag_lo)))
        return float(10.0 ** log)

    def on_exaggeration_slid(self, pos):
        if self._loading or self.mesh is None:
            return
        self.set_exaggeration(self._slider_to_exag(pos), from_slider=True)

    def on_exaggeration_typed(self, value):
        if self._loading or self.mesh is None:
            return
        self.set_exaggeration(value)

    def set_exaggeration(self, value, from_slider=False):
        if self.mesh is None:
            return
        value = self.mesh.set_exaggeration(value)
        self._loading = True
        self.exag_spin.setValue(value)
        if not from_slider:
            self.exag_slider.setValue(self._exag_to_slider(value))
        self._loading = False
        self._update_relief_label()
        # The mesh grew or shrank, so the bounding box has to follow it.
        if self.axes_check.isChecked():
            self._show_bounds()
        self.plotter.render()

    def auto_exaggeration(self):
        self.set_exaggeration(gwy_surface.natural_exaggeration(self.surface))

    def _update_relief_label(self):
        if self.mesh is None:
            return
        lo, hi = self.mesh.full_height_range()
        self.relief_label.setText(
            'relief %.2f x frame width   (%.3g %s of real height)'
            % (self.mesh.relief_height, hi - lo, self.mesh.z_label))

    # ------------------------------------------------------------ colour

    def _sync_colour_controls(self):
        counts, edges = self.surface.histogram(bins=256)
        factor = self.mesh.z_factor
        self.histogram.set_distribution(counts, edges * factor)
        self.histogram.set_unit(self.mesh.z_label)
        self.histogram.set_colormap(self.colormap())
        vmin, vmax = self.mesh.height_range(0.5, 99.5)
        self.histogram.set_limits(vmin, vmax)

    def colormap(self):
        cmap = gwy_colormaps.get(self.cmap_combo.currentText())
        return cmap.reversed() if self.invert_check.isChecked() else cmap

    def on_colormap(self, *_):
        if self._loading:
            return
        cmap = self.colormap()
        self.histogram.set_colormap(cmap)
        if self.actor is None:
            return
        if self._segment_mode() == 'fade':
            # A per-point opacity makes VTK draw baked RGBA rather than run
            # the lookup table, so the table no longer decides anything and
            # the actor has to be built again.
            self.rebuild_actor()
            return
        self.actor.mapper.lookup_table.cmap = cmap
        self.actor.mapper.lookup_table.scalar_range = self.histogram.limits()
        self.plotter.render()

    def on_range(self, vmin, vmax):
        if self.actor is None:
            return
        if self._segment_mode() == 'fade':
            self.rebuild_actor()
            return
        self.actor.mapper.scalar_range = (vmin, vmax)
        if self.scalar_bar is not None:
            self.plotter.update_scalar_bar_range((vmin, vmax))
        self.plotter.render()

    def set_percentiles(self, low, high):
        if self.surface is None:
            return
        lo, hi = self.surface.percentile_range(low, high)
        factor = self.mesh.z_factor
        self.histogram.set_limits(lo * factor, hi * factor, emit=True)

    def on_scalar_bar(self, _=None):
        self.rebuild_actor()

    # ------------------------------------------------------- segmentation

    def on_segment(self):
        """Open the flat view, and take back whatever comes out of it."""
        if self.surface is None:
            return
        import gwy_segment_view
        dialog = gwy_segment_view.SegmentDialog(
            self, self.surface, cmap=self.colormap(),
            clim=self._colour_limits_in_data(),
            segmentation=self.segmentation)
        accepted = dialog.exec_() if hasattr(dialog, 'exec_') else dialog.exec()
        if not accepted or dialog.result_segmentation is None:
            return
        self.segmentation = dialog.result_segmentation
        if self.seg_mode.currentText() == 'Off':
            self._loading = True
            self.seg_mode.setCurrentText('Flatten the rest to the background')
            self._loading = False
        self.apply_segmentation()

    def _colour_limits_in_data(self):
        """The histogram limits, back in the units the raw array is in.

        The histogram works in the printed prefix - nanometres - and the flat
        view colours the raw array, so the two have to be told the same range
        in the same units or the 2D and the 3D pictures disagree.
        """
        vmin, vmax = self.histogram.limits()
        factor = self.mesh.z_factor or 1.0
        return vmin / factor, vmax / factor

    def clear_segmentation(self):
        self.segmentation = None
        self._loading = True
        self.seg_mode.setCurrentText('Off')
        self._loading = False
        self.apply_segmentation()

    def on_segment_mode(self, *_):
        if self._loading:
            return
        if self.segmentation is None and self.seg_mode.currentText() != 'Off':
            self.statusBar().showMessage(
                'Nothing has been found yet - press "Find objects..." first.',
                6000)
        self.apply_segmentation()

    def on_segment_apply(self, *_):
        """A control that changes the heights, so the mesh has to be redone."""
        if self._loading:
            return
        if self._segment_mode() == 'flatten':
            self.apply_segmentation()

    def on_segment_view(self, *_):
        """A control that only changes what is seen through."""
        if self._loading:
            return
        if self._segment_mode() == 'fade':
            self.rebuild_actor()

    def _segment_mode(self):
        """`'off'`, `'fade'` or `'flatten'`, with no segmentation meaning off."""
        if self.segmentation is None or self.mesh is None:
            return 'off'
        name = self.seg_mode.currentText()
        if name.startswith('Fade'):
            return 'fade'
        if name.startswith('Flatten'):
            return 'flatten'
        return 'off'

    def apply_segmentation(self):
        """Put the mask into the scene, whichever way the mode asks for."""
        if self.mesh is None:
            return
        mode = self._segment_mode()
        if mode == 'flatten':
            import gwy_segment
            keep = self.segmentation.mask()
            self.mesh.set_heights(gwy_segment.flatten(
                self.surface.z, keep, self.segmentation.scale,
                smooth=self.bg_smooth.value() / 10.0))
        elif self.mesh.edited is not None:
            self.mesh.set_heights(None)
        self._update_segment_label()
        self.rebuild_actor()
        self._update_relief_label()

    def _segment_rgba(self):
        """The whole surface as explicit RGBA, faded outside the mask.

        VTK will only draw a partly transparent surface if it is told the
        colours directly - `add_mesh(..., rgba=True)`. Handing it a per-point
        opacity array instead is the documented route and does not work here:
        the mesh came out flat grey and fully opaque, because the actor was
        still being sorted into the opaque pass. Colouring the points here
        costs one pass over the array and behaves the same on every driver.

        The cost of doing it this way is that the colour map is no longer
        live - the lookup table is out of the picture, so changing the
        gradient or the range means building this again, which is what
        `on_colormap` and `on_range` do.
        """
        import gwy_segment
        height = np.asarray(self.mesh.mesh.point_data['height'])
        vmin, vmax = self.histogram.limits()
        span = (vmax - vmin) or 1.0
        good = np.isfinite(height)
        t = np.clip((np.where(good, height, vmin) - vmin) / span, 0.0, 1.0)

        rgba = (np.asarray(self.colormap()(t)) * 255.0).astype(np.uint8)
        rgba[~good] = (105, 105, 105, 255)      # the nan_color used elsewhere
        alpha = gwy_segment.alpha(self.segmentation.mask(),
                                  rest=self.rest_slider.value() / 100.0)
        rgba[:, 3] = (alpha.ravel(order='C') * 255.0).astype(np.uint8)
        self.mesh.mesh.point_data['rgba'] = rgba
        return rgba

    def _update_segment_label(self):
        seg = self.segmentation
        if seg is None:
            self.seg_label.setText('nothing found yet')
            return
        self.seg_label.setText(
            '%d of %d regions kept - %.1f %% of the frame'
            % (seg.kept_count, seg.region_count, 100.0 * seg.mask().mean()))

    # ---------------------------------------------------------- rendering

    def rebuild_actor(self, *_args, reset_camera=False):
        """Put the mesh back in the scene. Cheap enough to do on any change
        that the actor cannot be talked into making in place."""
        if self.mesh is None:
            return
        camera = None if reset_camera else self.plotter.camera_position

        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
            self.actor = None
        if self.scalar_bar is not None:
            try:
                self.plotter.remove_scalar_bar()
            except Exception:
                pass
            self.scalar_bar = None

        name = self.material_combo.currentText()
        metallic, roughness, pbr, diffuse, specular = MATERIALS[name]
        metallic = self.metallic.value() / 100.0
        roughness = self.roughness.value() / 100.0
        fading = self._segment_mode() == 'fade'

        shared = dict(
            smooth_shading=self.smooth_check.isChecked(),
            show_scalar_bar=False,
            pbr=pbr,
            metallic=metallic,
            roughness=roughness,
            diffuse=diffuse,
            specular=specular if not pbr else 0.0,
            lighting=name != 'Flat (no shading)',
        )
        if fading:
            self._segment_rgba()
            self.actor = self.plotter.add_mesh(
                self.mesh.mesh, scalars='rgba', rgba=True, **shared)
        else:
            self.actor = self.plotter.add_mesh(
                self.mesh.mesh, scalars='height', cmap=self.colormap(),
                clim=self.histogram.limits(), nan_color='dimgray', **shared)

        if self.bar_check.isChecked():
            self.scalar_bar = self.plotter.add_scalar_bar(
                title=_vtk_text(self.mesh.height_title), n_labels=5,
                vertical=True, position_x=0.88, position_y=0.15,
                width=0.06, height=0.7, title_font_size=14, label_font_size=12,
                # With explicit RGBA there is no lookup table on the actor to
                # read, so the bar is given one that says what the colours
                # mean anyway.
                **({'mapper': self._legend_mapper()} if fading else {}))

        self.on_environment()
        self._update_light_enabled()
        self._apply_lights()
        self.on_effects()
        self.on_bounds()

        if reset_camera:
            self.plotter.view_isometric()
        else:
            self.plotter.camera_position = camera
        self.plotter.render()

    def _legend_mapper(self):
        """A mapper that exists only to hold a colour scale for the bar."""
        clim = self.histogram.limits()
        mapper = pv.DataSetMapper()
        mapper.lookup_table = pv.LookupTable(cmap=self.colormap(),
                                             scalar_range=clim)
        mapper.scalar_range = clim
        return mapper

    def on_material_preset(self, name):
        metallic, roughness, _pbr, _d, _s = MATERIALS[name]
        self._loading = True
        self.metallic.setValue(int(metallic * 100))
        self.roughness.setValue(int(roughness * 100))
        self._loading = False
        self.rebuild_actor()

    def on_material(self, *_):
        """Metallic and roughness can be changed on the live actor."""
        if self._loading or self.actor is None:
            return
        self.actor.prop.metallic = self.metallic.value() / 100.0
        self.actor.prop.roughness = self.roughness.value() / 100.0
        self.plotter.render()

    def on_effects(self, *_):
        pl = self.plotter
        try:
            pl.disable_ssao()
        except Exception:
            pass
        if self.ssao_check.isChecked():
            # The radius is in world units and the scene is one frame wide,
            # so a few percent of that is the scale of detail to occlude.
            pl.enable_ssao(radius=0.04, bias=0.002, kernel_size=64, blur=True)

        try:
            pl.disable_shadows()
        except Exception:
            pass
        if self.shadow_check.isChecked():
            pl.enable_shadows()

        try:
            pl.disable_eye_dome_lighting()
        except Exception:
            pass
        if self.edl_check.isChecked():
            pl.enable_eye_dome_lighting()

        try:
            pl.disable_anti_aliasing()
        except Exception:
            pass
        mode = self.aa_combo.currentText()
        if mode != 'off':
            pl.enable_anti_aliasing(mode)
        pl.render()

    def on_environment(self, *_):
        pl = self.plotter
        if self.env_check.isChecked():
            if self._environment is None:
                self._environment = studio_cubemap()
            try:
                pl.set_environment_texture(self._environment)
            except Exception:
                pass
        else:
            try:
                pl.remove_environment_texture()
            except Exception:
                pass
        pl.render()

    # -------------------------------------------------------------- light

    def _light_mode(self):
        preset = LIGHT_PRESETS.get(self.light_combo.currentText())
        return preset.get('mode', 'rig') if preset else 'rig'

    def on_light_preset(self, name):
        """A preset only moves the sliders; the sliders do the work."""
        preset = LIGHT_PRESETS.get(name)
        if preset:
            self._loading = True
            for key, slider in (('azimuth', self.azimuth),
                                ('elevation', self.elevation),
                                ('key', self.key_slider),
                                ('fill', self.fill_slider),
                                ('rim', self.rim_slider)):
                if key in preset:
                    slider.setValue(preset[key])
            self._loading = False
        self._update_light_enabled()
        self._apply_lights()

    def on_light_slider(self, *_):
        if self._loading:
            return
        # Touching a slider means the preset no longer describes the scene.
        if self.light_combo.currentText() != 'Custom':
            self._loading = True
            self.light_combo.setCurrentText('Custom')
            self._loading = False
            self._update_light_enabled()
        self._apply_lights()

    def _update_light_enabled(self):
        """Grey out the angles for the two rigs that do not have any."""
        rig = self._light_mode() == 'rig'
        for slider in (self.azimuth, self.elevation,
                       self.fill_slider, self.rim_slider):
            slider.setEnabled(rig)
        self.key_slider.setEnabled(self._light_mode() != 'lightkit')

    def _rig_lights(self):
        """The three lights, placed from the sliders.

        Azimuth and elevation are spherical coordinates around the middle of
        the surface, which sits at (0.5, 0.5, 0) because the mesh is one
        frame wide. Fill and rim are hung off the key at fixed offsets - the
        useful control is where the key is, and three independent directions
        would be three times the fiddling for very little more.
        """
        centre = np.array([0.5, 0.5, 0.0])
        azimuth = np.radians(float(self.azimuth.value()))
        elevation = np.radians(float(self.elevation.value()))
        distance = 3.0

        def direction(a, e):
            return np.array([np.cos(e) * np.cos(a),
                             np.cos(e) * np.sin(a),
                             np.sin(e)])

        rig = (
            (0.0, elevation, self.key_slider.value()),
            (np.radians(130.0), max(elevation * 0.5, np.radians(12.0)),
             self.fill_slider.value()),
            (np.radians(215.0), elevation * 0.8 + np.radians(12.0),
             self.rim_slider.value()),
        )
        lights = []
        for offset, height, strength in rig:
            if strength <= 0:
                continue
            position = centre + direction(azimuth + offset, height) * distance
            lights.append(pv.Light(position=tuple(position),
                                   focal_point=tuple(centre),
                                   intensity=strength / 100.0,
                                   light_type='scene light'))
        return lights

    def _apply_lights(self):
        pl = self.plotter
        pl.remove_all_lights()
        mode = self._light_mode()
        if mode == 'lightkit':
            pl.enable_lightkit()
        elif mode == 'headlight':
            pl.add_light(pv.Light(
                light_type='headlight',
                intensity=self.key_slider.value() / 100.0))
        else:
            for light in self._rig_lights():
                pl.add_light(light)
        pl.render()

    def on_background(self, name):
        self._set_background(name)
        self.plotter.render()

    def _set_background(self, name):
        colour, top = BACKGROUNDS[name]
        self.plotter.set_background(colour, top=top)

    def on_bounds(self, *_):
        try:
            self.plotter.remove_bounds_axes()
        except Exception:
            pass
        if self.axes_check.isChecked() and self.mesh is not None:
            self._show_bounds()
        self.plotter.render()

    def _show_bounds(self):
        """Axes labelled in real units, even though the mesh is normalised.

        The mesh is one frame wide and stretched in Z, so nothing about its
        coordinates is a physical number. Every tick therefore has to be
        mapped back: undo the exaggeration, undo the normalisation, put the
        reference level the mesh was centred on back on top, and print the
        result in the same prefix the colour scale uses.
        """
        surface = self.surface
        mesh = self.mesh
        frame = surface.frame

        def physical(z):
            return ((z / mesh.exaggeration) * frame
                    + mesh.z_reference) * mesh.z_factor

        z_lo, z_hi = mesh.mesh.bounds[4], mesh.mesh.bounds[5]
        self.plotter.show_bounds(
            xtitle='x [%s]' % _vtk_text(mesh.xy_label or 'px'),
            ytitle='y [%s]' % _vtk_text(mesh.xy_label or 'px'),
            ztitle='z [%s] x%.4g' % (_vtk_text(mesh.z_label or 'a.u.'),
                                     mesh.exaggeration),
            grid='back', location='outer', ticks='outside',
            axes_ranges=[0.0, surface.x_real * mesh.xy_factor,
                         0.0, surface.y_real * mesh.xy_factor,
                         physical(z_lo), physical(z_hi)],
            show_zlabels=True, font_size=10)

    def set_camera(self, name):
        plane = CAMERAS[name]
        if plane is None:
            self.plotter.view_isometric()
        else:
            getattr(self.plotter, 'view_%s' % plane)()
        self.plotter.render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    # ------------------------------------------------------------ export

    def on_screenshot(self):
        if self.mesh is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save the view', self._suggested_name('.png'),
            'PNG image (*.png)')
        if not path:
            return
        scale, ok = QtWidgets.QInputDialog.getInt(
            self, 'Resolution', 'Render at this many times the window size:',
            2, 1, 8)
        if not ok:
            return
        try:
            self.plotter.screenshot(path, scale=scale)
        except Exception as exc:
            self._error('Could not save the screenshot', exc)
            return
        self.statusBar().showMessage('Saved %s' % path, 5000)

    def on_export_mesh(self):
        """Write the surface as a coloured mesh any 3D program can open."""
        if self.mesh is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export the surface', self._suggested_name('.ply'),
            'PLY with vertex colours (*.ply);;STL (*.stl);;OBJ (*.obj);;VTK (*.vtk)')
        if not path:
            return
        try:
            import gwy_blender_export
            gwy_blender_export.save_mesh(
                self.mesh, path, cmap=self.colormap(),
                clim=self.histogram.limits())
        except Exception as exc:
            self._error('Could not export the mesh', exc)
            return
        self.statusBar().showMessage('Wrote %s' % path, 5000)

    def on_blender(self):
        if self.mesh is None:
            return
        import gwy_blender_export
        dialog = gwy_blender_export.BlenderDialog(
            self, self.mesh, cmap=self.colormap(),
            clim=self.histogram.limits(),
            suggested=self._suggested_name('_render.png'),
            metallic=self.metallic.value() / 100.0,
            roughness=self.roughness.value() / 100.0)
        dialog.exec_() if hasattr(dialog, 'exec_') else dialog.exec()

    def _suggested_name(self, suffix):
        if not self.path:
            return 'surface' + suffix
        stem = os.path.splitext(self.path)[0]
        channel = (self.surface.name if self.surface else 'z')
        safe = ''.join(c if c.isalnum() else '_' for c in channel).strip('_')
        return '%s_%s_3d%s' % (stem, safe, suffix)

    # ------------------------------------------------------------- misc

    def _error(self, message, exc=None):
        detail = ('%s\n\n%s: %s' % (message, type(exc).__name__, exc)
                  if exc is not None else message)
        QtWidgets.QMessageBox.critical(self, APP_NAME, detail)
        self.statusBar().showMessage(message, 8000)

    def closeEvent(self, event):
        # The VTK render window has to be torn down explicitly or the process
        # hangs on exit.
        try:
            self.plotter.close()
        except Exception:
            pass
        super(Viewer3DWindow, self).closeEvent(event)


# ------------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    path, channel = None, None
    if '--channel' in argv:
        i = argv.index('--channel')
        channel = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    if argv:
        path = argv[0]

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = Viewer3DWindow(path, channel)
    window.show()
    return app.exec_() if hasattr(app, 'exec_') else app.exec()


if __name__ == '__main__':
    sys.exit(main())
