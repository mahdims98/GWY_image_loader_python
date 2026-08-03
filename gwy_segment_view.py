"""
The window where the segmentation is looked at and corrected.

A 3D view is the wrong place to decide what a region is. You cannot see
behind an object, a click lands on whatever the camera happens to be
pointing at, and the thing being judged - is this blob one object or two,
does this line really run all the way across - is a question about the map
and not about the surface. So the segmentation is edited flat, in the plane
the measurement was taken in, and the 3D view only shows the result.

The window is the image, the mask over it, the settings on the right and a
count at the bottom. Four tools:

  Pick     click a region to keep or drop it. This is the one that matters:
           the detector's job is to propose regions and yours is to say
           which ones you meant.
  Box      drag a rectangle. What that does is the combo next to it -
           keep, drop, erase, or search inside the box for objects.
  Brush    paint something in that the detector missed.
  Erase    paint something out that it should not have found.

Dropping and erasing are different on purpose. A dropped region stays on the
map, greyed out, and one more click brings it back; an erased one is gone.
Use drop while you are still deciding and erase once you are not.

Searching inside a box is the way out of a bad automatic result. Every
threshold is measured inside the rectangle, so a corner of the frame is
judged by that corner: a faint object that the whole-frame statistics
drowned comes straight out once the box is drawn round it. What was in the
box before is replaced, so a second attempt over the same place corrects
itself instead of piling one answer on top of another.

Regions are drawn in cycling colours by number and not by type, because
there are no types here - see the `gwy_segment` docstring. The colours only
serve to tell one region from the one next to it.

Two conventions worth stating because they are easy to get wrong.

*Row 0 is drawn at the bottom.* The 3D view builds its mesh with the first
scan line at y = 0, so that is where this window puts it too, and a feature
in the bottom left here is in the bottom left there. The processing GUI
draws its images the other way up; matching the view this window exists to
control was the more useful of the two.

*The image is drawn to its physical aspect, not to its pixel counts.* The
scans this was written against are 1024 x 512 pixels over a square 7 x 7 um
frame, so a pixel is twice as tall as it is wide. Drawing the array would
show round objects as ovals, and a round brush would paint an ellipse.
"""

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import gwy_segment as gs


TOOLS = (
    ('pick', 'Pick', 'Click a region to keep or drop it.'),
    ('box', 'Box', 'Drag a rectangle. What it does to what is inside is set '
                   'by the box action next to these buttons.'),
    ('brush', 'Brush', 'Paint in something the detector missed.'),
    ('erase', 'Erase', 'Paint out something it should not have found.'),
)

#: What a dragged rectangle does. `find` is the one worth knowing about: it
#: runs the detectors over that rectangle alone, with the thresholds measured
#: inside it.
BOX_ACTIONS = (
    ('find', 'Find objects inside',
     'Search inside the rectangle and replace whatever was there. Every '
     'threshold is measured inside the box, so an object the whole-frame '
     'settings missed usually comes out once the box is drawn round it.'),
    ('keep', 'Keep what is inside',
     'Keep every region the rectangle touches.'),
    ('drop', 'Drop what is inside',
     'Drop every region the rectangle touches. They stay on the map, greyed '
     'out, so a second click can bring them back.'),
    ('erase', 'Erase what is inside',
     'Take the rectangle off the map entirely - regions and painting both. '
     'For an area that came out as rubbish.'),
)

#: How the overlay is drawn. A kept region is tinted solidly enough to read
#: at a glance; a dropped one is left faint rather than hidden, because "the
#: detector found this and you turned it off" and "the detector never found
#: it" are different states and hiding the first one loses that.
KEPT_ALPHA = 105
DROPPED_ALPHA = 40
PAINT_ALPHA = 130


def _qimage_rgb(rgb):
    """A QImage owning its own copy of an (h, w, 3) uint8 array."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return image.copy()


def _qimage_rgba(rgba):
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    image = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
    return image.copy()


def colorize(z, cmap, clim):
    """The height map as RGB, the way the 3D view colours it."""
    lo, hi = float(clim[0]), float(clim[1])
    if not (hi > lo):
        hi = lo + 1.0
    t = np.clip((np.asarray(z, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    bad = ~np.isfinite(z)
    t[bad] = 0.0
    rgb = (np.asarray(cmap(t))[..., :3] * 255.0).astype(np.uint8)
    rgb[bad] = 110
    return rgb


def region_palette(segmentation):
    """RGBA per label, so the overlay is one array lookup.

    Colour cycles with the region number and says nothing about what the
    region is. Dropped regions all share one dull colour: at that point the
    only thing worth reading off the screen is that they are off.
    """
    count = segmentation.count
    table = np.zeros((count + 1, 4), dtype=np.uint8)
    if not count:
        return table
    index = np.arange(1, count + 1)
    colours = np.array(gs.REGION_COLOURS, dtype=np.uint8)
    table[1:, :3] = colours[(index - 1) % len(colours)]
    table[1:, 3] = KEPT_ALPHA
    off = ~segmentation.keep[1:]
    table[1:][off, :3] = gs.DROPPED_COLOUR
    table[1:][off, 3] = DROPPED_ALPHA
    return table


class MaskView(QtWidgets.QWidget):
    """The image with the mask on it, and the mouse that edits the mask.

    Both images are built once and blitted, so a repaint costs a scaled draw
    and not a pass over the array. The overlay is rebuilt only when something
    about the mask changes, which on a megapixel scan is the difference
    between a responsive brush and a slideshow.
    """

    changed = QtCore.Signal()
    hovered = QtCore.Signal(str)
    #: A box was dragged with the action set to `find`. Carries
    #: `(row0, row1, col0, col1)`; the dialog does the detecting, because
    #: this widget has no business knowing the settings.
    searched = QtCore.Signal(tuple)

    def __init__(self, parent=None):
        super(MaskView, self).__init__(parent)
        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setCursor(QtCore.Qt.CrossCursor)

        self.segmentation = None
        self._base = None           # QImage of the heights
        self._overlay = None        # QImage of the mask
        self._shape = (1, 1)
        self._aspect = 1.0          # width / height of the frame, physical
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._show_mask = True
        self._tool = 'pick'
        self._box_action = 'find'
        self._brush = 1.0           # percent of the frame
        self._drag = None
        self._box = None
        self._last = None

    # ------------------------------------------------------------ content

    def set_image(self, z, cmap, clim):
        self._shape = tuple(z.shape)
        # drawn bottom-up, see the module docstring
        self._base = _qimage_rgb(colorize(z, cmap, clim)[::-1])
        self.update()

    def set_segmentation(self, segmentation):
        self.segmentation = segmentation
        if segmentation is not None:
            scale = segmentation.scale
            self._aspect = (scale.x_real / scale.y_real) if scale.y_real else 1.0
        self.refresh()

    def refresh(self):
        """Rebuild the overlay from the segmentation as it stands."""
        seg = self.segmentation
        if seg is None or self._base is None:
            self._overlay = None
            self.update()
            return

        ny, nx = seg.shape
        if self._show_mask:
            rgba = region_palette(seg)[seg.labels]
            for where, colour in ((seg.added, gs.ADDED_COLOUR),
                                  (seg.removed, gs.REMOVED_COLOUR)):
                if where.any():
                    rgba[where, 0] = colour[0]
                    rgba[where, 1] = colour[1]
                    rgba[where, 2] = colour[2]
                    rgba[where, 3] = PAINT_ALPHA
        else:
            rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        self._overlay = _qimage_rgba(rgba[::-1])
        self.update()

    def set_show_mask(self, on):
        self._show_mask = bool(on)
        self.refresh()

    def set_tool(self, tool):
        self._tool = tool
        self.update()

    def set_box_action(self, action):
        self._box_action = action

    def set_brush(self, percent):
        self._brush = max(0.05, float(percent))
        self.update()

    # ------------------------------------------------------------ geometry

    def _target(self):
        """Where the image is drawn, at the physical aspect of the frame."""
        w, h = self.width(), self.height()
        fit = min(w / max(self._aspect, 1e-9), float(h))
        height = fit * self._zoom
        width = height * self._aspect
        left = 0.5 * (w - width) + self._pan.x()
        top = 0.5 * (h - height) + self._pan.y()
        return QtCore.QRectF(left, top, max(1.0, width), max(1.0, height))

    def _to_image(self, point):
        """Widget point -> `(row, col)` in the data, or None if outside."""
        rect = self._target()
        if not rect.contains(point):
            return None
        ny, nx = self._shape
        col = int((point.x() - rect.left()) / rect.width() * nx)
        row = int((point.y() - rect.top()) / rect.height() * ny)
        row = ny - 1 - row                      # drawn bottom-up
        if 0 <= row < ny and 0 <= col < nx:
            return row, col
        return None

    def _clamped(self, point):
        """`(row, col)` for a point that may be off the image, pulled inside.

        A box dragged past the border should mean "out to the border", not
        "no box at all".
        """
        rect = self._target()
        ny, nx = self._shape
        col = int((point.x() - rect.left()) / rect.width() * nx)
        row = int((point.y() - rect.top()) / rect.height() * ny)
        row = ny - 1 - row
        return (int(np.clip(row, 0, ny - 1)), int(np.clip(col, 0, nx - 1)))

    def fit(self):
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self.update()

    # ------------------------------------------------------------ painting

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._base is None:
            painter.setPen(self.palette().mid().color())
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, 'No image')
            painter.end()
            return

        rect = self._target()
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform,
                              rect.width() < self._shape[1])
        painter.drawImage(rect, self._base)
        if self._overlay is not None:
            painter.drawImage(rect, self._overlay)

        if self._box is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1,
                                      QtCore.Qt.DashLine))
            painter.setBrush(QtGui.QColor(255, 255, 255, 30))
            painter.drawRect(self._box)

        if self._tool in ('brush', 'erase') and self._last is not None:
            self._paint_brush_cursor(painter, rect)

        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(self.palette().mid().color(), 1))
        painter.drawRect(rect)
        painter.end()

    def _paint_brush_cursor(self, painter, rect):
        """A circle the size the brush will actually paint."""
        # The brush is a percentage of the frame's longer side, and the rect
        # is drawn at the frame's physical aspect, so that side is whichever
        # of the two is longer on screen.
        diameter = self._brush / 100.0 * max(rect.width(), rect.height())
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1))
        painter.drawEllipse(self._last, diameter / 2.0, diameter / 2.0)

    # -------------------------------------------------------------- mouse

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            return
        before = self._to_image(event.position()
                               if hasattr(event, 'position') else event.pos())
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom = float(np.clip(self._zoom * factor, 1.0, 40.0))
        if self._zoom <= 1.0:
            self._pan = QtCore.QPointF(0.0, 0.0)
        elif before is not None:
            # keep the pixel under the cursor under the cursor
            self.update()
            rect = self._target()
            ny, nx = self._shape
            x = rect.left() + (before[1] + 0.5) / nx * rect.width()
            y = rect.top() + (ny - 1 - before[0] + 0.5) / ny * rect.height()
            pos = (event.position() if hasattr(event, 'position')
                   else QtCore.QPointF(event.pos()))
            self._pan += QtCore.QPointF(pos.x() - x, pos.y() - y)
        self.update()

    def mousePressEvent(self, event):
        pos = self._pos(event)
        self._last = pos
        if event.button() == QtCore.Qt.MiddleButton:
            self._drag = ('pan', pos)
            return
        if event.button() != QtCore.Qt.LeftButton or self.segmentation is None:
            return
        if self._tool == 'pick':
            self._pick(pos)
        elif self._tool == 'box':
            self._drag = ('box', pos)
            self._box = QtCore.QRectF(pos, pos)
        else:
            self._drag = ('paint', pos)
            self._paint_at(pos)

    def mouseMoveEvent(self, event):
        pos = self._pos(event)
        self._last = pos
        if self._drag is None:
            self._report(pos)
            if self._tool in ('brush', 'erase'):
                self.update()
            return
        mode, start = self._drag
        if mode == 'pan':
            self._pan += pos - start
            self._drag = (mode, pos)
        elif mode == 'box':
            self._box = QtCore.QRectF(start, pos).normalized()
        else:
            self._paint_line(start, pos)
            self._drag = (mode, pos)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag is not None and self._drag[0] == 'box':
            self._apply_box(self._box)
            self._box = None
        self._drag = None
        self.update()

    def leaveEvent(self, event):
        self._last = None
        self.update()

    @staticmethod
    def _pos(event):
        return (event.position() if hasattr(event, 'position')
                else QtCore.QPointF(event.pos()))

    # ------------------------------------------------------------ editing

    def _pick(self, pos):
        where = self._to_image(pos)
        if where is None:
            return
        label = self.segmentation.label_at(*where)
        if label:
            self.segmentation.toggle(label)
            self.refresh()
            self.changed.emit()
        else:
            self.hovered.emit('Nothing there - no region claims that pixel. '
                              'Draw a box round it and search inside, or use '
                              'the brush.')

    def _paint_at(self, pos):
        where = self._to_image(pos)
        if where is None:
            return
        seg = self.segmentation
        adding = self._tool == 'brush'
        target = seg.added if adding else seg.removed
        other = seg.removed if adding else seg.added
        footprint = gs.disc(seg.scale, self._brush)
        gs.stamp(target, footprint, where[0], where[1], True)
        gs.stamp(other, footprint, where[0], where[1], False)
        self.refresh()
        self.changed.emit()

    def _paint_line(self, start, end):
        """Stamp along the drag, so a fast mouse does not leave gaps."""
        steps = int(max(abs(end.x() - start.x()), abs(end.y() - start.y())) / 2.0)
        for i in range(max(1, steps) + 1):
            t = i / float(max(1, steps))
            self._paint_at(QtCore.QPointF(start.x() + (end.x() - start.x()) * t,
                                          start.y() + (end.y() - start.y()) * t))

    def _box_bounds(self, box):
        """A dragged rectangle as `(row0, row1, col0, col1)`, or None."""
        if box is None or self.segmentation is None:
            return None
        if box.width() < 2 and box.height() < 2:
            return None
        a = self._clamped(box.topLeft())
        b = self._clamped(box.bottomRight())
        r0, r1 = sorted((a[0], b[0]))
        c0, c1 = sorted((a[1], b[1]))
        return r0, r1, c0, c1

    def _apply_box(self, box):
        """Do whatever the box action says to what the rectangle covers."""
        bounds = self._box_bounds(box)
        if bounds is None:
            return
        r0, r1, c0, c1 = bounds
        if self._box_action == 'find':
            self.searched.emit(bounds)
            return

        seg = self.segmentation
        window = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        if self._box_action == 'erase':
            where = np.zeros(seg.shape, dtype=bool)
            where[window] = True
            gone = seg.erase_where(where)
            self.hovered.emit('Erased %d region%s from that rectangle.'
                              % (gone, '' if gone == 1 else 's'))
        else:
            inside = seg.labels_in(window)
            if not inside.size:
                self.hovered.emit('No regions in that rectangle.')
                return
            seg.keep[inside] = (self._box_action == 'keep')
            self.hovered.emit('%s %d region%s.'
                              % ('Kept' if self._box_action == 'keep' else 'Dropped',
                                 inside.size, '' if inside.size == 1 else 's'))
        self.refresh()
        self.changed.emit()

    def _report(self, pos):
        where = self._to_image(pos)
        if where is None or self.segmentation is None:
            return
        seg = self.segmentation
        label = seg.label_at(*where)
        if not label:
            self.hovered.emit('row %d, col %d - not in a region'
                              % (where[0], where[1]))
            return
        row = seg.measure()[label]
        self.hovered.emit(
            'Region %d - %s across, %s wide, area %s%s'
            % (label, _length(row['length']), _length(row['width']),
               _area(row['area']),
               '' if seg.keep[label] else '   (dropped)'))


def _length(value):
    for size, prefix in ((1e-9, 'nm'), (1e-6, 'um'), (1e-3, 'mm')):
        if value < size * 1000.0:
            return '%.3g %s' % (value / size, prefix)
    return '%.3g m' % value


def _area(value):
    for size, prefix in ((1e-18, 'nm'), (1e-12, 'um'), (1e-6, 'mm')):
        if value < size * 1e6:
            return '%.3g %s2' % (value / size, prefix)
    return '%.3g m2' % value


# ------------------------------------------------------------------- dialog

class SegmentDialog(QtWidgets.QDialog):
    """Detect, correct, and hand the result back.

    Owns a working copy of the segmentation. `Cancel` throws it away, so
    trying a setting is free; `OK` gives it to the caller.
    """

    def __init__(self, parent, surface, cmap, clim, segmentation=None):
        super(SegmentDialog, self).__init__(parent)
        self.setWindowTitle('Segmentation - %s' % surface.name)
        self.resize(1220, 820)

        self.surface = surface
        self._cmap = cmap
        self._clim = clim
        self._loading = False
        self.result_segmentation = None

        self.view = MaskView(self)
        self.view.set_image(surface.z, cmap, clim)
        self.view.changed.connect(self._update_summary)
        self.view.hovered.connect(self._show_hover)
        self.view.searched.connect(self.detect_in)

        self._build_ui()
        if segmentation is not None and segmentation.shape == surface.z.shape:
            self._install(segmentation)
        else:
            self.detect()

    # -------------------------------------------------------------- layout

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.view, 1)

        panel = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self._group_detect())
        side.addWidget(self._group_regions())
        side.addWidget(self._group_tools())
        side.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(panel)
        scroll.setFixedWidth(360)
        body.addWidget(scroll, 0)
        outer.addLayout(body, 1)

        self.status = QtWidgets.QLabel('')
        self.status.setStyleSheet('color: gray;')
        outer.addWidget(self.status)

        self.summary = QtWidgets.QLabel('')
        outer.addWidget(self.summary)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _spin(self, form, key, title, low, high, step, suffix, tip):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        name = QtWidgets.QLabel(title)
        name.setFixedWidth(94)
        name.setToolTip(tip)
        row.addWidget(name)
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setSuffix(suffix)
        box.setValue(gs.DEFAULTS[key])
        box.setToolTip(tip)
        box.setFixedWidth(82)
        row.addWidget(box)
        hint = QtWidgets.QLabel('')
        hint.setStyleSheet('color: gray;')
        row.addWidget(hint, 1)
        form.addLayout(row)
        self._spins[key] = box
        self._hints[key] = hint
        box.valueChanged.connect(self._update_hints)
        return box

    def _group_detect(self):
        group = QtWidgets.QGroupBox('Detect')
        form = QtWidgets.QVBoxLayout(group)
        self._spins = {}
        self._hints = {}
        self._method_checks = {}

        note = QtWidgets.QLabel(
            'Three ways of looking, not three kinds of thing. Everything '
            'they find comes out as one list of regions.')
        note.setWordWrap(True)
        note.setStyleSheet('color: gray;')
        form.addWidget(note)

        # --- outlines
        inner = self._method_group(
            form, 'outline',
            'Regions walled off by their own outline. The height of a region '
            'plays no part, so an object whose parts sit at different levels '
            'still comes out whole.')
        self._spin(inner, 'detail', 'Detail', 0.05, 20.0, 0.1, ' %',
                   'The scale the outlines are measured at - about the width '
                   'of the thinnest wall worth seeing. Larger ignores fine '
                   'texture and rounds off corners.')
        self._spin(inner, 'edge_level', 'Edge strength', 0.1, 10.0, 0.25, ' sig',
                   'How sharp a change has to be before it counts as a wall. '
                   'Lower walls more of the image off into smaller pieces.')
        self._spin(inner, 'close_gaps', 'Bridge gaps', 0.0, 10.0, 0.1, ' %',
                   'Breaks in an outline narrower than this are bridged, so a '
                   'contour the threshold nicked still encloses its region.')
        self._spin(inner, 'smoothness', 'Smoothness', 0.0, 5.0, 0.1, ' x',
                   'How smooth the inside of a region has to be, against the '
                   'typical roughness of the whole image. Below 1 keeps only '
                   'the smoother-than-average patches, which is usually what '
                   'separates an object from the textured field it sits on. '
                   '0 turns the test off and keeps every patch.')

        # --- raised
        inner = self._method_group(
            form, 'rise',
            'Anything standing above its own immediate surroundings. The '
            'comparison is local, so it survives a tilted or uneven field.')
        self._spin(inner, 'rise_window', 'Window', 0.1, 20.0, 0.2, ' %',
                   'The distance the local background is measured over. '
                   'Wider than the things being looked for.')
        self._spin(inner, 'rise_level', 'Rise', 0.5, 10.0, 0.25, ' sig',
                   'How far above that background a region has to stand, in '
                   'robust standard deviations. Lower catches more.')

        # --- ridges
        inner = self._method_group(
            form, 'ridge',
            'Long narrow crests that have no inside for the outline detector '
            'to find. Found by curvature rather than by height, because they '
            'are often no taller than the texture around them.')
        self._spin(inner, 'ridge_width', 'Width', 0.05, 10.0, 0.1, ' %',
                   'How wide the crests are. This is the setting that matters '
                   'most here: too narrow and the texture comes through, too '
                   'wide and neighbouring crests smear together.')
        self._spin(inner, 'ridge_level', 'Strength', 0.5, 10.0, 0.25, ' sig',
                   'Threshold on the curvature response.')
        self._spin(inner, 'ridge_length', 'Length', 0.5, 60.0, 0.5, ' %',
                   'How far a mark has to run before it counts as a crest '
                   'rather than as texture. This is what separates a real '
                   'line from a grain that happens to look like one.')

        # --- everything
        every = QtWidgets.QGroupBox('Every region')
        inner = QtWidgets.QVBoxLayout(every)
        self._spin(inner, 'min_size', 'Smallest', 0.0, 60.0, 0.1, ' %',
                   'Regions smaller than a square of this side are thrown '
                   'away, whichever detector found them.')
        self._spin(inner, 'max_size', 'Largest', 0.0, 100.0, 0.5, ' %',
                   'Regions larger than a square of this side are thrown '
                   'away. 0 means no limit.')
        self._spin(inner, 'separate', 'Separate', 0.0, 60.0, 0.5, ' %',
                   'Cut regions apart where two of them have grown together, '
                   'so each can be kept on its own. Roughly the width of the '
                   'smallest thing that should come out separately. 0 leaves '
                   'every touching group as one region.')
        form.addWidget(every)

        row = QtWidgets.QHBoxLayout()
        button = QtWidgets.QPushButton('Detect')
        button.setDefault(False)
        button.setAutoDefault(False)
        button.setToolTip('Run the ticked detectors over the whole frame. '
                          'Hand corrections are kept.')
        button.clicked.connect(self.detect)
        row.addWidget(button, 1)
        reset = QtWidgets.QPushButton('Defaults')
        reset.setAutoDefault(False)
        reset.setMaximumWidth(80)
        reset.clicked.connect(self.reset_settings)
        row.addWidget(reset)
        form.addLayout(row)
        return group

    def _method_group(self, form, method, tip):
        """A checkable sub-box, so a detector's settings grey out with it."""
        box = QtWidgets.QGroupBox(gs.METHOD_TITLES[method])
        box.setCheckable(True)
        box.setChecked(True)
        box.setToolTip(tip)
        inner = QtWidgets.QVBoxLayout(box)
        inner.setSpacing(3)
        form.addWidget(box)
        self._method_checks[method] = box
        return inner

    def _group_regions(self):
        group = QtWidgets.QGroupBox('Regions')
        form = QtWidgets.QVBoxLayout(group)

        self.region_label = QtWidgets.QLabel('-')
        self.region_label.setStyleSheet('color: gray;')
        form.addWidget(self.region_label)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        for text, slot, tip in (
                ('Keep all', lambda: self._all(True), 'Keep every region.'),
                ('Drop all', lambda: self._all(False),
                 'Drop every region - the starting point for picking out a '
                 'few by hand.'),
                ('Invert', self._invert, 'Swap kept and dropped.')):
            b = QtWidgets.QPushButton(text)
            b.setAutoDefault(False)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        form.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        b = QtWidgets.QPushButton('Erase dropped')
        b.setAutoDefault(False)
        b.setToolTip('Take every dropped region off the map for good, so what '
                     'is left is only what you meant to keep.')
        b.clicked.connect(self._erase_dropped)
        row.addWidget(b)
        b = QtWidgets.QPushButton('Erase all')
        b.setAutoDefault(False)
        b.setToolTip('Clear the map completely and start from nothing - draw '
                     'boxes round the objects you want, or paint them in.')
        b.clicked.connect(self._erase_all)
        row.addWidget(b)
        form.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.mask_check = QtWidgets.QCheckBox('Show the mask')
        self.mask_check.setChecked(True)
        self.mask_check.setToolTip(
            'Off shows the bare image, for checking that a region really is '
            'where the overlay says it is.')
        self.mask_check.toggled.connect(self.view.set_show_mask)
        row.addWidget(self.mask_check)

        self.grey_check = QtWidgets.QCheckBox('Grey image')
        self.grey_check.setToolTip(
            'Draw the heights in grey instead of the gradient. An AFM colour '
            'map is warm and light nearly everywhere, which is the hardest '
            'thing to read a coloured overlay against.')
        self.grey_check.toggled.connect(self._on_grey)
        row.addWidget(self.grey_check)
        row.addStretch(1)
        form.addLayout(row)
        return group

    def _group_tools(self):
        group = QtWidgets.QGroupBox('Tools')
        form = QtWidgets.QVBoxLayout(group)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        self._tool_buttons = {}
        for name, title, tip in TOOLS:
            b = QtWidgets.QToolButton()
            b.setText(title)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Preferred)
            b.clicked.connect(lambda _=False, n=name: self._set_tool(n))
            row.addWidget(b)
            self._tool_buttons[name] = b
        self._tool_buttons['pick'].setChecked(True)
        form.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel('Box')
        name.setFixedWidth(50)
        row.addWidget(name)
        self.box_combo = QtWidgets.QComboBox()
        for key, title, tip in BOX_ACTIONS:
            self.box_combo.addItem(title, key)
        self.box_combo.setToolTip(
            '\n\n'.join('%s - %s' % (title, tip) for _, title, tip in BOX_ACTIONS))
        self.box_combo.currentIndexChanged.connect(self._on_box_action)
        row.addWidget(self.box_combo, 1)
        form.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel('Brush')
        name.setFixedWidth(50)
        row.addWidget(name)
        self.brush_spin = QtWidgets.QDoubleSpinBox()
        self.brush_spin.setRange(0.05, 20.0)
        self.brush_spin.setSingleStep(0.1)
        self.brush_spin.setDecimals(2)
        self.brush_spin.setSuffix(' %')
        self.brush_spin.setValue(1.0)
        self.brush_spin.setFixedWidth(82)
        self.brush_spin.valueChanged.connect(self._on_brush)
        row.addWidget(self.brush_spin)
        self.brush_hint = QtWidgets.QLabel('')
        self.brush_hint.setStyleSheet('color: gray;')
        row.addWidget(self.brush_hint, 1)
        form.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        clear = QtWidgets.QPushButton('Clear painting')
        clear.setAutoDefault(False)
        clear.setToolTip('Throw away every brush and erase stroke, and leave '
                         'the detected regions alone.')
        clear.clicked.connect(self._clear_painting)
        row.addWidget(clear)
        fit = QtWidgets.QPushButton('Fit view')
        fit.setAutoDefault(False)
        fit.clicked.connect(self.view.fit)
        row.addWidget(fit)
        form.addLayout(row)

        note = QtWidgets.QLabel('Wheel zooms, middle button pans.')
        note.setStyleSheet('color: gray;')
        form.addWidget(note)
        return group

    # ---------------------------------------------------------------- work

    def settings(self):
        return {key: box.value() for key, box in self._spins.items()}

    def methods(self):
        return tuple(m for m in gs.METHODS
                     if self._method_checks[m].isChecked())

    def _run(self, window=None):
        """One detector pass, with the cursor and the error box seen to."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return gs.segment(self.surface.z, self.surface.x_real,
                              self.surface.y_real, methods=self.methods(),
                              window=window, **self.settings())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Segmentation',
                'Could not segment that channel.\n\n%s: %s'
                % (type(exc).__name__, exc))
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def detect(self):
        """Run the detectors over the whole frame, keeping hand corrections."""
        if not self.methods():
            self.status.setText('No detector is ticked - nothing to look for.')
            return
        old = self.view.segmentation
        seg = self._run()
        if seg is None:
            return
        if old is not None and old.shape == seg.shape:
            seg.added = old.added.copy()
            seg.removed = old.removed.copy()
        self._install(seg)
        note = ''
        if gs.watershed_source() != 'scikit-image':
            # Not a crash and not a warning about the data - a statement that
            # a better answer is one "pip install scikit-image" away, made
            # where it is relevant rather than buried in a README.
            note = ('   (scikit-image is not installed, so objects that touch '
                    'are separated less well)')
        self.status.setText('Found %d region%s over the whole frame.%s'
                            % (seg.region_count,
                               '' if seg.region_count == 1 else 's', note))

    def detect_in(self, bounds):
        """Search inside one dragged rectangle and merge the answer in.

        The thresholds are measured inside the box, which is the whole point:
        a faint object that the frame-wide statistics drowned comes out as
        soon as the box is drawn round it.
        """
        seg = self.view.segmentation
        if seg is None or not self.methods():
            self.status.setText('No detector is ticked - nothing to look for.')
            return
        r0, r1, c0, c1 = bounds
        found = self._run(window=bounds)
        if found is None:
            return
        added = seg.absorb(found, (slice(r0, r1 + 1), slice(c0, c1 + 1)))
        self.view.refresh()
        self._update_summary()
        self.status.setText(
            'Searched a %d x %d box - %d region%s.'
            % (r1 - r0 + 1, c1 - c0 + 1, added, '' if added == 1 else 's'))

    def _install(self, seg):
        self.view.set_segmentation(seg)
        self._update_hints()
        self._update_summary()

    def reset_settings(self):
        self._loading = True
        for key, box in self._spins.items():
            box.setValue(gs.DEFAULTS[key])
        for box in self._method_checks.values():
            box.setChecked(True)
        self._loading = False
        self._update_hints()

    def _all(self, kept):
        seg = self.view.segmentation
        if seg is None:
            return
        seg.keep_all(kept)
        self.view.refresh()
        self._update_summary()

    def _invert(self):
        seg = self.view.segmentation
        if seg is None:
            return
        seg.invert()
        self.view.refresh()
        self._update_summary()

    def _erase_dropped(self):
        seg = self.view.segmentation
        if seg is None:
            return
        dropped = np.flatnonzero(seg.alive[1:] & ~seg.keep[1:]) + 1
        gone = seg.erase(dropped)
        self.view.refresh()
        self._update_summary()
        self.status.setText('Erased %d dropped region%s.'
                            % (gone, '' if gone == 1 else 's'))

    def _erase_all(self):
        seg = self.view.segmentation
        if seg is None:
            return
        seg.erase(np.arange(1, seg.count + 1))
        seg.clear_painting()
        self.view.refresh()
        self._update_summary()
        self.status.setText('Map cleared. Draw a box round something and '
                            'search inside it, or paint with the brush.')

    def _clear_painting(self):
        seg = self.view.segmentation
        if seg is None:
            return
        seg.clear_painting()
        self.view.refresh()
        self._update_summary()

    def _set_tool(self, name):
        for key, button in self._tool_buttons.items():
            button.setChecked(key == name)
        self.view.set_tool(name)

    def _on_box_action(self, *_):
        self.view.set_box_action(self.box_combo.currentData())
        self._set_tool('box')

    def _on_grey(self, on):
        import matplotlib
        self.view.set_image(self.surface.z,
                            matplotlib.colormaps['gray'] if on else self._cmap,
                            self._clim)

    def _on_brush(self, value):
        self.view.set_brush(value)
        self._update_hints()

    def _show_hover(self, text):
        self.status.setText(text)

    def _update_hints(self):
        """Say what each percentage currently works out to on this scan."""
        seg = self.view.segmentation
        scale = (seg.scale if seg is not None
                 else gs.Scale(self.surface.z.shape,
                               self.surface.x_real, self.surface.y_real))
        unit = self.surface.xy_unit or ''
        for key, hint in self._hints.items():
            box = self._spins[key]
            if key.endswith('_level') or key == 'smoothness':
                hint.setText('')
            elif box.value() <= 0.0:
                hint.setText('= off')
            else:
                hint.setText('= %s' % scale.describe(box.value(), unit))
        self.brush_hint.setText('= %s wide'
                                % scale.describe(self.brush_spin.value(), unit))

    def _update_summary(self):
        seg = self.view.segmentation
        if seg is None:
            self.summary.setText('')
            self.region_label.setText('-')
            return
        total, kept = seg.region_count, seg.kept_count
        self.region_label.setText(
            '%d region%s on the map, %d kept'
            % (total, '' if total == 1 else 's', kept))
        self.summary.setText(
            'Keeping %.1f %% of the frame - %d of %d regions%s'
            % (100.0 * seg.mask().mean(), kept, total,
               ', plus painting' if seg.painted else ''))

    def accept(self):
        self.result_segmentation = self.view.segmentation
        super(SegmentDialog, self).accept()
