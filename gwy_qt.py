"""
The Qt joinery the front end is built out of.

Nothing here knows anything about AFM. It is the small pile of things every
window in `gwy_processor_gui` needs and that Qt does not provide in the shape
this program wants them.

The observable value is the piece that earns its place. A dialog here is a set
of parameters that several widgets and several panels all look at, and the
useful question is "what changed" rather than "which widget emitted". `Var`
holds one value, tells whoever asked when it changes, and - the part that
matters - remembers text that is not a number yet. Half of a live-preview
dialog is deciding what to do while someone is in the middle of typing
`0.0` and the entry momentarily reads `0.`; `Var.get()` raises `VarError`
there, and every reader is written to leave the last good picture alone
until the typing lands somewhere valid.

The rest is joinery of the same kind: a restartable single-shot timer, since
debouncing a preview is not the same thing as scheduling one; the standard
question-and-answer boxes behind names that say what they ask; a plain window
that says whether it is still open, because a diagnostics window is opened,
closed and re-opened while the dialog that owns it keeps a handle on it; and
a matplotlib canvas with its toolbar, made the same way every time.

Three of them are here because the windows have to survive being resized. Qt's
box layouts do not wrap, so a bar of controls sets the smallest width the
window can ever have: `FlowLayout` wraps it onto a second line instead. A long
file path does the same thing to a label, so `ElidedLabel` shortens the text
rather than the window's freedom. And a matplotlib figure keeps the margins it
was laid out with, which stop fitting the moment the canvas changes shape, so
`figure_panel` re-runs `tight_layout` once a resize has settled.
"""

import os
import warnings

# qtpy takes the first Qt binding it can import and PyQt5 usually wins that
# race. Say so explicitly when PySide6 is installed - the binding the 3D
# viewer is written against, and the one this was tested on - but leave an
# existing choice alone. Two bindings loaded into one process crash it, so
# this has to happen before anything imports qtpy.
if 'QT_API' not in os.environ:
    try:
        import PySide6  # noqa: F401
        os.environ['QT_API'] = 'pyside6'
    except ImportError:
        pass

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Observable values
# ---------------------------------------------------------------------------

class VarError(ValueError):
    """`Var.get()` was called while the widget holds text that is not a
    value of that kind - almost always because someone is still typing."""


class Var(QtCore.QObject):
    """One value, plus everyone who wants to hear about it changing.

    `set` takes a value, `set_text` takes what a widget's edit box says.
    Text that does not parse is kept as-is: `text()` still returns it, so the
    widget is not fought with mid-word, but `get()` raises `VarError` until
    it parses again. Readers are expected to catch that and do nothing.

    Listeners are notified on a real change only, and the parse state counts
    as part of the value: text going from unparseable back to the number it
    already held is a change, because the dialog that gave up on the last
    keystroke has to be told to try again.
    """

    changed = QtCore.Signal()

    def __init__(self, value=None):
        super().__init__()
        self._value = value
        self._raw = None                 # text that would not parse
        self._callbacks = []

    # ---- reading ----

    def get(self):
        if self._raw is not None:
            raise VarError(f"{self._raw!r} is not a valid value")
        return self._value

    def text(self):
        """What a widget bound to this should be showing."""
        if self._raw is not None:
            return self._raw
        return "" if self._value is None else str(self._value)

    # ---- writing ----

    def set(self, value):
        value = self._coerce(value)
        if self._raw is None and value == self._value:
            return
        self._raw = None
        self._value = value
        self._fire()

    def set_text(self, text):
        """Take what a widget says, whether or not it parses yet."""
        stale = self._raw is not None
        try:
            value = self._coerce(text)
        except (TypeError, ValueError):
            if self._raw == text:
                return
            self._raw = text
            self._fire()
            return
        self._raw = None
        if not stale and value == self._value:
            return
        self._value = value
        self._fire()

    def _coerce(self, value):
        return value

    # ---- listeners ----

    def trace_add(self, callback):
        """Call `callback` whenever the value changes. Takes no arguments;
        a listener that wants the value asks for it."""
        self._callbacks.append(callback)

    def _fire(self):
        for callback in list(self._callbacks):
            callback()
        self.changed.emit()


class StringVar(Var):
    def __init__(self, value=""):
        super().__init__(str(value))

    def _coerce(self, value):
        return str(value)


class IntVar(Var):
    def __init__(self, value=0):
        super().__init__(int(value))

    def _coerce(self, value):
        return int(value)


class FloatVar(Var):
    def __init__(self, value=0.0):
        super().__init__(float(value))

    def _coerce(self, value):
        return float(value)


class BoolVar(Var):
    def __init__(self, value=False):
        super().__init__(bool(value))

    def _coerce(self, value):
        return bool(value)


# ---------------------------------------------------------------------------
# Binding a widget to a Var
# ---------------------------------------------------------------------------

def _bind(widget, var, signal, push, pull):
    """Keep a widget and a Var showing the same thing.

    `push()` sends the widget's state into the var, `pull()` puts the var's
    value back on the widget. The guard is one-directional on purpose: while
    the widget is driving, the var does not write back into it, so nobody's
    cursor jumps mid-word. Every other listener still hears the change.
    """
    state = {"busy": False}

    def widget_changed(*_args):
        if state["busy"]:
            return
        state["busy"] = True
        try:
            push()
        finally:
            state["busy"] = False

    def var_changed():
        if state["busy"]:
            return
        state["busy"] = True
        try:
            pull()
        finally:
            state["busy"] = False

    signal.connect(widget_changed)
    var.changed.connect(var_changed)
    widget._gwy_var = var       # a Var is parentless, so the widget holds it
    pull()


def bind_edit(widget, var):
    """A QLineEdit showing whatever the var holds, valid or not."""
    def pull():
        text = var.text()
        if widget.text() != text:
            widget.setText(text)
    _bind(widget, var, widget.textChanged, lambda: var.set_text(widget.text()),
          pull)
    return widget


def bind_spin(widget, var):
    """A QSpinBox / QDoubleSpinBox. Qt validates as you type, so unlike a
    free-text entry this one never leaves the var un-parseable."""
    def pull():
        try:
            value = var.get()
        except VarError:
            return
        if widget.value() != value:
            widget.setValue(value)
    _bind(widget, var, widget.valueChanged, lambda: var.set(widget.value()),
          pull)
    return widget


def bind_combo(widget, var):
    def pull():
        text = str(var.text())
        if widget.currentText() == text:
            return
        widget.setCurrentText(text)
        if widget.currentText() != text:
            # not one of the offered choices; a read-only combo refuses it
            # silently, so take the refusal back to the var rather than let
            # the two disagree about what is selected
            var.set(widget.currentText())
    _bind(widget, var, widget.currentTextChanged,
          lambda: var.set(widget.currentText()), pull)
    return widget


def bind_check(widget, var):
    def pull():
        checked = bool(var.get())
        if widget.isChecked() != checked:
            widget.setChecked(checked)
    _bind(widget, var, widget.toggled, lambda: var.set(widget.isChecked()),
          pull)
    return widget


class FloatSlider(QtWidgets.QSlider):
    """Qt's slider is integer-only; this one runs over a float range."""

    STEPS = 1000

    def __init__(self, low, high, var, parent=None):
        super().__init__(QtCore.Qt.Horizontal, parent)
        self._low, self._high = float(low), float(high)
        self.setRange(0, self.STEPS)
        span = (self._high - self._low) or 1.0

        def pull():
            try:
                value = float(var.get())
            except (VarError, TypeError, ValueError):
                return
            step = int(round((value - self._low) / span * self.STEPS))
            if self.value() != step:
                self.setValue(step)

        _bind(self, var, self.valueChanged,
              lambda: var.set(self._low + span * self.value() / self.STEPS),
              pull)


def set_items(combo, values, keep=True):
    """Replace a read-only combo's list. The current text is kept when it is
    still one of the choices, which is what every 'reload the channel list'
    caller wants."""
    current = combo.currentText()
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems([str(v) for v in values])
        if keep and current in [str(v) for v in values]:
            combo.setCurrentText(current)
    finally:
        combo.blockSignals(blocked)


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

class Timer:
    """A single-shot timer that can be restarted or called off.

    Every live preview in this program is debounced: a parameter changes, and
    the recomputation is scheduled for a moment later so that holding a spin
    button does not run the operation once per click. Restarting is therefore
    the common case, not scheduling.
    """

    def __init__(self, owner, callback=None):
        self._timer = QtCore.QTimer(owner)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fire)
        self._callback = callback

    def start(self, milliseconds, callback=None):
        if callback is not None:
            self._callback = callback
        self._timer.start(int(milliseconds))

    def cancel(self):
        self._timer.stop()

    def active(self):
        return self._timer.isActive()

    def _fire(self):
        if self._callback is not None:
            self._callback()


def process_events():
    """Let the window repaint in the middle of a long loop."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()


# ---------------------------------------------------------------------------
# Asking the user something
# ---------------------------------------------------------------------------

def show_error(parent, title, message):
    QtWidgets.QMessageBox.critical(parent, title, message)


def show_info(parent, title, message):
    QtWidgets.QMessageBox.information(parent, title, message)


def ask_yes_no(parent, title, message):
    return QtWidgets.QMessageBox.question(
        parent, title, message,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes


def ask_ok_cancel(parent, title, message):
    return QtWidgets.QMessageBox.question(
        parent, title, message,
        QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
        QtWidgets.QMessageBox.Ok) == QtWidgets.QMessageBox.Ok


def ask_directory(parent, title, initialdir=""):
    return QtWidgets.QFileDialog.getExistingDirectory(
        parent, title, initialdir or "") or None


def ask_open_filename(parent, title, filters, initialdir=""):
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, title, initialdir or "", filters)
    return path or None


def ask_save_filename(parent, title, filters, initialfile="", initialdir="",
                      confirm_overwrite=True):
    start = os.path.join(initialdir or "", initialfile or "")
    options = QtWidgets.QFileDialog.Options()
    if not confirm_overwrite:
        options |= QtWidgets.QFileDialog.DontConfirmOverwrite
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, title, start, filters, "", options)
    return path or None


# ---------------------------------------------------------------------------
# Windows and widgets
# ---------------------------------------------------------------------------

class ToolWindow(QtWidgets.QWidget):
    """A window that belongs to another window and says when it has closed.

    Every diagnostics and zoom view here is opened from a dialog, closed by
    the user, and expected to be re-openable from the same button - while the
    dialog goes on holding a handle to it and pushing updates. `is_open` is
    what that handle is tested with.
    """

    closed = QtCore.Signal()

    def __init__(self, parent, title, size=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle(title)
        if size is not None:
            self.resize(*size)
        self._open = True

    def is_open(self):
        return self._open

    def raise_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        self._open = False
        self.closed.emit()
        super().closeEvent(event)


def is_open(window):
    """True when `window` exists and has not been closed."""
    return window is not None and window.is_open()


def figure_panel(parent, figsize, dpi=100, responsive=True):
    """A matplotlib figure with its canvas and navigation toolbar.

    `figsize` is the shape the panel would like to be, not the shape it keeps:
    the canvas grows and shrinks with the window, and `responsive` re-fits the
    figure's margins whenever it does (see `_keep_tight`).
    """
    figure = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setParent(parent)
    canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
    canvas.setMinimumSize(120, 100)
    toolbar = NavigationToolbar2QT(canvas, parent)
    if responsive:
        _keep_tight(figure, canvas)
    return figure, canvas, toolbar


def _keep_tight(figure, canvas):
    """Re-fit the figure's margins after the canvas has been resized.

    Every panel here ends its drawing with `tight_layout`, which measures the
    labels and the colour bars once and writes the margins it needs as
    fractions of the canvas. Those fractions stop being right as soon as the
    canvas changes shape - the axes keep their share of a canvas half the
    height, and the tick labels grow into the title. Re-running the fit puts
    that back. It waits for the drag to stop first, because a resize arrives
    once per pixel and a re-fit is not free.
    """
    timer = Timer(canvas)

    def margins():
        p = figure.subplotpars
        return (p.left, p.right, p.bottom, p.top)

    def settled():
        if not figure.axes:
            return
        try:
            # A panel with a hand-placed axes says so as a warning and fits it
            # badly; that is a reason to leave this figure's margins alone, so
            # the warning is read as the refusal it is.
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                # One pass measures the labels where the *previous* size left
                # them, and on an image with a fixed aspect ratio it lands
                # short - far enough short, on a small window, to put the y
                # axis label off the edge of the canvas. A second pass measures
                # the answer the first one gave and settles.
                for _ in range(3):
                    before = margins()
                    figure.tight_layout()
                    if margins() == before:
                        break
        except Exception:
            return
        canvas.draw_idle()

    canvas._tight_timer = timer     # the canvas owns it, so it stays alive
    canvas.mpl_connect("resize_event", lambda _e: timer.start(120, settled))


class ColourStrip(QtWidgets.QWidget):
    """The chosen false-colour gradient, painted as a strip."""

    def __init__(self, parent=None, height=18):
        super().__init__(parent)
        self.setFixedHeight(height)
        self._cmap = None

    def set_cmap(self, cmap):
        self._cmap = cmap
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect()
        if self._cmap is not None and rect.width() > 1:
            gradient = QtGui.QLinearGradient(0, 0, rect.width(), 0)
            for i in range(65):
                t = i / 64.0
                r, g, b, _ = self._cmap(t)
                gradient.setColorAt(t, QtGui.QColor.fromRgbF(r, g, b))
            painter.fillRect(rect, gradient)
        painter.setPen(QtGui.QColor("#909090"))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))


def label(text="", wrap=False, colour=None, bold=False, fixed=False,
          align=None):
    """A QLabel, with the four things this program keeps asking of one."""
    widget = QtWidgets.QLabel(text)
    if wrap:
        widget.setWordWrap(True)
        # A wrapped label is taller the narrower it is, and a layout only asks
        # about that if the size policy says to. Without this the last line is
        # cut off whenever the text wraps to more lines than it started with.
        policy = widget.sizePolicy()
        policy.setHeightForWidth(True)
        widget.setSizePolicy(policy)
    if colour is not None:
        widget.setStyleSheet(f"color: {colour};")
    if bold or fixed:
        font = widget.font()
        font.setBold(bool(bold))
        if fixed:
            font = QtGui.QFontDatabase.systemFont(
                QtGui.QFontDatabase.FixedFont)
        widget.setFont(font)
    if align is not None:
        widget.setAlignment(align)
    return widget


def muted_colour():
    """A dimmed version of the window text colour, for the secondary lines -
    a hint, a count, a report. A fixed grey reads on one theme and vanishes
    on the other, so this follows whichever the desktop is set to."""
    palette = QtWidgets.QApplication.palette()
    return palette.color(QtGui.QPalette.Disabled,
                         QtGui.QPalette.WindowText).name()


WARNING_COLOUR = "#e08a3c"      # readable on a light and a dark background


def bound_label(var, **kwargs):
    """A QLabel that says whatever the var says."""
    widget = label(str(var.text()), **kwargs)
    var.trace_add(lambda: widget.setText(str(var.text())))
    return widget


class ElidedLabel(QtWidgets.QLabel):
    """A label that shortens its text instead of insisting on room for it all.

    This is what the file and folder names are shown in. A path is as long as
    it happens to be, and an ordinary label would take that length as a demand:
    the window could then never be made narrower than someone's directory
    layout. This one shows as much as fits, puts the rest under the pointer as
    a tooltip, and asks for nothing.
    """

    def __init__(self, text="", parent=None, minimum=60, preferred=260):
        super().__init__(text, parent)
        self._full = text
        self._minimum = minimum
        self._preferred = preferred
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)

    def setText(self, text):
        self._full = str(text)
        self.setToolTip(self._full)
        self.updateGeometry()       # a new name may want a new width
        self._elide()

    def text(self):
        """The whole text, not the shortened one that is on screen."""
        return self._full

    def _wanted(self):
        """How wide the whole text would be.

        Measured from the text that was set, not from the shortened text on
        screen - asking QLabel would ask about the ellipsis, and a width taken
        from that would shorten the text again next time round until nothing
        was left of it."""
        return QtGui.QFontMetrics(self.font()).horizontalAdvance(self._full)

    def sizeHint(self):
        """A width it would like, not the width the whole text would need."""
        hint = super().sizeHint()
        hint.setWidth(min(self._wanted() + 2, self._preferred))
        return hint

    def minimumSizeHint(self):
        """And a width it can be cut down to, which is the whole point: a
        label that insisted on its text is what this exists not to be."""
        hint = super().minimumSizeHint()
        hint.setWidth(min(self._minimum, self._wanted() + 2))
        return hint

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        metrics = QtGui.QFontMetrics(self.font())
        super().setText(metrics.elidedText(
            self._full, QtCore.Qt.ElideMiddle, max(self.width() - 2, 20)))


def group(title, layout):
    """A titled box around a layout."""
    box = QtWidgets.QGroupBox(title)
    box.setLayout(layout)
    return box


def _fill(layout, widgets):
    """An integer stands for a stretch, a (widget, n) pair for one that takes
    n shares of whatever room is spare."""
    for item in widgets:
        if isinstance(item, int):
            layout.addStretch(item)
        elif isinstance(item, tuple):
            thing, share = item
            if isinstance(thing, QtWidgets.QLayout):
                layout.addLayout(thing, share)
            else:
                layout.addWidget(thing, share)
        elif isinstance(item, QtWidgets.QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    return layout


def row(*widgets, spacing=4, margins=(0, 0, 0, 0), stretch_at_end=False):
    """A horizontal layout of widgets."""
    layout = QtWidgets.QHBoxLayout()
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    _fill(layout, widgets)
    if stretch_at_end:
        layout.addStretch(1)
    return layout


def column(*widgets, spacing=4, margins=(0, 0, 0, 0)):
    """A vertical layout of widgets."""
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return _fill(layout, widgets)


def button(text, callback, width=None):
    widget = QtWidgets.QPushButton(text)
    widget.clicked.connect(lambda *_a: callback())
    if width is not None:
        widget.setFixedWidth(width)
    return widget


class FlowLayout(QtWidgets.QLayout):
    """A horizontal layout that wraps onto the next line when it runs out.

    Qt's box layouts do not wrap. A bar of a dozen controls laid out in a row
    therefore sets the smallest width its window can ever have, and the tabs
    here have bars like that. This lays the same controls out left to right and
    starts a new line when the next one would not fit, so the window can be
    made as narrow as the widest single control and the bar simply gets taller.

    Hidden widgets are skipped rather than left as holes, which is what the
    operation dialogs need: they hide the parameters that the chosen method
    does not have.
    """

    def __init__(self, parent=None, spacing=6, margins=(0, 0, 0, 0)):
        super().__init__(parent)
        self._items = []
        self._space = spacing
        self.setContentsMargins(*margins)

    # ---- the five QLayout has no default for ----

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    # ---- height depends on width; that is the whole point ----

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._lay_out(QtCore.QRect(0, 0, width, 0), place=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._lay_out(rect, place=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._shown():
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(margins.left() + margins.right(),
                                   margins.top() + margins.bottom())

    def _shown(self):
        for item in self._items:
            widget = item.widget()
            if widget is None or not widget.isHidden():
                yield item

    def _lay_out(self, rect, place):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._shown():
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + self._space
                line_height = 0
            if place:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._space
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def flow(*widgets, spacing=6, margins=(0, 0, 0, 0)):
    """A `FlowLayout` of widgets; a layout is wrapped in a widget of its own."""
    layout = FlowLayout(spacing=spacing, margins=margins)
    for item in widgets:
        if isinstance(item, QtWidgets.QLayout):
            holder = QtWidgets.QWidget()
            holder.setLayout(item)
            layout.addWidget(holder)
        else:
            layout.addWidget(item)
    return layout


def cluster(*widgets, spacing=4):
    """Widgets that belong together, as one widget for a `FlowLayout`.

    A label and the box it names have to wrap as one thing, or a narrow window
    leaves "Cell size (% of frame):" at the end of one line and its value at
    the start of the next.
    """
    holder = QtWidgets.QWidget()
    holder.setLayout(row(*widgets, spacing=spacing))
    return holder


def scroll(widget):
    """`widget` in a scroll area, so a short window can still reach all of it."""
    area = QtWidgets.QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    return area


def splitter(orientation, *widgets, sizes=None, stretch=None):
    """A draggable divider between widgets, with the starting split given."""
    split = QtWidgets.QSplitter(orientation)
    for widget in widgets:
        split.addWidget(widget)
    split.setChildrenCollapsible(False)
    for index, factor in enumerate(stretch or ()):
        split.setStretchFactor(index, factor)
    if sizes is not None:
        split.setSizes(list(sizes))
    return split


def action(parent, text, callback, shortcut=None, tip=None):
    """A QAction, for the things that live on the ribbon."""
    act = QtGui.QAction(text, parent)
    act.triggered.connect(lambda *_a: callback())
    if shortcut is not None:
        act.setShortcut(QtGui.QKeySequence(shortcut))
        tip = f"{tip or text}  ({act.shortcut().toString()})"
    if tip is not None:
        act.setToolTip(tip)
        act.setStatusTip(tip)
    return act
