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
"""

import os

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


def figure_panel(parent, figsize, dpi=100):
    """A matplotlib figure with its canvas and navigation toolbar."""
    figure = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setParent(parent)
    canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
    toolbar = NavigationToolbar2QT(canvas, parent)
    return figure, canvas, toolbar


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


def group(title, layout):
    """A titled box around a layout."""
    box = QtWidgets.QGroupBox(title)
    box.setLayout(layout)
    return box


def row(*widgets, spacing=4, margins=(0, 0, 0, 0), stretch_at_end=False):
    """A horizontal layout of widgets; an integer stands for a stretch."""
    layout = QtWidgets.QHBoxLayout()
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    for item in widgets:
        if isinstance(item, int):
            layout.addStretch(item)
        elif isinstance(item, QtWidgets.QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    if stretch_at_end:
        layout.addStretch(1)
    return layout


def column(*widgets, spacing=4, margins=(0, 0, 0, 0)):
    """A vertical layout of widgets; an integer stands for a stretch."""
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    for item in widgets:
        if isinstance(item, int):
            layout.addStretch(item)
        elif isinstance(item, QtWidgets.QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    return layout


def button(text, callback, width=None):
    widget = QtWidgets.QPushButton(text)
    widget.clicked.connect(lambda *_a: callback())
    if width is not None:
        widget.setFixedWidth(width)
    return widget
