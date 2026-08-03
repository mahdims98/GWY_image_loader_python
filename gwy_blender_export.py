"""
Hand the surface to Blender for a path-traced still, from inside the viewer.

The live view approximates. Screen-space ambient occlusion darkens a crevice
by what happens to be near it *on screen*; a shadow map is a depth buffer
from the light's point of view; a reflection comes off a synthetic cube map
and not off the surface itself. All three are good enough to explore with,
and all three break in ways that show up in print.

Blender's Cycles does not approximate any of them - it traces the light. The
same mesh, the same colours and a real integrator, and the shadows are the
shadows the surface actually casts. It cannot run in a viewport at the size
we need, so it renders to a file and takes as long as it takes.

How the two programs meet. Blender contains a full Python interpreter with
`bpy` in it, and can be handed a script to run in that interpreter:

    blender --background --python gwy_blender_render.py -- scene.npz out.png

So this module writes `scene.npz` - the height grid, the vertex colours, and
the render settings as JSON - and runs that command as a subprocess. There
is no import of `bpy` on this side and no import of pyvista on that side.
The two halves never share an interpreter, which is what makes it work at
all: `bpy` on PyPI is pinned to the exact Python version its Blender was
built for, and pip-installing it into a working scientific environment is a
fight you do not need to have.

Dropping ``--background`` opens Blender's own window with the scene already
built. That is the one to use while learning: the mesh, the material nodes,
the lights and the camera are all there to be inspected.

Blender does not have to be installed for the rest of the viewer to work.
Nothing here is imported until the export menu is used, and if no Blender
is found the dialog still writes the scene file and shows the command to
run against it later.
"""

import glob
import json
import os
import shutil
import sys

import numpy as np
from qtpy import QtCore, QtWidgets


RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'gwy_blender_render.py')


# ------------------------------------------------------------------ blender

def _candidates():
    """Where a Blender install tends to be, newest first."""
    found = []
    env = os.environ.get('GWY_BLENDER')
    if env:
        found.append(env)
    on_path = shutil.which('blender')
    if on_path:
        found.append(on_path)
    if sys.platform.startswith('win'):
        patterns = [
            r'C:\Program Files\Blender Foundation\Blender*\blender.exe',
            r'C:\Program Files (x86)\Blender Foundation\Blender*\blender.exe',
            os.path.expanduser(
                r'~\AppData\Local\Programs\Blender Foundation\Blender*\blender.exe'),
            r'C:\Program Files\Blender Foundation\Blender*\*\blender.exe',
        ]
    elif sys.platform == 'darwin':
        patterns = ['/Applications/Blender.app/Contents/MacOS/Blender',
                    os.path.expanduser(
                        '~/Applications/Blender.app/Contents/MacOS/Blender')]
    else:
        patterns = ['/usr/bin/blender', '/usr/local/bin/blender',
                    '/snap/bin/blender',
                    os.path.expanduser('~/.local/bin/blender')]
    for pattern in patterns:
        found.extend(sorted(glob.glob(pattern), reverse=True))
    return found


def find_blender():
    """The Blender executable, or None. `GWY_BLENDER` overrides the search."""
    for path in _candidates():
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# ------------------------------------------------------------------- colour

def colorize(values, cmap, clim):
    """The false colour of every value, as the render will show it.

    Values outside `clim` are clamped rather than dropped, which is what the
    viewer does too - a clipped peak reads as the end colour of the gradient
    and not as a hole.
    """
    lo, hi = float(clim[0]), float(clim[1])
    span = hi - lo if hi > lo else 1.0
    t = np.clip((np.asarray(values, dtype=np.float64) - lo) / span, 0.0, 1.0)
    t = np.nan_to_num(t, nan=0.0)
    rgba = np.asarray(cmap(t))
    return np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- scene file

def scene_arrays(surface_mesh, cmap, clim):
    """The grid, the spacing and the colours, in the shapes Blender wants."""
    surface = surface_mesh.surface
    ny, nx = surface.shape
    frame = surface.frame

    relief = surface_mesh.mesh.points[:, 2].reshape(ny, nx).astype(np.float32)
    heights = np.asarray(surface_mesh.mesh['height']).reshape(ny, nx)
    rgb = colorize(heights, cmap, clim)

    dx = (surface.x_real / (nx - 1)) / frame
    dy = (surface.y_real / (ny - 1)) / frame
    return relief, rgb, dx, dy


def write_scene(surface_mesh, path, cmap, clim, params=None):
    """Write the .npz the render script reads. Returns the path written."""
    relief, rgb, dx, dy = scene_arrays(surface_mesh, cmap, clim)
    settings = {
        'name': surface_mesh.surface.name,
        'metallic': 0.10,
        'roughness': 0.50,
        'engine': 'CYCLES',
        'samples': 128,
        'width': 1920,
        'height': 1280,
        'gpu': True,
        'transparent': False,
        'world_color': (0.05, 0.06, 0.07),
        'world_strength': 0.6,
        'exaggeration': surface_mesh.exaggeration,
        'z_label': surface_mesh.z_label,
        'clim': [float(clim[0]), float(clim[1])],
    }
    settings.update(params or {})
    np.savez_compressed(path, relief=relief, rgb=rgb,
                        dx=np.float64(dx), dy=np.float64(dy),
                        params=json.dumps(settings))
    return path


def command_for(blender, scene_path, output=None, background=True):
    """The exact command line, for running and for showing to the user."""
    argv = [blender or 'blender']
    if background:
        argv.append('--background')
    argv += ['--python', RENDER_SCRIPT, '--', scene_path]
    if output and background:
        argv.append(output)
    return argv


# -------------------------------------------------------------- mesh export

def save_mesh(surface_mesh, path, cmap=None, clim=None):
    """Write the surface as a mesh file, with vertex colours where the
    format carries them (PLY does; STL does not)."""
    poly = surface_mesh.mesh.extract_surface(algorithm='dataset_surface')
    if path.lower().endswith('.ply') and cmap is not None:
        poly['RGB'] = colorize(poly['height'], cmap,
                               clim or surface_mesh.height_range())
        poly.save(path, texture='RGB')
    else:
        poly.save(path)
    return path


# ------------------------------------------------------------------- dialog

class BlenderDialog(QtWidgets.QDialog):
    """Settings for the render, and the log of Blender running.

    Blender is driven through QProcess rather than subprocess so its output
    arrives line by line while it works. A Cycles render of a megapixel
    surface is minutes, not seconds, and a frozen window with no output is
    indistinguishable from a crash.
    """

    def __init__(self, parent, surface_mesh, cmap, clim, suggested,
                 metallic=0.10, roughness=0.50):
        super(BlenderDialog, self).__init__(parent)
        self.setWindowTitle('Render with Blender')
        self.resize(680, 560)

        self.mesh = surface_mesh
        self.cmap = cmap
        self.clim = clim
        self.metallic = metallic
        self.roughness = roughness
        self.process = None
        self.scene_path = None

        self._build(suggested)

    # ---- layout ----

    def _build(self, suggested):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        row = QtWidgets.QHBoxLayout()
        self.exe_edit = QtWidgets.QLineEdit(find_blender() or '')
        self.exe_edit.setPlaceholderText('path to blender.exe')
        browse = QtWidgets.QPushButton('Browse...')
        browse.clicked.connect(self._browse_exe)
        row.addWidget(self.exe_edit, 1)
        row.addWidget(browse)
        form.addRow('Blender', row)

        self.engine_combo = QtWidgets.QComboBox()
        self.engine_combo.addItems(['Cycles (path traced)', 'EEVEE (fast)'])
        form.addRow('Engine', self.engine_combo)

        self.samples_spin = QtWidgets.QSpinBox()
        self.samples_spin.setRange(8, 8192)
        self.samples_spin.setValue(128)
        self.samples_spin.setToolTip(
            'More samples means less noise and proportionally more time. '
            '128 with denoising is usually enough for a figure.')
        form.addRow('Samples', self.samples_spin)

        size = QtWidgets.QHBoxLayout()
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(64, 16384)
        self.width_spin.setValue(1920)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(64, 16384)
        self.height_spin.setValue(1280)
        size.addWidget(self.width_spin)
        size.addWidget(QtWidgets.QLabel('x'))
        size.addWidget(self.height_spin)
        size.addStretch(1)
        form.addRow('Resolution', size)

        checks = QtWidgets.QHBoxLayout()
        self.gpu_check = QtWidgets.QCheckBox('Use the GPU')
        self.gpu_check.setChecked(True)
        self.transparent_check = QtWidgets.QCheckBox('Transparent background')
        checks.addWidget(self.gpu_check)
        checks.addWidget(self.transparent_check)
        checks.addStretch(1)
        form.addRow('', checks)

        row = QtWidgets.QHBoxLayout()
        self.out_edit = QtWidgets.QLineEdit(suggested)
        out_browse = QtWidgets.QPushButton('Browse...')
        out_browse.clicked.connect(self._browse_output)
        row.addWidget(self.out_edit, 1)
        row.addWidget(out_browse)
        form.addRow('Save to', row)

        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.render_button = QtWidgets.QPushButton('Render')
        self.render_button.setDefault(True)
        self.render_button.clicked.connect(self.on_render)
        self.open_button = QtWidgets.QPushButton('Open in Blender')
        self.open_button.setToolTip(
            'Build the scene in Blender\'s own window instead of rendering, '
            'so it can be looked at and changed by hand.')
        self.open_button.clicked.connect(self.on_open_in_blender)
        self.save_button = QtWidgets.QPushButton('Save scene only')
        self.save_button.setToolTip(
            'Write the .npz and show the command, without running anything.')
        self.save_button.clicked.connect(self.on_save_scene)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.reject)
        for b in (self.render_button, self.open_button, self.save_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText('Blender\'s output appears here.')
        layout.addWidget(self.log, 1)

        if not self.exe_edit.text():
            self._say('No Blender found. Point the box above at blender.exe, '
                      'or use "Save scene only" and render it later.')

    def _browse_exe(self):
        pattern = ('blender.exe' if sys.platform.startswith('win')
                   else 'blender')
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Where is Blender?', '', '%s (%s);;All files (*)'
            % ('Blender', pattern))
        if path:
            self.exe_edit.setText(path)

    def _browse_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save the render as', self.out_edit.text(),
            'PNG image (*.png)')
        if path:
            self.out_edit.setText(path)

    # ---- doing it ----

    def _params(self):
        return {
            'metallic': self.metallic,
            'roughness': self.roughness,
            'engine': ('CYCLES' if self.engine_combo.currentIndex() == 0
                       else 'EEVEE'),
            'samples': self.samples_spin.value(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'gpu': self.gpu_check.isChecked(),
            'transparent': self.transparent_check.isChecked(),
        }

    def _write_scene(self):
        output = self.out_edit.text().strip() or 'render.png'
        folder = os.path.dirname(os.path.abspath(output))
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        scene = os.path.splitext(os.path.abspath(output))[0] + '_scene.npz'
        write_scene(self.mesh, scene, self.cmap, self.clim, self._params())
        self.scene_path = scene
        self._say('Wrote %s (%.1f MB)'
                  % (scene, os.path.getsize(scene) / 1e6))
        return scene, os.path.abspath(output)

    def on_save_scene(self):
        try:
            scene, output = self._write_scene()
        except Exception as exc:
            self._say('Could not write the scene: %s: %s'
                      % (type(exc).__name__, exc))
            return
        argv = command_for(self.exe_edit.text().strip() or None, scene, output)
        self._say('Render it with:\n  ' + ' '.join(
            ('"%s"' % a if ' ' in a else a) for a in argv))

    def on_render(self):
        self._launch(background=True)

    def on_open_in_blender(self):
        self._launch(background=False)

    def _launch(self, background):
        if self.process is not None:
            self._say('Already running.')
            return
        blender = self.exe_edit.text().strip()
        if not blender or not os.path.isfile(blender):
            self._say('Set the path to Blender first.')
            return
        try:
            scene, output = self._write_scene()
        except Exception as exc:
            self._say('Could not write the scene: %s: %s'
                      % (type(exc).__name__, exc))
            return

        argv = command_for(blender, scene, output, background=background)
        self._say('\n$ ' + ' '.join(('"%s"' % a if ' ' in a else a)
                                    for a in argv) + '\n')

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(
            lambda err: self._say('Could not start Blender (%s)' % err))
        self.render_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.process.start(argv[0], argv[1:])

    def _read(self):
        data = bytes(self.process.readAllStandardOutput())
        self._say(data.decode('utf-8', 'replace').rstrip(), raw=True)

    def _finished(self, code, _status):
        self._say('\nBlender exited with code %d' % code)
        output = self.out_edit.text().strip()
        if code == 0 and output and os.path.isfile(output):
            self._say('Saved %s' % os.path.abspath(output))
        self.process = None
        self.render_button.setEnabled(True)
        self.open_button.setEnabled(True)

    def _say(self, text, raw=False):
        self.log.appendPlainText(text if raw else text)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def closeEvent(self, event):
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(2000)
        super(BlenderDialog, self).closeEvent(event)
