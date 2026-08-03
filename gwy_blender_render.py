"""
Build the AFM surface as a Blender scene and render it. Runs inside Blender.

This file is never imported by the viewer. Blender ships its own Python
interpreter with `bpy` - the whole of Blender - already in it, and the way
to drive it from outside is to hand that interpreter a script:

    blender --background --python gwy_blender_render.py -- scene.npz out.png

Everything after the bare ``--`` is passed through to the script instead of
being read by Blender. `gwy_blender_export` writes `scene.npz` and runs that
command; nothing is shared between the two processes except the file, which
is why this script imports numpy and `bpy` and nothing else. Blender bundles
numpy, so there is nothing to install.

Drop the ``--background`` and the same command opens Blender's own window
with the surface already built, lit and framed. That is the useful way to
learn what the script is doing: everything it set is there in the interface
to be looked at and changed by hand.

What the scene contains:

  * the height map as a mesh, one vertex per pixel, in the same frame units
    the viewer uses, with the Z exaggeration already applied;
  * the false colour as a per-vertex colour attribute, converted from sRGB
    to linear because Blender shades in linear light and handing it sRGB
    values directly washes the gradient out;
  * a Principled BSDF reading that attribute as its base colour;
  * three area lights and a camera framed on the surface.

The render engine is Cycles by default - a path tracer, so it computes real
shadows, real ambient occlusion and real interreflection rather than the
screen-space approximations the live view uses. That is the whole point of
coming here instead of taking a screenshot. EEVEE is offered as well and is
much faster if the still is only meant to be looked at.
"""

import json
import os
import sys

import numpy as np

import bpy
import mathutils


# ------------------------------------------------------------------ colour

def srgb_to_linear(rgb8):
    """Blender shades in linear light; the colour map hands out sRGB."""
    c = np.asarray(rgb8, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# -------------------------------------------------------------------- mesh

def build_mesh(relief, dx, dy, rgb, name='AFM surface'):
    """A grid mesh: one vertex per pixel, one quad per group of four."""
    ny, nx = relief.shape

    xs = np.arange(nx, dtype=np.float64) * dx
    ys = np.arange(ny, dtype=np.float64) * dy
    gx, gy = np.meshgrid(xs, ys)
    verts = np.column_stack([gx.ravel(), gy.ravel(), relief.ravel()])

    # Corner indices of every quad, in the winding Blender expects.
    idx = np.arange(ny * nx).reshape(ny, nx)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]],
                     axis=-1).reshape(-1, 4)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()

    # One colour per vertex. POINT domain keeps the array the same length as
    # the vertices, so it can be written straight from the height map.
    attr = mesh.color_attributes.new(name='height', type='FLOAT_COLOR',
                                     domain='POINT')
    linear = srgb_to_linear(rgb.reshape(-1, 3))
    rgba = np.ones((linear.shape[0], 4), dtype=np.float64)
    rgba[:, :3] = linear
    attr.data.foreach_set('color', rgba.ravel())

    for polygon in mesh.polygons:
        polygon.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_material(obj, metallic, roughness):
    material = bpy.data.materials.new('AFM surface')
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes.get('Principled BSDF')
    if bsdf is None:                     # a template without one
        bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
        tree.links.new(bsdf.outputs['BSDF'],
                       tree.nodes['Material Output'].inputs['Surface'])

    colour = tree.nodes.new('ShaderNodeVertexColor')
    colour.layer_name = 'height'
    colour.location = (-320, 240)
    tree.links.new(colour.outputs['Color'], bsdf.inputs['Base Color'])

    for key, value in (('Metallic', metallic), ('Roughness', roughness)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = float(value)

    obj.data.materials.append(material)
    return material


# ------------------------------------------------------------------- scene

def clear_scene():
    """Blender starts with a cube, a light and a camera. None of them help."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials,
                       bpy.data.lights, bpy.data.cameras):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def build_world(colour, strength):
    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    background = world.node_tree.nodes['Background']
    background.inputs[0].default_value = (colour[0], colour[1], colour[2], 1.0)
    background.inputs[1].default_value = float(strength)
    bpy.context.scene.world = world
    return world


def build_lights(centre, size):
    """A three-point rig, scaled to the object rather than to Blender's units.

    Area lights and not points: a point light on a rough surface gives a hard
    specular sparkle on every bump, which reads as noise. A large soft source
    is what makes the shape legible.
    """
    rig = [
        ('key',  (1.8, -1.6, 1.9), 1.00, 1.4),
        ('fill', (-2.0, -0.9, 1.0), 0.35, 2.0),
        ('rim',  (0.1, 2.2, 1.4), 0.45, 1.6),
    ]
    for name, offset, power, span in rig:
        data = bpy.data.lights.new(name, type='AREA')
        data.shape = 'SQUARE'
        data.size = span * size
        # Power has to grow with the square of the distance to keep the same
        # exposure whatever the scan size.
        data.energy = power * 900.0 * (size ** 2)
        light = bpy.data.objects.new(name, data)
        position = mathutils.Vector(centre) + mathutils.Vector(offset) * size
        light.location = position
        direction = mathutils.Vector(centre) - position
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        bpy.context.collection.objects.link(light)


def build_camera(centre, size, direction=(1.4, -1.8, 1.15), lens=60.0):
    data = bpy.data.cameras.new('Camera')
    data.lens = lens
    camera = bpy.data.objects.new('Camera', data)
    offset = mathutils.Vector(direction).normalized() * (size * 2.6)
    camera.location = mathutils.Vector(centre) + offset
    look = mathutils.Vector(centre) - camera.location
    camera.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


# ------------------------------------------------------------------ render

def _eevee_name():
    """EEVEE was renamed in Blender 4.2; ask the build which one it has."""
    items = bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
    names = [item.identifier for item in items]
    for candidate in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        if candidate in names:
            return candidate
    return names[0]


def enable_gpu():
    """Point Cycles at whatever accelerator this machine has.

    Returns the backend that took, or None - falling back to the CPU is slow
    but correct, and worth saying out loud rather than failing.
    """
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except KeyError:
        return None
    for backend in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == backend]
        if not devices:
            continue
        for device in prefs.devices:
            device.use = (device.type == backend)
        bpy.context.scene.cycles.device = 'GPU'
        return backend
    return None


def configure_render(params):
    scene = bpy.context.scene
    engine = params.get('engine', 'CYCLES')
    scene.render.engine = 'CYCLES' if engine == 'CYCLES' else _eevee_name()
    scene.render.resolution_x = int(params.get('width', 1920))
    scene.render.resolution_y = int(params.get('height', 1280))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = bool(params.get('transparent', False))

    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = int(params.get('samples', 128))
        scene.cycles.use_denoising = True
        if params.get('gpu', True):
            backend = enable_gpu()
            print('[gwy] Cycles device: %s' % (backend or 'CPU'))
    else:
        if hasattr(scene, 'eevee'):
            scene.eevee.taa_render_samples = int(params.get('samples', 128))


# -------------------------------------------------------------------- main

def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    if not argv:
        print('[gwy] usage: blender --background --python %s -- scene.npz '
              '[out.png]' % os.path.basename(__file__))
        return 1

    scene_path = argv[0]
    output = argv[1] if len(argv) > 1 else None

    data = np.load(scene_path, allow_pickle=False)
    params = json.loads(str(data['params']))
    relief = data['relief'].astype(np.float64)
    rgb = data['rgb']
    dx, dy = float(data['dx']), float(data['dy'])

    print('[gwy] %d x %d pixels, %d faces'
          % (relief.shape[1], relief.shape[0],
             (relief.shape[0] - 1) * (relief.shape[1] - 1)))

    clear_scene()
    obj = build_mesh(relief, dx, dy, rgb, name=params.get('name', 'AFM'))
    build_material(obj, params.get('metallic', 0.1),
                   params.get('roughness', 0.5))

    width = dx * (relief.shape[1] - 1)
    depth = dy * (relief.shape[0] - 1)
    centre = (width / 2.0, depth / 2.0, float(np.mean(relief)))
    size = max(width, depth)

    build_world(params.get('world_color', (0.05, 0.06, 0.07)),
                params.get('world_strength', 0.6))
    build_lights(centre, size)
    build_camera(centre, size)
    configure_render(params)

    if not bpy.app.background:
        # Opened with the interface: leave the scene sitting there to be
        # looked at rather than rendering over the top of it.
        print('[gwy] scene built - press F12 to render')
        return 0

    if output:
        bpy.context.scene.render.filepath = os.path.abspath(output)
        print('[gwy] rendering to %s' % bpy.context.scene.render.filepath)
        bpy.ops.render.render(write_still=True)
        print('[gwy] done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
