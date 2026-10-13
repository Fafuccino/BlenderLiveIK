bl_info = {
    "name": "LiveIK",
    "author": "fafuccino",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > LiveIK",
    "description": "Pose with IK behavior without setting up IK/FK rigs",
    "category": "Animation",
}

if "bpy" in locals():
    import importlib

    importlib.reload(properties)
    importlib.reload(keying)
    importlib.reload(heel)
    importlib.reload(handlers)
    importlib.reload(state)
    importlib.reload(ui)
    importlib.reload(gizmos)
else:
    from . import gizmos, handlers, heel, keying, properties, state, ui

import bpy


def register():
    properties.register()
    ui.register()
    gizmos.register()
    handlers.register()
    state.register()


def unregister():
    state.unregister()
    handlers.unregister()
    gizmos.unregister()
    ui.unregister()
    properties.unregister()
