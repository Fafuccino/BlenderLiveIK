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
    importlib.reload(names)
    importlib.reload(grab)
    importlib.reload(parent_chain)
    importlib.reload(ui)
    importlib.reload(gizmos)
else:
    from . import gizmos, grab, handlers, heel, keying, names, parent_chain, properties, state, ui

import bpy


def register():
    properties.register()
    ui.register()
    gizmos.register()
    handlers.register()
    state.register()
    names.register()
    grab.register()
    parent_chain.register()


def unregister():
    parent_chain.unregister()
    grab.unregister()
    names.unregister()
    state.unregister()
    handlers.unregister()
    gizmos.unregister()
    ui.unregister()
    properties.unregister()
