"""Saves the panel's state in the .blend file.

The mode row, the orientation, Show Heels and which chains are unfolded are interface
state, kept on the window manager so that changing them is not an undo step. Blender does
not save those, so they are copied into the scene when the file is saved, and put back
when it is opened.
"""

import json

import bpy
from bpy.app.handlers import persistent

from . import properties, ui

KEY = "live_ik_ui"
_SETTINGS = ("live_ik_gizmo_mode", "live_ik_orientation", "live_ik_show_ends", "live_ik_show_heels")


def _collect(wm):
    data = {name: getattr(wm, name) for name in _SETTINGS}
    data["expanded"] = sorted([name, index] for name, index in ui._expanded)
    return data


def save():
    wm = bpy.context.window_manager
    text = json.dumps(_collect(wm))
    for scene in bpy.data.scenes:
        scene[KEY] = text


def load():
    """Put the saved state back. A file without any keeps the current panel as it is."""
    text = next((scene[KEY] for scene in bpy.data.scenes if KEY in scene), None)
    if text is None:
        return
    try:
        data = json.loads(text)
    except ValueError:
        return
    wm = bpy.context.window_manager
    properties.quiet = True  # putting a mode back must not run what changing it does
    try:
        for name in _SETTINGS:
            if name in data:
                try:
                    setattr(wm, name, data[name])
                except (TypeError, ValueError):
                    pass  # a mode this version does not have
    finally:
        properties.quiet = False
    ui._expanded.clear()
    ui._expanded.update((name, index) for name, index in data.get("expanded", []))


@persistent
def _on_save_pre(*_):
    save()


@persistent
def _on_load_post(*_):
    load()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def register():
    bpy.app.handlers.save_pre.append(_on_save_pre)
    bpy.app.handlers.load_post.append(_on_load_post)
    try:
        load()  # the add-on was switched on with a file already open
    except (AttributeError, RuntimeError):
        pass  # at startup Blender has not loaded the file yet: load_post will


def unregister():
    for handlers, function in ((bpy.app.handlers.save_pre, _on_save_pre), (bpy.app.handlers.load_post, _on_load_post)):
        if function in handlers:
            handlers.remove(function)
