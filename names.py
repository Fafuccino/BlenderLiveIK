"""Keeps the chains pointing at the right bones when bones are renamed.

A chain stores its bones by name, and Blender does not update a plain text field when a
bone is renamed. Renaming a bone does not reorder the armature's bones, so comparing each
armature's bone names, in order, with the ones remembered last time tells exactly which
bone got which new name. Every reference to it is then updated.
"""

import bpy
from bpy.app.handlers import persistent

from . import properties

# The chain's settings that hold a bone name.
BONE_PROPS = ("bone_0", "bone_1", "bone_2", "bone_3", "ik_parent", "pole_anchor")

_known = {}  # armature data name -> its bone names, in order


def _names(armature):
    return [bone.name for bone in armature.bones]


def remember(armature):
    _known[armature.name] = _names(armature)


def is_known(armature):
    return armature.name in _known


def reconcile(armature):
    """Follow the renames since the last look. Returns how many references were updated."""
    old = _known.get(armature.name)
    new = _names(armature)
    _known[armature.name] = new
    if old is None or len(old) != len(new):
        return 0  # bones were added or removed: positions no longer say who is who
    renames = {before: after for before, after in zip(old, new) if before != after}
    if not renames:
        return 0
    updated = 0
    properties.quiet = True  # the bone is the same one: nothing to set up again
    try:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE" and obj.data == armature:
                for chain in obj.live_ik.chains:
                    for prop in BONE_PROPS:
                        value = getattr(chain, prop)
                        if value in renames:
                            setattr(chain, prop, renames[value])
                            updated += 1
                for parent_chain in obj.live_ik.parent_chains:
                    for item in parent_chain.bones:
                        if item.bone in renames:
                            item.bone = renames[item.bone]
                            updated += 1
    finally:
        properties.quiet = False
    return updated


@persistent
def _on_load_post(*_):
    _known.clear()  # another file: the names remembered belong to other armatures


def register():
    bpy.app.handlers.load_post.append(_on_load_post)
    try:
        for armature in bpy.data.armatures:
            remember(armature)
    except (AttributeError, RuntimeError):
        pass  # at startup the file is not loaded yet: the first update remembers them


def unregister():
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    _known.clear()
