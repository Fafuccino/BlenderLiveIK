"""IK mode: re-solve the legs after the pose changed.

The pose cannot be edited while the depsgraph is evaluating, so the update handler only
notes which armatures changed and a timer does the solving right after.
"""

import bpy
from bpy.app.handlers import persistent

from . import heel, names

_pending = set()
_renamed = set()  # armature data that changed: bones may have been renamed
_scheduled = False
_last_frame = None
_frame_changed = set()  # armatures touched by a pass in which the frame changed


def _solve_pending():
    global _scheduled
    _scheduled = False
    for armature_name in list(_renamed):
        armature = bpy.data.armatures.get(armature_name)
        if armature:
            names.reconcile(armature)
    _renamed.clear()
    object_names = list(_pending)
    _pending.clear()
    playing = bool(bpy.context.screen and bpy.context.screen.is_animation_playing)
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "ARMATURE":
            # A pose that changed with the frame is the animation playing, not an edit to key.
            user_edit = not playing and name not in _frame_changed
            if name in _frame_changed:
                # The pose now comes from the animation, with the heel rotation and the
                # toes' counter-turn already in it. What was remembered about the last
                # edit no longer describes it, and would be undone from the wrong pose.
                for chain in obj.live_ik.chains:
                    heel.clear(chain)
            for chain in obj.live_ik.chains:
                if chain.ik_enabled:
                    heel.sync_ik(obj, chain, user_edit)
    _frame_changed.clear()
    return None


@persistent
def on_depsgraph_update(scene, depsgraph):
    global _scheduled, _last_frame
    frame_changed = _last_frame is not None and scene.frame_current != _last_frame
    _last_frame = scene.frame_current
    for update in depsgraph.updates:
        original = update.id.original
        if isinstance(original, bpy.types.Armature):
            _renamed.add(original.name)  # looked at by the timer, whatever the mode
        elif isinstance(original, bpy.types.Object) and original.type == "ARMATURE" and not names.is_known(original.data):
            names.remember(original.data)
    if not heel.solving_enabled():
        if _renamed and not _scheduled:
            _scheduled = True
            bpy.app.timers.register(_solve_pending, first_interval=0.0)
        return
    for update in depsgraph.updates:
        obj = update.id.original
        if isinstance(obj, bpy.types.Object) and obj.type == "ARMATURE" and len(obj.live_ik.chains):
            if frame_changed:
                _pending.add(obj.name)
                _frame_changed.add(obj.name)
            elif any(chain.ik_enabled for chain in obj.live_ik.chains):
                _pending.add(obj.name)
    # Our own solving updates the depsgraph again; that pass finds nothing moved and stops.
    if (_pending or _renamed) and not _scheduled:
        _scheduled = True
        bpy.app.timers.register(_solve_pending, first_interval=0.0)


def register():
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)


def unregister():
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    _pending.clear()
    _renamed.clear()
