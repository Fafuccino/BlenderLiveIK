"""Keyframes for the bones an IK chain poses, by hand or with Blender's auto keying.

The IK moves every bone of the chain, so every bone of the chain is keyed together, and
the mirrored chain too when pose symmetry is on. Auto keying waits for the edit to be
validated: a gizmo drag or transform is finished, or a panel value was typed.
"""

import bpy

_pending = set()  # (object name, chain index) edited since the last key
_scheduled = False

_ROTATION_PATH = {
    "QUATERNION": "rotation_quaternion",
    "AXIS_ANGLE": "rotation_axis_angle",
}


def autokey_enabled():
    scene = bpy.context.scene
    return bool(scene and scene.tool_settings.use_keyframe_insert_auto)


def key_bone(obj, name, frame):
    pose_bone = obj.pose.bones.get(name) if name and obj.pose else None
    if not pose_bone:
        return
    group = pose_bone.name
    pose_bone.keyframe_insert(_ROTATION_PATH.get(pose_bone.rotation_mode, "rotation_euler"), frame=frame, group=group)
    # A connected bone has no use for its location unless something moved it anyway.
    if not pose_bone.bone.use_connect or any(abs(v) > 1e-6 for v in pose_bone.location):
        pose_bone.keyframe_insert("location", frame=frame, group=group)
    if any(abs(v - 1.0) > 1e-6 for v in pose_bone.scale):
        pose_bone.keyframe_insert("scale", frame=frame, group=group)


def key_chain(obj, chain, frame=None):
    """Key the bones of a chain, and of its mirror when pose symmetry is on."""
    from . import heel

    if frame is None:
        frame = bpy.context.scene.frame_current
    chains = [chain]
    partner = heel.symmetry_partner(obj, chain)
    if partner:
        chains.append(partner)
    for each in chains:
        for name in (each.bone_0, each.bone_1, each.bone_2, each.bone_3):
            key_bone(obj, name, frame)


def flush_pending():
    """Key what was edited since the last time."""
    pending = list(_pending)
    _pending.clear()
    for obj_name, index in pending:
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.type == "ARMATURE" and index < len(obj.live_ik.chains):
            key_chain(obj, obj.live_ik.chains[index])


def _busy():
    """A modal operator (a gizmo drag, a transform) is still running."""
    windows = bpy.context.window_manager.windows
    return any(len(getattr(window, "modal_operators", ())) for window in windows)


def _tick():
    global _scheduled
    if _busy():
        return 0.1
    _scheduled = False
    flush_pending()
    return None


def request(obj, chain):
    """Note that a chain was edited; with auto keying on, it is keyed once the edit is done."""
    global _scheduled
    if not autokey_enabled():
        return
    index = next((i for i, other in enumerate(obj.live_ik.chains) if other == chain), None)
    if index is None:
        return
    _pending.add((obj.name, index))
    if not _scheduled:
        _scheduled = True
        bpy.app.timers.register(_tick, first_interval=0.1)
