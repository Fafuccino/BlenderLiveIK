"""IK for legs and arms, and the heel controller.

The end of a chain (foot or hand) is a target matrix; a two-bone IK poses the other
bones to reach it, bending toward the pole.

Heel controller: rotates the foot around a pivot and poses the leg to follow.

The pivot is a point anchored to the foot bone. The controller applies a rotation
(pitch, roll, yaw) around it to the foot, then solves the leg with a two-bone IK
(thigh, calf, foot) toward the pole. Everything is in armature space.
"""

from math import sqrt

from contextlib import contextmanager

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import keying
from .properties import (
    _unscaled,
    bone_pose_matrix,
    mirror_partner,
    pose_anchor,
    rest_anchor,
    reparent_pole,
    valid_ik_parent,
)

IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
IDENTITY_MATRIX = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

# Set while the heel values are reset from code, so the update callbacks do not pose anything.
suspended = False


def solving_enabled():
    """False when the panel's gizmo mode is Off: no IK solving at all, whatever the chains say."""
    return bpy.context.window_manager.live_ik_gizmo_mode != "OFF"


def is_rotated(chain):
    return any(abs(angle) > 1e-5 for angle in chain.heel_done)


def clear(chain):
    """Forget any applied heel rotation without posing (bones or type changed)."""
    global suspended
    suspended = True
    try:
        chain.heel_pitch = chain.heel_roll = chain.heel_yaw = 0.0
        chain.heel_done = (0.0, 0.0, 0.0)
        chain.end_goal = chain.end_left = chain.end_basis_left = IDENTITY_MATRIX
        chain.end_thigh_left = chain.end_calf_left = chain.end_root_left = IDENTITY_MATRIX
        chain.end_parent_left = IDENTITY_MATRIX
    finally:
        suspended = False


def _pose_bones(obj, chain):
    pose_bones = obj.pose.bones if obj.pose else {}
    names = (chain.bone_0, chain.bone_1, chain.bone_2, chain.bone_3)
    return [pose_bones.get(n) if n else None for n in names]


def _about(pivot, rotation):
    return Matrix.Translation(pivot) @ rotation.to_matrix().to_4x4() @ Matrix.Translation(-pivot)


def ground_frame(directions):
    """Frame (columns X right, Y forward, Z up) from the first usable direction.

    Z is the armature's up axis (orthogonal to its ground plane). Y is the direction
    projected on the ground plane, so it points away from the foot. A direction that
    points (almost) straight down has no usable ground direction and is skipped.
    """
    up = Vector((0, 0, 1))
    forward = Vector((0, -1, 0))  # armature front
    for direction in directions:
        flat = Vector((direction.x, direction.y, 0))
        if flat.length > 1e-6 and flat.length > direction.length * 0.1:
            forward = flat.normalized()
            break
    right = forward.cross(up)
    return Matrix((right, forward, up)).transposed()


def _flat(matrix):
    return [value for row in matrix for value in row]


def _mat(values):
    return Matrix([values[0:4], values[4:8], values[8:12], values[12:16]])


def _close(a, b):
    return all(abs(x - y) <= 1e-4 * max(1.0, abs(x)) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def ik_parent_matrix(obj, chain):
    """Armature-space matrix of the IK parent (no scale), identity when there is none.

    The IK parent is the bone the IK foot is parented to. A bone that is part of the leg
    (or below it) cannot be one, since the foot would be chasing itself.
    """
    return bone_pose_matrix(obj, valid_ik_parent(obj, chain))


def reparent(obj, chain):
    """The IK parent changed: keep the foot target and the pole where they are, now relative to it."""
    old_goal = _mat(chain.end_parent_left) @ _mat(chain.end_goal)  # armature space
    new_parent = ik_parent_matrix(obj, chain)
    chain.end_goal = _flat(new_parent.inverted() @ old_goal)
    chain.end_parent_left = _flat(new_parent)
    # Last: moving the pole solves the leg, which needs the foot target already reparented.
    reparent_pole(obj, chain)


def _root_matrix(thigh):
    """The matrix of the bone the chain hangs from (the pelvis, the chest...)."""
    return thigh.parent.matrix.copy() if thigh.parent else Matrix.Identity(4)


def limb_untouched(obj, chain):
    """True while the chain is as the solver left it.

    The solver writes the pose of the three bones, and what they hang from decides where
    they end up. So "untouched" is whether those four are what the solver left, not whether
    the foot's evaluated matrix matches: a constraint, or anything else that changes a bone
    after its pose is set, makes the two differ all the time, and the chain would then look
    moved on every update.
    """
    thigh, calf, foot, _ = _pose_bones(obj, chain)
    if not (thigh and calf and foot):
        return False
    return (
        _close(foot.matrix_basis, _mat(chain.end_basis_left))
        and _close(calf.matrix_basis, _mat(chain.end_calf_left))
        and _close(thigh.matrix_basis, _mat(chain.end_thigh_left))
        and _close(_root_matrix(thigh), _mat(chain.end_root_left))
    )


def end_target(obj, chain):
    """The matrix the IK is asked to bring the foot to (armature space, no scale).

    The solver remembers the target it was given. While the chain is as it left it, that
    target is returned: the leg may not reach it, and the foot's evaluated matrix may not be
    exactly what was asked (constraints...), so starting from it would add the difference
    to every step. If the chain was changed (posed by hand, the body moved in FK, animated),
    the foot's actual matrix is the target.
    """
    _, _, foot, _ = _pose_bones(obj, chain)
    if limb_untouched(obj, chain):
        return ik_parent_matrix(obj, chain) @ _mat(chain.end_goal)
    return _unscaled(foot.matrix)


def move_end(obj, chain, target):
    """Bring the foot to a new target matrix; the heel rotation stays applied on top of it."""
    result = solve_limb(obj, chain, target)
    if result:
        mirror_chain(obj, chain)
        keying.request(obj, chain)
    return result


def resolve(obj, chain):
    """Solve the chain again toward the end it was last given, after the pole moved.

    A chain that was never solved (still posed by hand, IK off) is left alone.
    """
    if suspended:
        return
    if not chain.ik_enabled and _close(_mat(chain.end_left), Matrix.Identity(4)):
        return
    bpy.context.view_layer.update()  # make the pose bone matrices current
    _, _, foot, _ = _pose_bones(obj, chain)
    if foot and solve_limb(obj, chain, end_target(obj, chain)):
        keying.request(obj, chain)


def sync_ik(obj, chain, user_edit=True):
    """IK mode: keep the chain solved after its end (foot, hand) or something above it was moved.

    user_edit is False when the pose changed because the frame did (playback, scrubbing):
    the chain is solved, but not keyed.
    """
    if suspended:
        return
    bpy.context.view_layer.update()  # make the pose bone matrices current
    _, _, foot, _ = _pose_bones(obj, chain)
    if not foot:
        return
    current = _unscaled(foot.matrix)
    parent = ik_parent_matrix(obj, chain)
    adopted = False
    if _close(_mat(chain.end_left), Matrix.Identity(4)):
        adopted = True
        target = current  # never solved: IK takes over from the pose as it is
    elif not _close(foot.matrix_basis, _mat(chain.end_basis_left)):
        target = current  # the foot itself was posed: it is the new target
    elif limb_untouched(obj, chain) and _close(parent, _mat(chain.end_parent_left)):
        return  # nothing moved
    else:
        # Something above the foot moved, or its IK parent did: the foot goes where its
        # parent takes it (planted, when it has none).
        target = parent @ _mat(chain.end_goal)
    foot_matrix = solve_limb(obj, chain, target)
    if foot_matrix:
        mirror_chain(obj, chain)
        if user_edit and not adopted:
            keying.request(obj, chain)


def _forward_frame(foot_matrix, foot, toes):
    """Ground frame of a foot, with forward pointing toward the toes.

    Forward has to answer two things at once. It must point toward the toes, whichever way
    the foot bone is drawn (it can lean back): or the pitch, and with it the counter-turn
    of the toes, comes out reversed. And it must not change when the foot is banked or
    pitched, only when it is swiveled: the heel is edited in small steps measured in this
    frame, and a frame that drifts makes them add up to the wrong pose.

    The toes' own direction (head to tail) is the one that does both: banking turns the
    toes about their own axis, and the counter-turn only tips them, neither changes where
    they point on the ground. The direction from the foot's head to the toes' head is the
    fallback (banking swings it sideways when the foot is held high above the toes), then
    the foot bone's own direction, which is the only choice without toes.
    """
    directions = []
    if toes and toes.parent == foot:
        directions.append(toes.tail - toes.head)
        directions.append((foot_matrix @ _rest_rel(toes.bone)).translation - foot_matrix.translation)
    directions.append(foot_matrix.col[1].to_3d())
    return ground_frame(directions)


def _pivot_ground_frame(obj, chain):
    """The foot's ground frame, taken from the foot target.

    Unlike heel_frame's, this one does not ignore the heel rotation: it is fixed while the
    pivot is being edited.
    """
    _, _, foot, toes = _pose_bones(obj, chain)
    return _forward_frame(end_target(obj, chain), foot, toes)


def handle_axes(obj, chain, kind):
    """Axes a handle is shown and moved along, in armature space (columns of a 3x3).

    World (the default): the scene's axes, for every handle. Rig: the armature object's own
    axes. Local, per handle: the pole follows the axes of the bone it is attached to, the
    heel pivot the foot's ground frame (Z up, Y away from the foot), and the end (foot,
    hand) the armature's axes turned by the rotation it has from its rest pose, so it
    starts aligned with the armature and follows the foot when it rotates.
    """
    orientation = bpy.context.window_manager.live_ik_orientation
    if orientation == "WORLD":
        return obj.matrix_world.to_quaternion().to_matrix().inverted()  # the scene's axes, seen from the armature
    if orientation == "RIG":
        return Matrix.Identity(3)
    if kind == "heel_pos":
        return _pivot_ground_frame(obj, chain)
    if kind == "end":
        bone = obj.data.bones.get(chain.bone_2)
        if not bone:
            return Matrix.Identity(3)
        return end_target(obj, chain).to_3x3() @ bone.matrix_local.to_3x3().inverted()
    return pose_anchor(obj, chain, kind).to_3x3()


def pivot_offset(obj, chain):
    """The pivot relative to the foot's head, along the handle's axes."""
    _, _, foot, _ = _pose_bones(obj, chain)
    if not foot:
        return Vector(chain.heel_pos)
    target = end_target(obj, chain)
    return handle_axes(obj, chain, "heel_pos").inverted() @ (target @ Vector(chain.heel_pos) - target.translation)


def set_pivot_offset(obj, chain, offset):
    _, _, foot, _ = _pose_bones(obj, chain)
    if not foot:
        chain.heel_pos = offset
        return
    target = end_target(obj, chain)
    chain.heel_pos = target.inverted() @ (target.translation + handle_axes(obj, chain, "heel_pos") @ offset)


def pole_offset(obj, chain):
    """The pole relative to the head of the bone it follows, along the handle's axes."""
    anchor = pose_anchor(obj, chain, "pole_pos")
    return handle_axes(obj, chain, "pole_pos").inverted() @ (anchor @ Vector(chain.pole_pos) - anchor.translation)


def set_pole_offset(obj, chain, offset):
    anchor = pose_anchor(obj, chain, "pole_pos")
    chain.pole_pos = anchor.inverted() @ (anchor.translation + handle_axes(obj, chain, "pole_pos") @ offset)


def heel_frame(obj, chain):
    """(pivot, frame, target) in armature space, or None.

    The pivot is a point of the foot, so it is where the foot target puts it. The frame is
    the foot's ground frame as it is now (forward toward the toes, up perpendicular to the
    armature's ground plane): the pitch / roll / yaw axes follow the foot's direction.
    """
    _, _, foot, toes = _pose_bones(obj, chain)
    if not foot:
        return None
    target = end_target(obj, chain)
    return target @ Vector(chain.heel_pos), _forward_frame(target, foot, toes), target


def _heel_rotation(frame, angles):
    """Rotation (3x3, armature space) of the heel angles (pitch, roll, yaw) in a ground frame."""
    pitch, roll, yaw = angles
    local = Matrix.Rotation(yaw, 3, "Z") @ Matrix.Rotation(roll, 3, "Y") @ Matrix.Rotation(pitch, 3, "X")
    return frame @ local @ frame.inverted()


def _toes_stand(foot, toes, target):
    """The point of the foot that stands on the ground: the toes' head, else the foot's tip."""
    if toes and toes.parent == foot:
        return (target @ _rest_rel(toes.bone)).translation
    return target @ Vector((0, foot.bone.length, 0))


def _toes_follow(frame, old, new):
    """The part of the foot's turn that the toes follow: the toes-up part of the pitch, the
    bank and the swivel. What is left of the turn is the raised-heel part of the pitch."""

    def follows(angles):
        return (max(angles[0], 0.0), angles[1], angles[2])

    return _heel_rotation(frame, follows(new)) @ _heel_rotation(frame, follows(old)).inverted()


def _counter_toes(obj, chain, foot_matrix, follow, turn):
    """Keep the toes flat while the heel is raised.

    The foot turned by `turn` and the toes only follow `follow` of it: they turn back about
    their own head by the rest, so they do not dig into the ground. This is done on top of
    the toes' pose as it is now, so whatever was done to the foot or the toes since the last
    heel edit is kept.
    """
    _, _, foot, toes = _pose_bones(obj, chain)
    if not toes or toes.parent != foot:
        return
    counter = follow @ turn.inverted()
    # The turn is about the toes head, so in the toes' own rest frame it is a plain rotation.
    parent_rest = (foot_matrix @ _rest_rel(toes.bone)).to_3x3()
    local = parent_rest.inverted() @ counter @ parent_rest
    toes.matrix_basis = local.to_4x4() @ toes.matrix_basis


def apply_heel(obj, chain):
    """Turn the foot about the pivot by the change of the chain's pitch / roll / yaw, and pose the leg.

    It works from the foot as it is now and only by what changed since the last time
    (heel_done). Nothing about the foot before the heel was used has to be remembered, so
    the foot can be moved or turned by hand in between and the heel carries on from there
    without a jump.

    Pitch, roll and yaw turn the foot about the pivot. Raising the heel (a negative pitch)
    also moves the foot so the toes stay where they are, height and place: that turn carries
    them along an arc, down when the pivot is behind them, forward as well, up when it is
    beyond them, and the toes are what stands on the ground. Raising the toes does not.
    """
    if suspended or chain.chain_type != "LEG" or not solving_enabled():
        return
    old = tuple(chain.heel_done)
    new = (chain.heel_pitch, chain.heel_roll, chain.heel_yaw)
    if max(abs(a - b) for a, b in zip(old, new)) < 1e-7:
        return
    bpy.context.view_layer.update()  # make the pose bone matrices current
    state = heel_frame(obj, chain)
    if not state:
        return
    pivot, frame, target = state
    _, _, foot, toes = _pose_bones(obj, chain)

    turn = _heel_rotation(frame, new) @ _heel_rotation(frame, old).inverted()
    follow = _toes_follow(frame, old, new)
    # The toes are meant to go only where `follow` takes them. The rest of the turn (the
    # raised-heel part of the pitch) would carry them elsewhere: the foot is moved back by
    # the difference.
    stand = _toes_stand(foot, toes, target)
    shift = _about(pivot, follow.to_quaternion()) @ stand - _about(pivot, turn.to_quaternion()) @ stand

    goal = Matrix.Translation(shift) @ _about(pivot, turn.to_quaternion()) @ target
    foot_matrix = solve_limb(obj, chain, goal)
    if foot_matrix:
        chain.heel_done = new
        _counter_toes(obj, chain, foot_matrix, follow, turn)
        keying.request(obj, chain)


def _rest_rel(bone):
    """Rest transform of a bone relative to its parent's rest transform."""
    if bone.parent:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local
    return bone.matrix_local.copy()


def _swing(reference, local_vector, direction, position):
    """Reference orientation turned so that local_vector points along direction."""
    rotation = reference.to_quaternion().to_matrix()
    turn = (rotation @ local_vector).rotation_difference(direction).to_matrix()
    return Matrix.Translation(position) @ (turn @ rotation).to_4x4()


def _set_basis(pose_bone, parent_matrix, desired):
    pose_bone.matrix_basis = _rest_rel(pose_bone.bone).inverted() @ parent_matrix.inverted() @ desired


def solve_limb(obj, chain, target):
    """Pose thigh, calf and foot so the foot reaches the target matrix, bending toward the pole.

    Returns the foot matrix it left (armature space), or None if it could not solve.
    """
    if not solving_enabled():
        return None
    thigh, calf, foot, _ = _pose_bones(obj, chain)
    if not (thigh and calf and foot) or calf.parent != thigh or foot.parent != calf:
        return None
    b0, b1, b2 = thigh.bone, calf.bone, foot.bone

    parent_matrix = thigh.parent.matrix.copy() if thigh.parent else Matrix.Identity(4)
    thigh_ref = parent_matrix @ _rest_rel(b0)
    root = thigh_ref.translation
    to_calf = b0.matrix_local.inverted() @ b1.head_local  # calf head in thigh space
    to_foot = b1.matrix_local.inverted() @ b2.head_local  # foot head in calf space
    upper, lower = to_calf.length, to_foot.length

    reach = target.translation - root
    distance = reach.length
    if distance < 1e-6:
        return None
    toward = reach / distance
    distance = min(max(distance, abs(upper - lower) + 1e-5), upper + lower - 1e-5)

    # Law of cosines: the joint sits `along` the reach line and `height` off it.
    along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
    height = sqrt(max(upper * upper - along * along, 0.0))
    pole = pose_anchor(obj, chain, "pole_pos") @ Vector(chain.pole_pos)
    side = pole - root
    side -= toward * side.dot(toward)
    if side.length < 1e-6:
        side = calf.head - root
        side -= toward * side.dot(toward)
    if side.length < 1e-6:
        side = toward.orthogonal()
    mid = root + toward * along + side.normalized() * height

    thigh_matrix = _swing(thigh_ref, to_calf, mid - root, root)
    calf_ref = thigh_matrix @ _rest_rel(b1)
    calf_matrix = _swing(calf_ref, to_foot, root + toward * distance - mid, mid)
    foot_matrix = Matrix.Translation(calf_matrix @ to_foot) @ target.to_quaternion().to_matrix().to_4x4()

    _set_basis(thigh, parent_matrix, thigh_matrix)
    _set_basis(calf, thigh_matrix, calf_matrix)
    _set_basis(foot, calf_matrix, foot_matrix)
    parent = ik_parent_matrix(obj, chain)
    chain.end_goal = _flat(parent.inverted() @ target)
    chain.end_parent_left = _flat(parent)
    chain.end_left = _flat(foot_matrix)
    chain.end_basis_left = _flat(foot.matrix_basis)
    chain.end_calf_left = _flat(calf.matrix_basis)
    chain.end_thigh_left = _flat(thigh.matrix_basis)
    chain.end_root_left = _flat(parent_matrix)
    return foot_matrix


# --- pose symmetry -----------------------------------------------------------------------
# With the armature's X-Mirror pose option on, what is done to a chain is repeated, mirrored,
# on the chain that holds the opposite bones.

mirroring = False  # True while a mirrored edit is being applied, so it does not mirror back


@contextmanager
def no_mirroring():
    """Edits made inside are not mirrored (setting a chain up, creating its partner)."""
    global mirroring
    previous = mirroring
    mirroring = True
    try:
        yield
    finally:
        mirroring = previous


def symmetry_partner(obj, chain):
    """The chain edits to this one are mirrored onto, or None when symmetry is off."""
    if mirroring or not chain.symmetric or not obj.pose or not obj.pose.use_mirror_x or not solving_enabled():
        return None
    return mirror_partner(obj, chain)


_MIRROR_X = Matrix.Scale(-1, 4, Vector((1, 0, 0)))


def _mirror_point(point):
    return Vector((-point.x, point.y, point.z))


def _rest_end(obj, chain):
    return _unscaled(obj.data.bones[chain.bone_2].matrix_local)


def _mirror_target(obj, chain, partner, target):
    """The target of the opposite end: the same move away from the rest pose, mirrored."""
    moved = target @ _rest_end(obj, chain).inverted()
    return _MIRROR_X @ moved @ _MIRROR_X @ _rest_end(obj, partner)


def mirror_chain(obj, chain):
    """Solve the opposite chain toward the mirror of this one (end target and heel)."""
    partner = symmetry_partner(obj, chain)
    if not partner or chain.bone_2 not in obj.data.bones or partner.bone_2 not in obj.data.bones:
        return
    with no_mirroring():
        bpy.context.view_layer.update()  # the solve that brought us here is not evaluated yet
        if chain.chain_type == "LEG":
            foot_matrix = solve_limb(obj, partner, _mirror_target(obj, chain, partner, end_target(obj, chain)))
            if not foot_matrix:
                return
            # The angles are only shown: the pose is mirrored directly, so it stays right
            # however the foot or the toes were moved by hand.
            global suspended
            suspended = True
            try:
                partner.heel_pitch = chain.heel_pitch
                partner.heel_roll = -chain.heel_roll
                partner.heel_yaw = -chain.heel_yaw
                partner.heel_done = (chain.heel_done[0], -chain.heel_done[1], -chain.heel_done[2])
            finally:
                suspended = False
            _, _, _, toes = _pose_bones(obj, chain)
            _, _, _, partner_toes = _pose_bones(obj, partner)
            if toes and partner_toes and partner_toes.parent.name == partner.bone_2:
                moved = _unscaled(toes.matrix) @ _unscaled(toes.bone.matrix_local).inverted()
                desired = _MIRROR_X @ moved @ _MIRROR_X @ _unscaled(partner_toes.bone.matrix_local)
                _set_basis(partner_toes, foot_matrix, desired)
        else:
            solve_limb(obj, partner, _mirror_target(obj, chain, partner, end_target(obj, chain)))


def mirror_pole(obj, chain):
    partner = symmetry_partner(obj, chain)
    if partner:
        with no_mirroring():
            world = _mirror_point(pose_anchor(obj, chain, "pole_pos") @ Vector(chain.pole_pos))
            partner.pole_pos = pose_anchor(obj, partner, "pole_pos").inverted() @ world


def mirror_pivot(obj, chain):
    partner = symmetry_partner(obj, chain)
    if partner and chain.chain_type == "LEG":
        with no_mirroring():
            # In the rest pose, like Symmetrize: the pivot is a point of the foot.
            point = rest_anchor(obj, chain, "heel_pos") @ Vector(chain.heel_pos)
            partner.heel_pos = rest_anchor(obj, partner, "heel_pos").inverted() @ _mirror_point(point)
