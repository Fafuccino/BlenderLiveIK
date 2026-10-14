import bpy
from bpy.props import (
    CollectionProperty,
    EnumProperty,
    BoolProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from mathutils import Matrix, Vector

CHAIN_TYPES = [
    ("ARM", "Arm", "Arm, Forearm, Hand"),
    ("LEG", "Leg", "Thigh, Calf, Foot, Toes"),
]

# Slot labels per chain type. The last slot of a leg (Toes) is optional.
BONE_SLOTS = {
    "ARM": ("Arm", "Forearm", "Hand"),
    "LEG": ("Thigh", "Calf", "Foot", "Toes"),
}
OPTIONAL_SLOTS = {"ARM": 0, "LEG": 1}  # number of optional trailing slots

MAX_CHAINS = 8

IDENTITY_MATRIX = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _bones(obj, chain):
    """Rest-pose bones of the chain, None for unset or unknown names."""
    bones = obj.data.bones
    names = (chain.bone_0, chain.bone_1, chain.bone_2, chain.bone_3)
    return [bones.get(n) if n else None for n in names]


def _unscaled(matrix):
    """Location and rotation of a bone matrix, without any scale."""
    return Matrix.Translation(matrix.to_translation()) @ matrix.to_quaternion().to_matrix().to_4x4()


def mirrored_bones(obj, chain):
    """The chain's bone names with their left/right counterparts, or None.

    None unless every bone of the chain that is set has a counterpart that exists in the
    armature (Blender's own name flipping: .L/.R, _L/_R, Left/Right...), and the required
    bones are all set. Unset optional bones (the toes) stay unset.
    """
    bones = obj.data.bones
    names = (chain.bone_0, chain.bone_1, chain.bone_2, chain.bone_3)
    required = len(BONE_SLOTS[chain.chain_type]) - OPTIONAL_SLOTS[chain.chain_type]
    mirrored = []
    for slot, name in enumerate(names[: len(BONE_SLOTS[chain.chain_type])]):
        if not name:
            if slot < required:
                return None
            mirrored.append("")
            continue
        flipped = bpy.utils.flip_name(name)
        if flipped == name or flipped not in bones:
            return None
        mirrored.append(flipped)
    return mirrored + [""] * (4 - len(mirrored))


def mirror_partner(obj, chain):
    """The chain that holds the mirrored bones of this one, or None."""
    mirrored = mirrored_bones(obj, chain)
    if mirrored:
        for other in obj.live_ik.chains:
            if other != chain and other.chain_type == chain.chain_type and other.bone_0 == mirrored[0]:
                return other
    return None


def valid_ik_parent(obj, chain):
    """Name of the IK parent bone, or None when it is unset, unknown or part of the leg.

    A bone that is the leg's first bone, or below it, cannot be the parent: the foot and
    the pole would be chasing themselves.
    """
    bone = obj.data.bones.get(chain.ik_parent) if chain.ik_parent else None
    name = bone.name if bone else None
    while bone:
        if bone.name == chain.bone_0:
            return None
        bone = bone.parent
    return name


def _anchor_name(obj, chain, prop):
    """Bone a handle sticks to: the heel follows the foot. The pole follows the IK parent when
    the chain has one, otherwise the parent of the chain's first bone (pelvis, chest...).
    None leaves the handle fixed in armature space."""
    b0, _, b2, _ = _bones(obj, chain)
    if prop == "heel_pos":
        return b2.name if b2 else None
    if valid_ik_parent(obj, chain):
        return valid_ik_parent(obj, chain)
    return b0.parent.name if b0 and b0.parent else None


def bone_pose_matrix(obj, name):
    """Armature-space matrix of a bone in the current pose (no scale), identity if unknown."""
    pose_bone = obj.pose.bones.get(name) if name and obj.pose else None
    return _unscaled(pose_bone.matrix) if pose_bone else Matrix.Identity(4)


def reparent_pole(obj, chain):
    """The pole's anchor changed: keep the pole where it is, relative to the new anchor."""
    b0 = _bones(obj, chain)[0]
    # Files from before the IK parent existed have no record: the anchor was the thigh's parent.
    old_name = chain.pole_anchor or (b0.parent.name if b0 and b0.parent else "")
    new_name = _anchor_name(obj, chain, "pole_pos") or ""
    world = bone_pose_matrix(obj, old_name) @ Vector(chain.pole_pos)
    chain.pole_anchor = new_name
    chain.pole_pos = bone_pose_matrix(obj, new_name).inverted() @ world


def rest_anchor(obj, chain, prop):
    """Armature-space matrix of the handle's anchor in the rest pose."""
    name = _anchor_name(obj, chain, prop)
    bone = obj.data.bones.get(name) if name else None
    return _unscaled(bone.matrix_local) if bone else Matrix.Identity(4)


def pose_anchor(obj, chain, prop):
    """Armature-space matrix of the handle's anchor in the current pose."""
    name = _anchor_name(obj, chain, prop)
    pose_bone = obj.pose.bones.get(name) if name and obj.pose else None
    return _unscaled(pose_bone.matrix) if pose_bone else Matrix.Identity(4)


def default_pole(obj, chain):
    """Put the pole where it is by default: on the side the limb bends toward at rest."""
    b0, b1, b2, _ = _bones(obj, chain)
    if not (b0 and b1 and b2):
        return
    root, mid, end = b0.head_local, b1.head_local, b2.head_local
    length = (mid - root).length + (end - mid).length

    bend = mid - (root + end) / 2
    if bend.length < length * 0.01:
        # Straight chain: elbows point back, knees point forward (-Y is front).
        bend = Vector((0, 1, 0)) if chain.chain_type == "ARM" else Vector((0, -1, 0))
    pole = mid + bend.normalized() * length * 0.5
    chain.pole_pos = rest_anchor(obj, chain, "pole_pos").inverted() @ pole
    chain.pole_angle = 0.0
    chain.pole_anchor = _anchor_name(obj, chain, "pole_pos") or ""


def compute_defaults(obj, chain):
    """Place the pole and heel from the rest pose. Stored relative to their anchor bone."""
    b0, b1, b2, b3 = _bones(obj, chain)
    if not (b0 and b1 and b2):
        return
    default_pole(obj, chain)

    if chain.chain_type == "LEG":
        low = min(b2.head_local.z, b2.tail_local.z)
        if b3:
            low = min(low, b3.tail_local.z)
        heel = Vector((b2.head_local.x, b2.head_local.y, low))
        chain.heel_pos = rest_anchor(obj, chain, "heel_pos").inverted() @ heel


def _redraw(self, context):
    for area in context.screen.areas:
        area.tag_redraw()


def _on_symmetric_changed(self, context):
    """The two chains of a pair share the toggle: the partner follows."""
    obj = self.id_data
    if isinstance(obj, bpy.types.Object) and obj.type == "ARMATURE":
        partner = mirror_partner(obj, self)
        if partner and partner.symmetric != self.symmetric:
            partner.symmetric = self.symmetric  # its own update finds them equal and stops
    _redraw(self, context)


quiet = False  # True while values are put back or renamed by the add-on itself (state.py, names.py)


def _on_mode_changed(self, context):
    """Switching IK off forgets what the solver remembered: nothing keeps the limbs where it
    left them, so when it is switched back on it takes over from the pose as it is then."""
    from . import heel

    if quiet:
        return
    if context.window_manager.live_ik_gizmo_mode == "OFF":
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                for chain in obj.live_ik.chains:
                    heel.clear(chain)
    _redraw(self, context)


def _on_pivot_changed(self, context):
    from . import heel

    obj = context.object
    if obj and obj.type == "ARMATURE":
        heel.mirror_pivot(obj, self)
    for area in context.screen.areas:
        area.tag_redraw()


def _on_pole_changed(self, context):
    from . import heel

    obj = context.object
    if obj and obj.type == "ARMATURE":
        heel.resolve(obj, self)
        heel.mirror_pole(obj, self)
    for area in context.screen.areas:
        area.tag_redraw()


def _on_chain_changed(self, context):
    from . import heel

    if quiet:
        return
    obj = context.object
    if obj and obj.type == "ARMATURE":
        with heel.no_mirroring():  # setting a chain up must not overwrite its partner
            heel.clear(self)
            compute_defaults(obj, self)
    for area in context.screen.areas:
        area.tag_redraw()


def _on_heel_changed(self, context):
    from . import heel

    obj = context.object
    if obj and obj.type == "ARMATURE":
        heel.apply_heel(obj, self)
        heel.mirror_chain(obj, self)


def _on_ik_parent_changed(self, context):
    from . import heel

    if quiet:
        return
    obj = context.object
    if obj and obj.type == "ARMATURE":
        with heel.no_mirroring():
            heel.reparent(obj, self)
        if self.ik_enabled:
            heel.sync_ik(obj, self)


def _on_ik_toggled(self, context):
    from . import heel

    obj = context.object
    if obj and obj.type == "ARMATURE" and self.ik_enabled:
        heel.sync_ik(obj, self)


def _armature(chain):
    obj = chain.id_data
    return obj if isinstance(obj, bpy.types.Object) and obj.type == "ARMATURE" else None


def _get_pole_offset(self):
    from . import heel

    obj = _armature(self)
    return heel.pole_offset(obj, self) if obj else self.pole_pos


def _set_pole_offset(self, value):
    from . import heel

    obj = _armature(self)
    if obj:
        heel.set_pole_offset(obj, self, Vector(value))


def _get_heel_offset(self):
    from . import heel

    obj = _armature(self)
    return heel.pivot_offset(obj, self) if obj else self.heel_pos


def _set_heel_offset(self, value):
    from . import heel

    obj = _armature(self)
    if obj:
        heel.set_pivot_offset(obj, self, Vector(value))


class LiveIKChain(bpy.types.PropertyGroup):
    chain_type: EnumProperty(name="Type", items=CHAIN_TYPES, default="ARM", update=_on_chain_changed)
    bone_0: StringProperty(name="Bone 1", update=_on_chain_changed)
    bone_1: StringProperty(name="Bone 2", update=_on_chain_changed)
    bone_2: StringProperty(name="Bone 3", update=_on_chain_changed)
    bone_3: StringProperty(name="Bone 4", update=_on_chain_changed)
    # Stored relative to the anchor bone (see pose_anchor). The *_offset properties below
    # are what the panel shows and edits: the same point along the gizmo's axes.
    pole_pos: FloatVectorProperty(name="IK Pole", subtype="TRANSLATION", size=3, update=_on_pole_changed)
    heel_pos: FloatVectorProperty(name="IK Heel Pivot", subtype="TRANSLATION", size=3, update=_on_pivot_changed)
    pole_angle: FloatProperty(
        name="Pole Angle",
        description="Turns the bend of the knee or elbow around the line from the hip to the foot",
        subtype="ANGLE",
        update=_on_pole_changed,
    )
    pole_offset: FloatVectorProperty(
        name="IK Pole",
        description="From the bone the pole follows, along the armature's axes",
        subtype="TRANSLATION",
        size=3,
        get=_get_pole_offset,
        set=_set_pole_offset,
    )
    heel_offset: FloatVectorProperty(
        name="IK Heel Pivot",
        description="From the foot, along its ground axes: X right, Y away from the foot, Z up",
        subtype="TRANSLATION",
        size=3,
        get=_get_heel_offset,
        set=_set_heel_offset,
    )
    heel_pitch: FloatProperty(
        name="Pitch", subtype="ANGLE", description="Raise the heel or the toes", update=_on_heel_changed
    )
    heel_roll: FloatProperty(name="Roll", subtype="ANGLE", description="Bank the foot", update=_on_heel_changed)
    heel_yaw: FloatProperty(name="Yaw", subtype="ANGLE", description="Swivel the foot", update=_on_heel_changed)
    # The rotation currently applied to the foot (armature space), to undo it when it changes.
    # The pitch / roll / yaw the foot has been turned by so far: a change of the sliders turns it by the difference.
    heel_done: FloatVectorProperty(name="Done", size=3)
    # The foot target and the toes' pose from before the heel was first turned, and what the
    # heel left them as: sliders back at 0 return to the first when nothing touched the second.
    heel_neutral: FloatVectorProperty(name="Neutral", size=16, default=IDENTITY_MATRIX)
    heel_neutral_toes: FloatVectorProperty(name="Neutral Toes", size=16, default=IDENTITY_MATRIX)
    heel_last: FloatVectorProperty(name="Last", size=16, default=IDENTITY_MATRIX)
    heel_last_toes: FloatVectorProperty(name="Last Toes", size=16, default=IDENTITY_MATRIX)
    # The toes were turned by hand while the heel was turned: their neutral pose is no longer theirs.
    heel_toes_edited: BoolProperty(name="Toes Edited", default=False)
    # The sole plane: the rotation (armature space, about the toes' head) of the plane the foot
    # lies on, away from the ground. Identity while only the heel moves the foot; when the toes
    # are turned some other way it is turned with them, and the heel then works against it.
    heel_plane: FloatVectorProperty(name="Sole Plane", subtype="QUATERNION", size=4, default=(1.0, 0.0, 0.0, 0.0))
    # What the IK solver remembers (row-major matrices): the target it was given, the foot
    # matrix it left in armature space, and the foot's local pose it left. See heel.end_target.
    end_goal: FloatVectorProperty(name="Foot Goal", size=16, default=IDENTITY_MATRIX)
    end_left: FloatVectorProperty(name="Foot Left", size=16, default=IDENTITY_MATRIX)
    end_basis_left: FloatVectorProperty(name="Foot Basis Left", size=16, default=IDENTITY_MATRIX)
    end_parent_left: FloatVectorProperty(name="Foot Parent Left", size=16, default=IDENTITY_MATRIX)
    # The pose of thigh and calf the solver left, and the matrix of the bone they hang from.
    end_thigh_left: FloatVectorProperty(name="Thigh Left", size=16, default=IDENTITY_MATRIX)
    end_calf_left: FloatVectorProperty(name="Calf Left", size=16, default=IDENTITY_MATRIX)
    end_root_left: FloatVectorProperty(name="Root Left", size=16, default=IDENTITY_MATRIX)
    # The bone the pole was last measured from, to move it when that changes.
    pole_anchor: StringProperty(name="Pole Anchor")
    ik_parent: StringProperty(
        name="IK Parent",
        description="Optional bone the IK foot and its pole are parented to: they follow it when it moves",
        update=_on_ik_parent_changed,
    )
    symmetric: BoolProperty(
        name="Pose Symmetry",
        description="With the armature's X-Mirror pose option on, what is done to this chain is mirrored "
        "onto the opposite chain: IK solving, heel, pole and pivot. Shared with that chain",
        default=True,
        update=_on_symmetric_changed,
    )
    visible: BoolProperty(
        name="Visible",
        description="Show this chain's gizmos in the viewport",
        default=True,
        update=_redraw,
    )
    ik_enabled: BoolProperty(
        name="IK",
        description="Keep the chain solved by IK while posing: moving the foot or hand bone moves the limb, "
        "and it stays planted when something above it moves",
        default=False,
        update=_on_ik_toggled,
    )


def _on_count_changed(self, context):
    chains = self.chains
    while len(chains) < self.chain_count:
        chains.add()
    while len(chains) > self.chain_count:
        chains.remove(len(chains) - 1)
    for area in context.screen.areas:
        area.tag_redraw()


class LiveIKSettings(bpy.types.PropertyGroup):
    chain_count: IntProperty(name="Chains", min=0, max=MAX_CHAINS, default=0, update=_on_count_changed)
    chains: CollectionProperty(type=LiveIKChain)


classes = (LiveIKChain, LiveIKSettings)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # UI state, kept on the window manager so changing it is not an undo step of its own.
    bpy.types.WindowManager.live_ik_gizmo_mode = EnumProperty(
        name="Gizmos",
        items=[
            ("OFF", "Off", "No IK solving at all, whatever each chain says"),
            ("HIDE", "Hide", "Gizmos hidden. Chains with IK on keep solving"),
            ("SETUP", "Place", "Move the heel pivot"),
            ("MOVE", "Move", "Move the IK feet and hands, their poles, and turn the heels"),
        ],
        default="HIDE",
        update=_on_mode_changed,
    )
    bpy.types.WindowManager.live_ik_show_ends = BoolProperty(
        name="Show Ends",
        description="In Move mode, show the handles that move each foot or hand (G does the same with the bone selected)",
        default=True,
        update=_redraw,
    )
    bpy.types.WindowManager.live_ik_show_poles = BoolProperty(
        name="Show Poles",
        description="In Move mode, always show the pole of each visible chain. Off: a pole is only shown "
        "while it is being moved with G on the thigh or calf (arm or forearm)",
        default=True,
        update=_redraw,
    )
    bpy.types.WindowManager.live_ik_show_heels = BoolProperty(
        name="Show Heels",
        description="In Move mode, show the heel dials that rotate each foot around its pivot",
        default=True,
        update=_redraw,
    )
    bpy.types.WindowManager.live_ik_orientation = EnumProperty(
        name="Orientation",
        items=[
            ("WORLD", "World", "Gizmos follow the scene's axes"),
            ("RIG", "Rig", "Gizmos follow the armature object's axes"),
            ("LOCAL", "Local", "Gizmos follow the axes of what they move: the foot or hand, the foot's ground frame "
             "for the heel pivot, the bone the pole is attached to"),
        ],
        default="WORLD",
        update=_redraw,
    )
    bpy.types.Object.live_ik = bpy.props.PointerProperty(type=LiveIKSettings)


def unregister():
    del bpy.types.WindowManager.live_ik_orientation
    del bpy.types.WindowManager.live_ik_show_heels
    del bpy.types.WindowManager.live_ik_show_poles
    del bpy.types.WindowManager.live_ik_show_ends
    del bpy.types.WindowManager.live_ik_gizmo_mode
    del bpy.types.Object.live_ik
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
