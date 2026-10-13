import bpy

from mathutils import Vector

from . import heel, keying
from .properties import (
    BONE_SLOTS,
    MAX_CHAINS,
    OPTIONAL_SLOTS,
    compute_defaults,
    mirror_partner,
    mirrored_bones,
    rest_anchor,
)


class LIVEIK_OT_reset_gizmos(bpy.types.Operator):
    bl_idname = "live_ik.reset_gizmos"
    bl_label = "Reset Pole / Heel"
    bl_description = "Put the IK pole and heel back to their default position for this chain"
    # No REGISTER: the index is only meaningful for the button that set it, so the
    # operator must not show up in the redo panel or be replayed by "Repeat Last".
    bl_options = {"UNDO", "INTERNAL"}

    index: bpy.props.IntProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "ARMATURE"

    def execute(self, context):
        chains = context.object.live_ik.chains
        if not 0 <= self.index < len(chains):
            self.report({"WARNING"}, "That chain no longer exists")
            return {"CANCELLED"}
        chain = chains[self.index]
        chain.heel_pitch = chain.heel_roll = chain.heel_yaw = 0.0  # un-pose the foot first
        compute_defaults(context.object, chain)
        return {"FINISHED"}


class LIVEIK_OT_reset_heel(bpy.types.Operator):
    bl_idname = "live_ik.reset_heel"
    bl_label = "Reset Heel Rotation"
    bl_description = "Bring the foot back from its pitch, roll and yaw around the heel pivot"
    bl_options = {"UNDO", "INTERNAL"}

    index: bpy.props.IntProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "ARMATURE"

    def execute(self, context):
        chains = context.object.live_ik.chains
        if not 0 <= self.index < len(chains):
            return {"CANCELLED"}
        chain = chains[self.index]
        chain.heel_pitch = chain.heel_roll = chain.heel_yaw = 0.0
        return {"FINISHED"}


# Chain panels that are unfolded, as (object name, chain index): they all start folded.
# Kept out of the object so folding a panel is not an undo step and does not touch the file.
_expanded = set()


class LIVEIK_OT_toggle_chain(bpy.types.Operator):
    bl_idname = "live_ik.toggle_chain"
    bl_label = "Fold Chain"
    bl_description = "Fold or unfold this chain's settings"
    bl_options = {"INTERNAL"}  # no UNDO: it is UI state

    index: bpy.props.IntProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "ARMATURE"

    def execute(self, context):
        _expanded.symmetric_difference_update({(context.object.name, self.index)})
        for area in context.screen.areas:
            area.tag_redraw()
        return {"FINISHED"}


class LIVEIK_OT_key_chain(bpy.types.Operator):
    bl_idname = "live_ik.key_chain"
    bl_label = "Key Chain"
    bl_description = (
        "Insert keyframes on the bones of this chain at the current frame "
        "(and on the opposite chain when pose symmetry is on)"
    )
    bl_options = {"UNDO", "INTERNAL"}

    index: bpy.props.IntProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "ARMATURE"

    def execute(self, context):
        chains = context.object.live_ik.chains
        if not 0 <= self.index < len(chains):
            return {"CANCELLED"}
        keying.key_chain(context.object, chains[self.index])
        return {"FINISHED"}


def mirror_target(obj, chain):
    """Index of the chain that already holds the mirrored bones, or None."""
    mirrored = mirrored_bones(obj, chain)
    if mirrored:
        for index, other in enumerate(obj.live_ik.chains):
            if other.chain_type == chain.chain_type and other.bone_0 == mirrored[0]:
                return index
    return None


def _mirror_x(point):
    return Vector((-point.x, point.y, point.z))


class LIVEIK_OT_symmetrize(bpy.types.Operator):
    bl_idname = "live_ik.symmetrize"
    bl_label = "Symmetrize"
    bl_description = (
        "Copy this chain to the opposite side: the mirrored bones, IK parent, pole and heel pivot. "
        "Updates the chain that already uses the mirrored bones, or adds a new one"
    )
    bl_options = {"UNDO", "INTERNAL"}

    index: bpy.props.IntProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "ARMATURE"

    def execute(self, context):
        with heel.no_mirroring():  # the new chain is set up from this one, not edited
            return self._symmetrize(context)

    def _symmetrize(self, context):
        obj = context.object
        settings = obj.live_ik
        if not 0 <= self.index < len(settings.chains):
            return {"CANCELLED"}
        source = settings.chains[self.index]
        mirrored = mirrored_bones(obj, source)
        if not mirrored:
            self.report({"WARNING"}, "Every bone of the chain needs a left/right counterpart")
            return {"CANCELLED"}

        target_index = mirror_target(obj, source)
        if target_index is None:
            if len(settings.chains) >= MAX_CHAINS:
                self.report({"WARNING"}, "No room for another chain")
                return {"CANCELLED"}
            settings.chain_count = len(settings.chains) + 1
            target_index = len(settings.chains) - 1
        target = settings.chains[target_index]

        # Setting the type and the bones places the defaults; the mirrored setup replaces them.
        target.chain_type = source.chain_type
        target.bone_0, target.bone_1, target.bone_2, target.bone_3 = mirrored
        target.symmetric = source.symmetric
        target.ik_parent = ""
        if source.ik_parent:
            # A parent without a counterpart (root, hips...) is shared by both sides.
            flipped = bpy.utils.flip_name(source.ik_parent)
            target.ik_parent = flipped if flipped in obj.data.bones else source.ik_parent

        # Mirrored in the rest pose, so a pose on either side does not matter.
        for prop in ("pole_pos", "heel_pos"):
            if prop == "heel_pos" and source.chain_type != "LEG":
                continue
            point = rest_anchor(obj, source, prop) @ Vector(getattr(source, prop))
            setattr(target, prop, rest_anchor(obj, target, prop).inverted() @ _mirror_x(point))
        return {"FINISHED"}


class LIVEIK_PT_main(bpy.types.Panel):
    bl_label = "LiveIK"
    bl_idname = "LIVEIK_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LiveIK"

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "ARMATURE"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        settings = obj.live_ik

        layout.prop(settings, "chain_count")
        layout.prop(context.window_manager, "live_ik_gizmo_mode", expand=True)
        layout.prop(context.window_manager, "live_ik_orientation", expand=True)
        row = layout.row()
        row.active = context.window_manager.live_ik_gizmo_mode == "MOVE"  # the dials live in Move mode
        row.prop(context.window_manager, "live_ik_show_heels")

        off = context.window_manager.live_ik_gizmo_mode == "OFF"
        for i, chain in enumerate(settings.chains):
            box = layout.box()
            box.active = not off  # IK is off: the chains are only a setup
            collapsed = (obj.name, i) not in _expanded
            row = box.row(align=True)
            arrow = "TRIA_RIGHT" if collapsed else "TRIA_DOWN"
            row.operator("live_ik.toggle_chain", text="", icon=arrow, emboss=False).index = i
            title = f"Chain {i + 1}: {chain.chain_type.title()}"
            row.label(text=f"{title} {chain.bone_0}" if chain.bone_0 else title)
            row.prop(chain, "ik_enabled", toggle=True)
            if obj.pose and obj.pose.use_mirror_x and mirror_partner(obj, chain):
                row.prop(chain, "symmetric", text="", icon="MOD_MIRROR", toggle=True)
            row.prop(chain, "visible", text="", icon="HIDE_OFF" if chain.visible else "HIDE_ON", emboss=False)
            row.operator("live_ik.key_chain", text="", icon="KEY_HLT").index = i
            row.operator("live_ik.reset_gizmos", text="", icon="LOOP_BACK").index = i
            if collapsed:
                continue
            box.prop(chain, "chain_type", expand=True)
            if mirrored_bones(obj, chain):
                target_index = mirror_target(obj, chain)
                if target_index is not None:
                    text = f"Symmetrize to Chain {target_index + 1}"
                elif len(settings.chains) < MAX_CHAINS:
                    text = "Symmetrize to New Chain"
                else:
                    text = None
                if text:
                    box.operator("live_ik.symmetrize", text=text, icon="MOD_MIRROR").index = i

            labels = BONE_SLOTS[chain.chain_type]
            first_optional = len(labels) - OPTIONAL_SLOTS[chain.chain_type]
            for slot, label in enumerate(labels):
                text = f"{label} (optional)" if slot >= first_optional else label
                box.prop_search(chain, f"bone_{slot}", obj.data, "bones", text=text)

            box.prop_search(chain, "ik_parent", obj.data, "bones", text="IK Parent (optional)")

            col = box.column(align=True)
            col.prop(chain, "pole_offset", text="Pole")

            if chain.chain_type == "LEG":
                col.prop(chain, "heel_offset", text="Pivot")

                col = box.column(align=True)
                row = col.row()
                row.label(text="Heel")
                op = row.operator("live_ik.reset_heel", text="", icon="LOOP_BACK")
                op.index = i
                col.prop(chain, "heel_pitch")
                col.prop(chain, "heel_roll")
                col.prop(chain, "heel_yaw")


classes = (
    LIVEIK_OT_reset_gizmos,
    LIVEIK_OT_reset_heel,
    LIVEIK_OT_toggle_chain,
    LIVEIK_OT_key_chain,
    LIVEIK_OT_symmetrize,
    LIVEIK_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
