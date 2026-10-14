"""G on the bones of a chain with IK on: move what the solver is given instead of the bones.

With IK on, the bones of a chain are posed by the solver, so Blender's own G cannot do
anything useful with them. The foot (hand) is a child of the limb being posed: every move
makes the solver move its parent, which moves the foot again, and the foot runs away. The
thigh and calf (arm and forearm) are simply put back where the solver wants them.

So, for a chain with IK on:
- G on the foot or hand moves the IK target, and the limb follows;
- G on the thigh or calf (arm or forearm) moves the pole: the knee or elbow goes where you
  drag it.

It only takes G when every selected bone is one of those; for any other selection G is
Blender's own.

E, with the foot of a leg chain with IK on selected, turns the heel about its pivot with the
mouse. In Pose Mode Blender uses E only with Ctrl, Alt or Shift, so the plain key is free.

Alt+G (Blender's "clear location") puts what G moves back to where it starts: the IK target
to its rest position, the pole to its default. Ctrl+Alt+E resets the heel.
"""

import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from . import heel
from .properties import default_pole, pose_anchor

# What a move changes on a chain, besides the bones' poses.
_CHAIN_PROPS = (
    "end_goal",
    "end_left",
    "end_basis_left",
    "end_parent_left",
    "end_thigh_left",
    "end_calf_left",
    "end_root_left",
    "heel_pitch",
    "heel_roll",
    "heel_yaw",
    "heel_done",
    "heel_neutral",
    "heel_neutral_toes",
    "heel_last",
    "heel_last_toes",
    "heel_toes_edited",
    "heel_plane",
    "pole_pos",
)

END, POLE = "end", "pole"

_KEYMAPS = []

# The poles being moved right now, as (object name, chain index): the gizmos show them even when
# Show Poles is off, so the point being dragged can be seen.
editing_poles = set()


def grabbed(context):
    """What a G would move, as (chain, kind) pairs, or None to leave G to Blender.

    That is when every selected pose bone belongs to a chain with IK on (and IK is not
    switched off): the foot or hand (kind END, its target) or the other bones (kind POLE).
    When the opposite chain of a pair is in the list too, only one of the two is moved: pose
    symmetry moves the other.
    """
    from . import parent_chain

    obj = context.object
    if not obj or obj.type != "ARMATURE" or context.mode != "POSE" or not heel.solving_enabled():
        return None
    if parent_chain.driven(context):
        return None  # the active Parent Chain has these bones
    selected = context.selected_pose_bones
    if not selected:
        return None
    found = []
    for pose_bone in selected:
        name = pose_bone.name
        hit = None
        for chain in obj.live_ik.chains:
            if not chain.ik_enabled:
                continue
            if chain.bone_2 == name:
                hit = (chain, END)
                break
            if name in (chain.bone_0, chain.bone_1):
                hit = (chain, POLE)
                break
        if hit is None:
            return None
        if hit not in found:
            found.append(hit)
    kept = []
    for chain, kind in found:
        if not any(k == kind and heel.symmetry_partner(obj, c) == chain for c, k in kept):
            kept.append((chain, kind))
    return kept


def snapshot(obj, grabs):
    """What a move can change, for a cancel: the poses of the bones and the chains' settings."""
    involved = []
    for chain, _ in grabs:
        for each in (chain, heel.symmetry_partner(obj, chain)):
            if each is not None and each not in involved:
                involved.append(each)
    saved = []
    for chain in involved:
        bones = {}
        for name in (chain.bone_0, chain.bone_1, chain.bone_2, chain.bone_3):
            pose_bone = obj.pose.bones.get(name) if name else None
            if pose_bone:
                bones[name] = pose_bone.matrix_basis.copy()
        props = {name: tuple(getattr(chain, name)) if hasattr(getattr(chain, name), "__len__") else getattr(chain, name)
                 for name in _CHAIN_PROPS}
        saved.append((chain, bones, props))
    return saved


def restore(obj, saved):
    # The settings are put back without running what changing them does (posing, mirroring).
    with heel.no_mirroring():
        heel.suspended = True
        try:
            for chain, bones, props in saved:
                for name, basis in bones.items():
                    obj.pose.bones[name].matrix_basis = basis
                for name, value in props.items():
                    setattr(chain, name, value)
        finally:
            heel.suspended = False


_AXIS_COLORS = ((0.9, 0.25, 0.3, 1.0), (0.5, 0.8, 0.2, 1.0), (0.3, 0.5, 0.95, 1.0))  # X, Y, Z
_LINE_LENGTH = 1000.0


def constrain(delta, axis, plane, direction):
    """A movement kept to an axis, or to the plane perpendicular to it."""
    if axis is None:
        return delta
    along = direction * delta.dot(direction)
    return delta - along if plane else along


class LIVEIK_OT_grab(bpy.types.Operator):
    bl_idname = "live_ik.grab"
    bl_label = "Move IK Target / Pole"
    bl_description = (
        "With IK on, move the foot or hand (its IK target) or the thigh and calf, arm and forearm (the pole). "
        "X / Y / Z constrain (again for local, with Shift for the plane), Shift is for precision"
    )
    bl_options = {"UNDO", "BLOCKING", "GRAB_CURSOR"}

    @classmethod
    def poll(cls, context):
        return bool(grabbed(context))

    def _mouse_location(self, context, event):
        """The mouse in space, at the depth of the point being moved."""
        coord = (event.mouse_region_x, event.mouse_region_y)
        return view3d_utils.region_2d_to_location_3d(context.region, context.region_data, coord, self.depth)

    def _object(self):
        return bpy.data.objects.get(self.obj_name)

    def _direction(self, obj, number):
        """The axis of the constraint, in world space."""
        axis = self.axis
        if not self.local:
            return Vector(tuple(1.0 if i == axis else 0.0 for i in range(3)))
        if self.kinds[number] == END:
            frame = self.starts[number].to_3x3()  # the target's own axes
        else:
            frame = self.starts[number][0].to_3x3()  # the axes of the bone the pole follows
        return (obj.matrix_world.to_quaternion().to_matrix() @ frame.col[axis]).normalized()

    def _apply(self, context):
        obj = self._object()
        if not obj:
            return
        world_to_armature = obj.matrix_world.to_3x3().inverted()
        for number, index in enumerate(self.indices):
            chain = obj.live_ik.chains[index]
            direction = self._direction(obj, number) if self.axis is not None else None
            delta = world_to_armature @ constrain(self.offset, self.axis, self.plane, direction)
            if self.kinds[number] == END:
                heel.move_end(obj, chain, Matrix.Translation(delta) @ self.starts[number])
            else:
                anchor, point = self.starts[number]
                chain.pole_pos = anchor.inverted() @ (point + delta)
        self._status(context)

    def _draw_axes(self):
        """The line (or the two lines of the plane) the move is kept to, through where it started,
        like Blender's own constraint axes."""
        obj = self._object()
        if not obj or self.axis is None:
            return
        shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 2.0)
        axes = [a for a in range(3) if a != self.axis] if self.plane else [self.axis]
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        world = obj.matrix_world
        for number in range(len(self.indices)):
            if self.kinds[number] == END:
                origin = world @ self.starts[number].translation
                frame = self.starts[number].to_3x3()
            else:
                origin = world @ self.starts[number][1]
                frame = self.starts[number][0].to_3x3()
            for axis in axes:
                if self.local:
                    direction = (world.to_quaternion().to_matrix() @ frame.col[axis]).normalized()
                else:
                    direction = Vector(tuple(1.0 if i == axis else 0.0 for i in range(3)))
                points = [origin - direction * _LINE_LENGTH, origin + direction * _LINE_LENGTH]
                shader.uniform_float("color", _AXIS_COLORS[axis])
                batch_for_shader(shader, "LINES", {"pos": points}).draw(shader)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")

    def _status(self, context):
        if self.axis is None:
            where = ""
        else:
            where = f"  [{'Local ' if self.local else ''}{'not ' if self.plane else ''}{'XYZ'[self.axis]}]"
        d = self.offset
        what = "pole" if POLE in self.kinds and END not in self.kinds else "IK target"
        context.workspace.status_text_set(
            f"Move {what}  {d.x:+.3f} {d.y:+.3f} {d.z:+.3f}{where}   "
            "X/Y/Z: constrain (again: local)   Shift+X/Y/Z: plane   Shift: precision   "
            "LMB / Enter: confirm   Esc / RMB: cancel"
        )

    def invoke(self, context, event):
        grabs = grabbed(context)
        if not grabs:
            return {"PASS_THROUGH"}
        obj = context.object
        bpy.context.view_layer.update()  # the pose bone matrices must be current
        self.obj_name = obj.name
        self.indices = [next(i for i, c in enumerate(obj.live_ik.chains) if c == chain) for chain, _ in grabs]
        self.kinds = [kind for _, kind in grabs]
        self.starts = []
        for chain, kind in grabs:
            if kind == END:
                self.starts.append(heel.end_target(obj, chain))
            else:
                anchor = pose_anchor(obj, chain, "pole_pos")
                self.starts.append((anchor, anchor @ Vector(chain.pole_pos)))
        self.saved = snapshot(obj, grabs)
        editing_poles.clear()
        editing_poles.update((obj.name, index) for index, kind in zip(self.indices, self.kinds) if kind == POLE)
        if editing_poles:
            context.area.tag_redraw()
        # The mouse is followed at the depth of what is selected, so the joint stays under it.
        first = context.selected_pose_bones[0]
        self.depth = obj.matrix_world @ first.head
        self.offset = Vector((0.0, 0.0, 0.0))  # the movement so far, in world space
        self.axis = None
        self.plane = False
        self.local = False
        self.last = self._mouse_location(context, event)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw_axes, (), "WINDOW", "POST_VIEW")
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if getattr(self, "_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            self._handle = None
        if context.area:
            context.area.tag_redraw()
        context.workspace.status_text_set(None)
        if editing_poles:
            editing_poles.clear()
            if context.area:
                context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            here = self._mouse_location(context, event)
            self.offset += (here - self.last) * (0.1 if event.shift else 1.0)
            self.last = here
            self._apply(context)
        elif event.type in {"X", "Y", "Z"} and event.value == "PRESS":
            axis = "XYZ".index(event.type)
            if self.axis == axis and self.plane == event.shift:
                if self.local:
                    self.axis, self.local = None, False  # a third press clears it
                else:
                    self.local = True
            else:
                self.axis, self.plane, self.local = axis, event.shift, False
            self._apply(context)
            context.area.tag_redraw()
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            obj = self._object()
            if obj:
                restore(obj, self.saved)
            self._finish(context)
            return {"CANCELLED"}
        elif event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM"}:
            return {"PASS_THROUGH"}  # the view can still be moved around
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        obj = self._object()
        if obj:
            restore(obj, self.saved)
        self._finish(context)


def heel_chains(context):
    """The leg chains an E would turn the heel of, or None to leave E alone.

    That is when every selected pose bone is the foot of a leg chain with IK on.
    """
    grabs = grabbed(context)
    if not grabs:
        return None
    if any(kind != END or chain.chain_type != "LEG" for chain, kind in grabs):
        return None
    return [chain for chain, _ in grabs]


def screen_angle(before, after):
    """The angle, counterclockwise on the screen, from one mouse position to the next, both
    measured from the same centre."""
    return math.atan2(before.x * after.y - before.y * after.x, before.dot(after))


def facing_sign(axis, toward_viewer):
    """+1 when the axis points toward the viewer (a counterclockwise drag then turns the
    foot the positive way about it), -1 when it points away, 0 when it is seen edge-on."""
    facing = axis.dot(toward_viewer)
    return 0 if abs(facing) < 0.15 else (1 if facing > 0 else -1)


_HEEL_PROPS = ("heel_pitch", "heel_roll", "heel_yaw")
_HEEL_NAMES = ("Pitch", "Roll", "Yaw")


class LIVEIK_OT_heel_rotate(bpy.types.Operator):
    bl_idname = "live_ik.heel_rotate"
    bl_label = "Turn Heel"
    bl_description = (
        "With IK on, turn the selected foot about its heel pivot: pitch (raise the heel or the toes), "
        "or roll and yaw. X / Y / Z pick pitch / roll / yaw, Shift is for precision"
    )
    bl_options = {"UNDO", "BLOCKING", "GRAB_CURSOR"}

    @classmethod
    def poll(cls, context):
        return bool(heel_chains(context))

    def _object(self):
        return bpy.data.objects.get(self.obj_name)

    def _mouse(self, event):
        return Vector((event.mouse_region_x, event.mouse_region_y))

    def invoke(self, context, event):
        chains = heel_chains(context)
        if not chains:
            return {"PASS_THROUGH"}
        obj = context.object
        bpy.context.view_layer.update()  # the pose bone matrices must be current
        self.obj_name = obj.name
        self.indices = [next(i for i, c in enumerate(obj.live_ik.chains) if c == chain) for chain in chains]
        self.saved = snapshot(obj, [(chain, END) for chain in chains])
        self.starts = [(chain.heel_pitch, chain.heel_roll, chain.heel_yaw) for chain in chains]
        # Where the pivot is and which way the axes point, kept from the start: the foot moves
        # while it is turned, and a centre or an axis that followed it would feed that
        # movement back into the angle that is read from the mouse.
        world_rotation = obj.matrix_world.to_quaternion().to_matrix()
        self.centers, self.axes = [], []
        for chain in chains:
            pivot, frame, _ = heel.heel_frame(obj, chain)
            self.centers.append(obj.matrix_world @ pivot)
            self.axes.append([(world_rotation @ frame.col[i]).normalized() for i in range(3)])
        region, view = context.region, context.region_data
        center = view3d_utils.location_3d_to_region_2d(region, view, self.centers[0])
        self.center2d = center if center is not None else self._mouse(event)
        viewer = view.view_matrix.inverted().translation
        self.toward = [(viewer - c).normalized() for c in self.centers]
        self.axis = 0  # pitch, the one that is used most
        self.angle = 0.0  # how far the mouse has gone round the pivot
        self.sideways = 0.0  # and how far across, for an axis seen edge-on
        self.last = self._mouse(event) - self.center2d
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _apply(self, context):
        obj = self._object()
        if not obj:
            return
        for number, index in enumerate(self.indices):
            chain = obj.live_ik.chains[index]
            sign = facing_sign(self.axes[number][self.axis], self.toward[number])
            turn = self.angle * sign if sign else self.sideways / 200.0
            for i, prop in enumerate(_HEEL_PROPS):
                value = self.starts[number][i] + (turn if i == self.axis else 0.0)
                if abs(getattr(chain, prop) - value) > 1e-9:
                    setattr(chain, prop, value)
        self._status(context)

    def _status(self, context):
        context.workspace.status_text_set(
            f"Turn heel: {_HEEL_NAMES[self.axis]} {math.degrees(self.angle):+.1f} deg   "
            "X: pitch  Y: roll  Z: yaw   Shift: precision   LMB / Enter: confirm   Esc / RMB: cancel"
        )

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            here = self._mouse(event) - self.center2d
            scale = 0.1 if event.shift else 1.0
            if here.length > 2.0 and self.last.length > 2.0:
                self.angle += screen_angle(self.last, here) * scale
            self.sideways += (here.x - self.last.x) * scale
            self.last = here
            self._apply(context)
        elif event.type in {"X", "Y", "Z"} and event.value == "PRESS":
            self.axis = "XYZ".index(event.type)
            self._apply(context)
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            context.workspace.status_text_set(None)
            return {"FINISHED"}
        elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            obj = self._object()
            if obj:
                restore(obj, self.saved)
            context.workspace.status_text_set(None)
            return {"CANCELLED"}
        elif event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM"}:
            return {"PASS_THROUGH"}  # the view can still be moved around
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        obj = self._object()
        if obj:
            restore(obj, self.saved)
        context.workspace.status_text_set(None)


class LIVEIK_OT_reset_end(bpy.types.Operator):
    bl_idname = "live_ik.reset_end"
    bl_label = "Reset IK Target / Pole"
    bl_description = (
        "With IK on, put the selected foot or hand back at its rest position (its rotation is kept), "
        "or the pole of the thigh and calf, arm and forearm back at its default"
    )
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(grabbed(context))

    def execute(self, context):
        grabs = grabbed(context)
        if not grabs:
            return {"CANCELLED"}
        obj = context.object
        bpy.context.view_layer.update()  # the pose bone matrices must be current
        for chain, kind in grabs:
            if kind == END:
                rest = heel.rest_end_position(obj, chain)
                if rest is None:
                    continue
                target = heel.end_target(obj, chain)
                heel.move_end(obj, chain, Matrix.Translation(rest - target.translation) @ target)
            else:
                default_pole(obj, chain)
        return {"FINISHED"}


class LIVEIK_OT_reset_heel_key(bpy.types.Operator):
    bl_idname = "live_ik.reset_heel_key"
    bl_label = "Reset Heel"
    bl_description = "With IK on, bring the selected foot back from its pitch, roll and yaw about the heel pivot"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(heel_chains(context))

    def execute(self, context):
        chains = heel_chains(context)
        if not chains:
            return {"CANCELLED"}
        for chain in chains:
            # Each slider turns the foot by what it changed: back to 0 is back to where it was.
            chain.heel_pitch = chain.heel_roll = chain.heel_yaw = 0.0
        return {"FINISHED"}


def register():
    bpy.utils.register_class(LIVEIK_OT_grab)
    bpy.utils.register_class(LIVEIK_OT_heel_rotate)
    bpy.utils.register_class(LIVEIK_OT_reset_end)
    bpy.utils.register_class(LIVEIK_OT_reset_heel_key)
    config = bpy.context.window_manager.keyconfigs.addon
    if config:  # there is none when Blender runs without a window
        keymap = config.keymaps.new(name="Pose", space_type="EMPTY")
        for operator, key, modifiers in (
            (LIVEIK_OT_grab, "G", {}),
            (LIVEIK_OT_heel_rotate, "E", {}),
            (LIVEIK_OT_reset_end, "G", {"alt": True}),
            (LIVEIK_OT_reset_heel_key, "E", {"ctrl": True, "alt": True}),
        ):
            _KEYMAPS.append((keymap, keymap.keymap_items.new(operator.bl_idname, key, "PRESS", **modifiers)))


def unregister():
    for keymap, item in _KEYMAPS:
        keymap.keymap_items.remove(item)
    _KEYMAPS.clear()
    bpy.utils.unregister_class(LIVEIK_OT_reset_heel_key)
    bpy.utils.unregister_class(LIVEIK_OT_reset_end)
    bpy.utils.unregister_class(LIVEIK_OT_heel_rotate)
    bpy.utils.unregister_class(LIVEIK_OT_grab)
