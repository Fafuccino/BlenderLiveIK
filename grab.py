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
from mathutils import Matrix, Quaternion, Vector, geometry

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
    "joint_pos",
)

END, POLE = "end", "pole"


def pole_prop(chain):
    """The setting a POLE move changes: the pinned joint, or else the pole."""
    return "joint_pos" if chain.joint_pinned else "pole_pos"

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


def trackball_rotation(right, up, moved):
    """The trackball rotation, as Blender works it out (transform_mode_trackball.cc).

    `moved` is how far the mouse has gone since the trackball began, in pixels. 0.01 radian per
    pixel (down about the view's horizontal axis `right`, right about its vertical one `up`) makes
    one rotation: the axis is the two axes weighted by those angles and the angle is the length of
    that. It is a function of where the mouse is, not a sum of its steps, so going round in circles
    brings it back."""
    axis = right * (-moved.y * 0.01) + up * (moved.x * 0.01)
    angle = axis.length
    return Quaternion(axis / angle, angle) if angle > 1e-9 else Quaternion()


def draw_trackball_cursor(centre):
    """Blender's trackball helper at the mouse (region pixels): arrows across (red) and up and
    down (green). The system cursor is hidden while it is shown."""
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", 2.0)
    gpu.state.blend_set("ALPHA")
    gap, length, head = 5.0, 16.0, 5.0
    for direction, color in ((Vector((1.0, 0.0)), _AXIS_COLORS[0]), (Vector((0.0, 1.0)), _AXIS_COLORS[1])):
        across = Vector((-direction.y, direction.x))
        segments = []
        for sign in (1.0, -1.0):
            start, tip = centre + direction * gap * sign, centre + direction * length * sign
            segments += [start, tip]
            for side in (1.0, -1.0):
                segments += [tip, tip - direction * head * sign + across * head * side]
        shader.uniform_float("color", color)
        points = [(point.x, point.y, 0.0) for point in segments]
        batch_for_shader(shader, "LINES", {"pos": points}).draw(shader)
    gpu.state.blend_set("NONE")


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
                setattr(chain, pole_prop(chain), anchor.inverted() @ (point + delta))
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
                prop = pole_prop(chain)
                anchor = pose_anchor(obj, chain, prop)
                self.starts.append((anchor, anchor @ Vector(getattr(chain, prop))))
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
        "or roll and yaw. X / Y / Z pick the axis (again for the other space), E again is the trackball, Shift is for precision"
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
        self.world_rotation = obj.matrix_world.to_quaternion().to_matrix()
        self.centers, self.feet, self.frames, self.start_feet = [], [], [], []
        # The foot is turned about the panel's orientation: the scene's axes (World), the armature's
        # (Rig) or the foot's own as it is posed (Local). Pressing the axis key again swaps it with
        # Local (or with World, when it started as Local).
        self.start_space = context.window_manager.live_ik_orientation
        self.space = self.start_space
        self.pressed = False  # an axis has been picked with its key
        for chain in chains:
            pivot, _, _ = heel.heel_frame(obj, chain)
            self.centers.append(obj.matrix_world @ pivot)
            self.feet.append(heel.foot_axes(obj, chain))  # its own axes, kept from the start
            self.start_feet.append(obj.pose.bones[chain.bone_2].matrix.copy())  # where it began, to find its pivot
            self.frames.append(heel.angle_frame(obj, chain))  # what the angles are measured in
        region, view = context.region, context.region_data
        center = view3d_utils.location_3d_to_region_2d(region, view, self.centers[0])
        self.center2d = center if center is not None else self._mouse(event)
        viewer = view.view_matrix.inverted().translation
        self.toward = [(viewer - c).normalized() for c in self.centers]
        self.axis = 0  # pitch, the one that is used most
        self.angle = 0.0  # how far the mouse has gone round the axis
        self.sideways = 0.0  # and how far across, for an axis seen edge-on
        self.last = self._mouse(event) - self.center2d
        self.last_plane = self._plane_point(context, event)
        # E again: the trackball, as Blender's (see trackball_rotation), turning about the view's own axes.
        self.trackball = False
        self.virtual = Vector((0.0, 0.0))  # how far the mouse has gone since the trackball began, in pixels
        self._cursor = None
        view_axes = view.view_matrix.inverted().to_3x3()
        self.right, self.up = view_axes.col[0].normalized(), view_axes.col[1].normalized()
        self.mouse = self._mouse(event)
        self._axis_handle = bpy.types.SpaceView3D.draw_handler_add(self._draw_axis, (), "WINDOW", "POST_VIEW")
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _axis(self, obj, number):
        """The axis the heel is turned about, in armature space (for the chain's number)."""
        orient = heel.space_axes(obj, self.space)
        return (self.feet[number] if orient is None else orient).col[self.axis].normalized()

    def _edge_on(self, obj):
        """True when the axis is seen (nearly) edge-on: the mouse can not go round it on the screen."""
        axis = self.world_rotation @ self._axis(obj, 0)
        return abs(axis.dot(self.toward[0])) < 0.15

    def _plane_point(self, context, event):
        """Where the mouse is on the plane across the axis through the pivot, relative to the pivot,
        or None. The angle the foot is turned by is the angle this goes round the axis: the true turn
        in space, as for Blender's own dials, so it is the same whichever way the view looks."""
        obj = self._object()
        if not obj:
            return None
        coord = (event.mouse_region_x, event.mouse_region_y)
        region, view = context.region, context.region_data
        axis = self.world_rotation @ self._axis(obj, 0)
        origin = view3d_utils.region_2d_to_origin_3d(region, view, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, view, coord)
        if abs(direction.dot(axis)) < 0.1:
            return None  # the plane is seen edge-on: the mouse says nothing about where it is on it
        hit = geometry.intersect_line_plane(origin, origin + direction, self.centers[0], axis)
        return hit - self.centers[0] if hit else None

    def _apply(self, context):
        obj = self._object()
        if not obj:
            return
        if self.trackball:
            axis, turn = trackball_rotation(self.right, self.up, self.virtual).to_axis_angle()
            axis = self.world_rotation.inverted() @ axis  # armature space, for every chain
        else:
            turn = self.sideways / 200.0 if self._edge_on(obj) else self.angle
        for number, index in enumerate(self.indices):
            chain = obj.live_ik.chains[index]
            about = axis if self.trackball else self._axis(obj, number)
            angles = heel.angles_after_turn(self.frames[number], self.starts[number], about, turn)
            for prop, value in zip(_HEEL_PROPS, angles):
                if abs(getattr(chain, prop) - value) > 1e-9:
                    setattr(chain, prop, value)
        self._status(context)

    def _status(self, context):
        if self.trackball:
            about, keys = "trackball", "E: axis   X / Y / Z: the axis"
        else:
            other = "Local" if self.start_space != "LOCAL" else "World"
            about = f"about {self.space.title()} {'XYZ'[self.axis]} {math.degrees(self.angle):+.1f} deg"
            keys = f"E: trackball   X / Y / Z: the axis (again: {other})"
        context.workspace.status_text_set(
            f"Turn heel: {about}   {keys}   Shift: precision   LMB / Enter: confirm   Esc / RMB: cancel"
        )

    def _set_trackball(self, context, on):
        """Switch the trackball on or off, each time from the start. Blender hides the cursor
        while it is on and draws its four arrows there instead."""
        self.trackball = on
        self.virtual = Vector((0.0, 0.0))
        self.last_plane = None
        if context.window:
            if on:
                context.window.cursor_modal_set("NONE")
            else:
                context.window.cursor_modal_restore()
        if on and self._cursor is None:
            self._cursor = bpy.types.SpaceView3D.draw_handler_add(self._draw_cursor, (), "WINDOW", "POST_PIXEL")
        elif not on and self._cursor is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._cursor, "WINDOW")
            self._cursor = None
        if context.area:
            context.area.tag_redraw()

    def _draw_cursor(self):
        draw_trackball_cursor(self.mouse)

    def _pivot(self, obj, number):
        """The point the foot is turning about right now, in world space.

        The heel pivot is where the turn is meant to be about, but the foot also moves so the
        toes stay planted: raising the heel turns it about the toes' head, for instance. So the
        pivot is read from what the foot has done since the turn began: a rigid motion is a turn
        about some line, and this is where that line goes. Until the foot has turned (a degree),
        which line is not defined, and the heel pivot stands for it."""
        fallback = self.centers[number]
        foot = obj.pose.bones.get(obj.live_ik.chains[self.indices[number]].bone_2)
        if not foot:
            return fallback
        net = foot.matrix @ self.start_feet[number].inverted()  # armature space
        axis, angle = net.to_3x3().normalized().to_quaternion().to_axis_angle()
        if min(angle, 2.0 * math.pi - angle) < 0.02:
            return fallback
        move = net.translation
        across = move - axis * axis.dot(move)  # what is left of the movement across the axis
        # The turn by `angle` that does this movement: the point it leaves where it is (the
        # fixed point of x -> R x + move, in the plane across the axis).
        centre = across * 0.5 + axis.cross(across) * (0.5 / math.tan(angle / 2.0))
        return obj.matrix_world @ centre

    def _draw_axis(self):
        """The axis the heel is turned about, as a line through each pivot in the colour of that axis,
        like the line a locked axis shows in the other modes."""
        obj = self._object()
        if not obj or self.trackball:
            return
        shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 2.0)
        shader.uniform_float("color", _AXIS_COLORS[self.axis])
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        for number in range(len(self.centers)):
            origin = self._pivot(obj, number)
            direction = (self.world_rotation @ self._axis(obj, number)).normalized()
            points = [origin - direction * _LINE_LENGTH, origin + direction * _LINE_LENGTH]
            batch_for_shader(shader, "LINES", {"pos": points}).draw(shader)
        gpu.state.blend_set("NONE")

    def _leave(self, context):
        """The turn is over, confirmed or not: the cursor and its arrows go."""
        if self.trackball:
            self._set_trackball(context, False)
        if getattr(self, "_axis_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._axis_handle, "WINDOW")
            self._axis_handle = None
        if context.area:
            context.area.tag_redraw()
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            here = self._mouse(event) - self.center2d
            scale = 0.1 if event.shift else 1.0
            self.mouse = self._mouse(event)
            if self.trackball:
                # What the mouse moved this time (the window's own steps, which stay right when the
                # cursor wraps at the edge), added up; Shift slows it to a thirtieth, like Blender.
                step = Vector((event.mouse_x - event.mouse_prev_x, event.mouse_y - event.mouse_prev_y))
                self.virtual += step * (1.0 / 30.0 if event.shift else 1.0)
                self.last = here
                self._apply(context)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            plane = self._plane_point(context, event)
            before = self.last_plane
            if before is not None and plane is not None and before.length > 1e-6 and plane.length > 1e-6:
                axis = self.world_rotation @ self._axis(self._object(), 0)
                # Right-handed about the axis: a foot turned by this follows the mouse round it.
                self.angle += math.atan2(axis.dot(before.cross(plane)), before.dot(plane)) * scale
            self.last_plane = plane
            self.sideways += (here.x - self.last.x) * scale
            self.last = here
            self._apply(context)
        elif event.type == "E" and event.value == "PRESS" and not event.is_repeat:
            # Back and forth between turning about an axis and the trackball, each from the start.
            self._set_trackball(context, not self.trackball)
            self.angle = 0.0
            self._apply(context)
        elif event.type in {"X", "Y", "Z"} and event.value == "PRESS":
            if self.trackball:
                self._set_trackball(context, False)  # an axis: back to turning about it
                self.angle = 0.0
            axis = "XYZ".index(event.type)
            if self.pressed and self.axis == axis:
                # The same key again, as when locking a move to an axis: the space it is in swaps.
                other = "LOCAL" if self.start_space != "LOCAL" else "WORLD"
                self.space = other if self.space == self.start_space else self.start_space
            else:
                self.axis, self.space, self.pressed = axis, self.start_space, True
            self.last_plane = None  # measured again, about the new axis, from the next move
            self._apply(context)
            context.area.tag_redraw()
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._leave(context)
            return {"FINISHED"}
        elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            obj = self._object()
            if obj:
                restore(obj, self.saved)
            self._leave(context)
            return {"CANCELLED"}
        elif event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM"}:
            return {"PASS_THROUGH"}  # the view can still be moved around
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        obj = self._object()
        if obj:
            restore(obj, self.saved)
        self._leave(context)


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
            elif chain.joint_pinned:
                chain.joint_pos = obj.data.bones[chain.bone_1].head_local  # the joint at rest
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
            (LIVEIK_OT_reset_heel_key, "E", {"ctrl": True, "shift": True}),
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
