"""Parent Chain: G, R and S on a list of bones treated as if each were the child of the one before.

For the list [Hand, Forearm, Arm]: moving the hand carries the forearm, the arm and the arm's
own parent with it, as if the forearm were the hand's child, the arm the forearm's child and the
arm's parent the arm's child. Moving the forearm carries the arm and its parent; the hand stays
where it is. The bones before the one being moved are held in place.

Only the Parent Chain that is switched on does this, and only when one of its bones is among the
selected ones; for any other selection G, R and S are Blender's own.

The bones are placed in armature space and each is given the pose that puts it there, so the
real hierarchy stays what it is: only the bones that have to move are posed, and the ones that
hang from them in the rig follow. A bone that is connected to a bone that moves cannot stay
behind: Blender keeps its head on the parent's tail.

IK chains with IK on follow along. Their end (foot, hand) keeps its IK target when the transform
leaves it where it is, so the limb bends to keep it planted while the bones above it move (a
hand holding a ledge while the body is moved); when the transform moves the end, its target goes
with it. The solver then poses the chain's bones, whatever the transform did to them.

The same keys also act on the bones above a chain whose joint is pinned (see above_pins): the
bones move as Blender would move them, and the bone the chain hangs from (the shoulder) is turned
so the chain's first bone stays one upper bone away from the pin. The update handler then solves
the chain, and the joint stays where it was pinned.
"""

import math

import bpy
import gpu
from bpy.props import EnumProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from . import grab, heel, keying
from .properties import _unscaled

_KEYMAPS = []

_AXIS_COLORS = grab._AXIS_COLORS
_LINE_LENGTH = grab._LINE_LENGTH

_WORDS = {"TRANSLATE": "Move", "ROTATE": "Rotate", "SCALE": "Scale"}


def active_chain(obj):
    """The Parent Chain that is switched on, or None."""
    return next((chain for chain in obj.live_ik.parent_chains if chain.active), None)


def chain_names(obj, chain):
    """The chain's bone names in order, as far as each is the real parent of the one before.

    The list is read from the first bone until a name is empty, unknown or not the parent of the
    previous bone: what comes after is left out, so the transforms stay a walk up the rig.
    """
    names = []
    previous = None
    for item in chain.bones:
        bone = obj.pose.bones.get(item.bone)
        if bone is None or (previous is not None and previous.parent != bone):
            break
        names.append(bone.name)
        previous = bone
    return names


def broken_at(obj, chain):
    """Index of the first listed bone that is left out of the chain, or None when all are used."""
    used = len(chain_names(obj, chain))
    return used if used < len(chain.bones) else None


def driven(context):
    """(bone names of the active Parent Chain, names of the selected bones), or None.

    None unless Pose Mode has an active Parent Chain and at least one of its bones selected.
    """
    obj = context.object
    if not obj or obj.type != "ARMATURE" or context.mode != "POSE" or not obj.pose:
        return None
    chain = active_chain(obj)
    if not chain:
        return None
    names = chain_names(obj, chain)
    selected = {pose_bone.name for pose_bone in context.selected_pose_bones or ()}
    return (names, selected) if selected.intersection(names) else None


def place(obj, desired, old):
    """Pose the bones in `desired` ({name: armature-space matrix}) so each lies where it says.

    `old` holds every bone's matrix from before. A bone that is not in `desired` keeps its pose
    and goes where its parent takes it, which is what tells a bone its parent has moved.
    Returns a function giving the matrix any bone ends up with, by name.
    """
    bones = obj.pose.bones
    moved = {}

    def matrix_of(name):
        if name in desired:
            return desired[name]
        if name not in moved:
            parent = bones[name].parent
            moved[name] = old[name] if parent is None else matrix_of(parent.name) @ old[parent.name].inverted() @ old[name]
        return moved[name]

    for name, matrix in desired.items():
        pose_bone = bones[name]
        parent = matrix_of(pose_bone.parent.name) if pose_bone.parent else Matrix.Identity(4)
        heel._set_basis(pose_bone, parent, matrix)
    return matrix_of


class _TransformOperator(bpy.types.Operator):
    """The modal G / R / S the operators below share. Not registered itself: registering a class
    that is derived from a registered one breaks the base's Python class in Blender."""

    bl_options = {"UNDO", "BLOCKING", "GRAB_CURSOR"}

    individual_ok = False  # may the scene's Individual Origins pivot be followed (see _plan)

    mode: EnumProperty(
        items=[("TRANSLATE", "Move", ""), ("ROTATE", "Rotate", ""), ("SCALE", "Scale", "")],
        options={"SKIP_SAVE"},
    )

    def _object(self):
        return bpy.data.objects.get(self.obj_name)

    def _mouse(self, event):
        return Vector((event.mouse_region_x, event.mouse_region_y))

    def _mouse_location(self, context, event):
        """The mouse in space, at the depth of the pivot."""
        coord = (event.mouse_region_x, event.mouse_region_y)
        return view3d_utils.region_2d_to_location_3d(context.region, context.region_data, coord, self.depth)

    def _plan(self, context):
        """What the transform acts on, or None to leave the key to Blender.

        Returns (moved, pinned, selected, frame, joint): the bones that are transformed, the bones
        held in place, the selected ones (their median is the pivot), the bone whose axes a local
        constraint follows, and the bone whose head is the pivot instead (None for the median).
        """
        found = driven(context)
        if not found:
            return None
        names, selected = found
        bones = context.object.pose.bones
        # The bones from the first selected one on move; the ones before it stay.
        first = min(i for i, name in enumerate(names) if name in selected)
        follow = names[first:]
        tail_parent = bones[names[-1]].parent
        if tail_parent and tail_parent.name not in names[:first]:
            follow.append(tail_parent.name)
        moved = list(dict.fromkeys(follow + [name for name in selected if name not in names]))
        # For Individual Origins: which of the moved bones are selected, and which of those are
        # not part of the chain (they are turned on their own and carry nothing).
        self.chain_selected = {name for name in names if name in selected}
        self.extra_selected = {name for name in selected if name not in names}
        self.joints = {names[i]: names[i - 1] for i in range(1, len(names))}  # bone -> the bone before it
        # The bone before the first one moved is its parent in the chain: it stays, and the
        # rest turns about the joint where they meet (that bone's head).
        return moved, names[:first], [n for n in names if n in selected], names[first], names[first - 1] if first else None

    def _chains(self, obj):
        """The IK chains that follow the transform, as ({chain index: IK target}, [(chain, END)]).

        Each is asked to reach where its end bone was (armature space), moved by whatever
        happens to that bone."""
        goals = {}
        bones = obj.pose.bones
        if heel.solving_enabled():
            for index, chain in enumerate(obj.live_ik.chains):
                if chain.ik_enabled and all(name in bones for name in (chain.bone_0, chain.bone_1, chain.bone_2)):
                    goals[index] = heel.end_target(obj, chain)
        return goals, [(obj.live_ik.chains[index], grab.END) for index in goals]

    def _adjust(self, change):
        """The change the mouse asks for, as the transform allows it (armature space), and the
        matrices of other bones that have to be posed along with it ({name: matrix})."""
        return change, {}

    def invoke(self, context, event):
        plan = self._plan(context)
        if not plan:
            return {"PASS_THROUGH"}
        self.moved, self.pinned, selected, frame, joint = plan
        obj = context.object
        bpy.context.view_layer.update()  # the pose bone matrices must be current
        bones = obj.pose.bones

        self.obj_name = obj.name
        self.matrices = {pose_bone.name: pose_bone.matrix.copy() for pose_bone in bones}
        self.bases = {name: bones[name].matrix_basis.copy() for name in self.moved + self.pinned}
        self.goals, snapshot = self._chains(obj)
        self.saved = grab.snapshot(obj, snapshot)

        world = obj.matrix_world
        heads = [self.matrices[name].translation for name in selected]
        pivot_mode = context.scene.tool_settings.transform_pivot_point
        self.individual = self.individual_ok and pivot_mode == "INDIVIDUAL_ORIGINS"
        if pivot_mode == "CURSOR":
            self.pivot = context.scene.cursor.location.copy()
        elif pivot_mode == "ACTIVE_ELEMENT" and context.active_pose_bone:
            self.pivot = world @ self.matrices[context.active_pose_bone.name].translation
        elif joint:
            self.pivot = world @ self.matrices[joint].translation
        else:
            self.pivot = world @ (sum(heads, Vector()) / len(heads))
        self.depth = self.pivot
        # The axes of the first bone that is moved, for a local constraint.
        self.frame = (world.to_quaternion().to_matrix() @ self.matrices[frame].to_3x3().normalized())
        # Which way the view looks back at us: R turns about this axis whatever the pivot's place
        # on screen, as Blender's does (the direction to the camera would differ in perspective).
        self.toward = context.region_data.view_matrix.inverted().to_3x3().col[2].normalized()

        center = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, self.pivot)
        self.center2d = center if center is not None else self._mouse(event)
        self.last = self._mouse(event) - self.center2d
        self.last_location = self._mouse_location(context, event)
        self.offset = Vector((0.0, 0.0, 0.0))  # the movement so far, in world space
        self.angle = 0.0
        self.factor = 1.0
        self.axis = None
        self.plane = False
        self.local = False
        self.trackball = False  # R again while rotating: the mouse turns it about the view's own axes
        self.virtual = Vector((0.0, 0.0))  # how far the mouse has gone since the trackball began, in pixels
        view = context.region_data.view_matrix.inverted().to_3x3()
        self.right, self.up = view.col[0].normalized(), view.col[1].normalized()
        self.mouse = self._mouse(event)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw_axes, (), "WINDOW", "POST_VIEW")
        self._cursor = bpy.types.SpaceView3D.draw_handler_add(self._draw_cursor, (), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _direction(self):
        """The axis of the constraint, in world space, or None."""
        if self.axis is None:
            return None
        if self.local:
            return self.frame.col[self.axis].normalized()
        return Vector(tuple(1.0 if i == self.axis else 0.0 for i in range(3)))

    def _trackball(self):
        """The trackball rotation (see grab.trackball_rotation)."""
        return grab.trackball_rotation(self.right, self.up, self.virtual)

    def _transform(self, pivot=None):
        """The change, as a world-space matrix, about the pivot (the one chosen, unless given)."""
        direction = self._direction()
        if self.mode == "TRANSLATE":
            return Matrix.Translation(grab.constrain(self.offset, self.axis, self.plane, direction))
        about = Matrix.Translation(self.pivot if pivot is None else pivot)
        if self.mode == "ROTATE" and self.trackball:
            change = self._trackball().to_matrix().to_4x4()
        elif self.mode == "ROTATE":
            if direction is None:
                axis, sign = self.toward, 1  # about the view, like Blender's R
            else:
                axis, sign = direction, (1 if direction.dot(self.toward) >= 0 else -1)
            change = Matrix.Rotation(self.angle * sign, 4, axis)
        elif direction is None:
            change = Matrix.Scale(self.factor, 4)
        else:
            outer = Matrix([[direction[i] * direction[j] for j in range(3)] for i in range(3)])
            along = outer if not self.plane else Matrix.Identity(3) - outer
            change = (Matrix.Identity(3) + (self.factor - 1.0) * along).to_4x4()
        return about @ change @ about.inverted()

    def _apply(self, context):
        obj = self._object()
        if not obj:
            return
        world = obj.matrix_world
        if self.individual and self.mode != "TRANSLATE":
            desired, extra = self._individual(world), {}
        else:
            change, extra = self._adjust(world.inverted() @ self._transform() @ world)
            desired = {name: change @ self.matrices[name] for name in self.moved}
        desired.update({name: self.matrices[name] for name in self.pinned})
        desired.update(extra)
        matrix_of = place(obj, desired, self.matrices)
        if self.goals:
            bpy.context.view_layer.update()  # the solver reads the bones the chains hang from
            self._solve(obj, matrix_of)
        self._status(context)

    def _individual(self, world):
        """The new matrices with Individual Origins: each selected bone turns or scales about its own head.

        Going down the list, a selected bone adds its own change, about where its head has got to,
        to what the bones before it already did, and carries the bones after it along: a run of
        selected bones adds up, as it does in Blender. A selected bone outside the chain is
        changed on its own."""
        inverse = world.inverted()
        desired = {}
        carried = Matrix.Identity(4)  # what the bones before this one did, in armature space
        for name in self.moved:
            head = self.matrices[name].translation
            if name in self.extra_selected:
                desired[name] = inverse @ self._transform(world @ head) @ world @ self.matrices[name]
                continue
            if name in self.chain_selected:
                # Its origin is the joint with the bone before it in the list (that bone's head,
                # where it has got to); the first bone of the list has only its own.
                joint = self.joints.get(name)
                origin = desired[joint].translation if joint in desired else self.matrices[joint or name].translation
                carried = inverse @ self._transform(world @ origin) @ world @ carried
            desired[name] = carried @ self.matrices[name]
        return desired

    def _solve(self, obj, matrix_of):
        """Bring the IK chains the transform touched to where their end (foot, hand) now is.

        An end that moved takes its IK target with it; an end that stayed (a pinned bone, or
        one the transform did not reach) keeps the target it had, and the solver bends the
        limb to keep it there while what the limb hangs from moves.
        """
        for index, goal in self.goals.items():
            chain = obj.live_ik.chains[index]
            thigh = obj.pose.bones[chain.bone_0]
            watched = [chain.bone_0, chain.bone_1, chain.bone_2]
            if thigh.parent:
                watched.append(thigh.parent.name)
            if all(heel._close(matrix_of(name), self.matrices[name]) for name in watched):
                continue
            delta = matrix_of(chain.bone_2) @ self.matrices[chain.bone_2].inverted()
            heel.move_end(obj, chain, _unscaled(delta @ goal))

    def _restore(self):
        obj = self._object()
        if obj:
            for name, basis in self.bases.items():
                obj.pose.bones[name].matrix_basis = basis
            grab.restore(obj, self.saved)  # the IK chains' poses and what the solver remembered

    def _status(self, context):
        if self.axis is None:
            where = ""
        else:
            where = f"  [{'Local ' if self.local else ''}{'not ' if self.plane else ''}{'XYZ'[self.axis]}]"
        if self.mode == "TRANSLATE":
            d = self.offset
            value = f"{d.x:+.3f} {d.y:+.3f} {d.z:+.3f}"
        elif self.mode == "ROTATE" and self.trackball:
            value = "trackball"
        elif self.mode == "ROTATE":
            value = f"{math.degrees(self.angle):+.1f} deg"
        else:
            value = f"x{self.factor:.3f}"
        trackball = "R: trackball   " if self.mode == "ROTATE" else ""
        context.workspace.status_text_set(
            f"{_WORDS[self.mode]} parent chain  {value}{where}   {trackball}"
            "X/Y/Z: constrain (again: local)   Shift+X/Y/Z: plane   Shift: precision   "
            "LMB / Enter: confirm   Esc / RMB: cancel"
        )

    def _draw_axes(self):
        """The line (or the two lines of the plane) the change is kept to, through the pivot."""
        if self.axis is None:
            return
        shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 2.0)
        plane = self.plane and self.mode != "ROTATE"
        axes = [a for a in range(3) if a != self.axis] if plane else [self.axis]
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        for axis in axes:
            if self.local:
                direction = self.frame.col[axis].normalized()
            else:
                direction = Vector(tuple(1.0 if i == axis else 0.0 for i in range(3)))
            points = [self.pivot - direction * _LINE_LENGTH, self.pivot + direction * _LINE_LENGTH]
            shader.uniform_float("color", _AXIS_COLORS[axis])
            batch_for_shader(shader, "LINES", {"pos": points}).draw(shader)
        gpu.state.blend_set("NONE")

    def _set_trackball(self, context, on):
        """Switch the trackball on or off, each time from the start. Blender hides the cursor
        while it is on and draws its four arrows there instead."""
        self.trackball = on
        self.angle = 0.0
        self.virtual = Vector((0.0, 0.0))
        if context.window:
            if on:
                context.window.cursor_modal_set("NONE")
            else:
                context.window.cursor_modal_restore()

    def _draw_cursor(self):
        """The dashed line from the pivot to the mouse, with the arrow at the mouse that says
        which way it is read: across the line to rotate, along it to scale (like Blender's)."""
        if self.mode == "ROTATE" and self.trackball:
            grab.draw_trackball_cursor(self.mouse)
            return
        if self.mode == "TRANSLATE":
            return
        start, end = self.center2d, self.mouse
        along = end - start
        if along.length < 1.0:
            return
        along = along / along.length
        segments = []
        t = 0.0
        length = (end - start).length
        while t < length:  # the dashes
            segments += [start + along * t, start + along * min(t + 6.0, length)]
            t += 11.0
        direction = Vector((-along.y, along.x)) if self.mode == "ROTATE" else along
        arrow, head = 12.0, 5.0
        for sign in (1.0, -1.0):  # the two heads of the arrow
            tip = end + direction * arrow * sign
            segments += [end, tip]
            for side in (1.0, -1.0):
                back = Vector((-direction.y, direction.x)) * side * head * 0.7 - direction * sign * head
                segments += [tip, tip + back]
        shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 1.5)
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
        gpu.state.blend_set("ALPHA")
        points = [(point.x, point.y, 0.0) for point in segments]
        batch_for_shader(shader, "LINES", {"pos": points}).draw(shader)
        gpu.state.blend_set("NONE")

    def _finish(self, context):
        if self.trackball and context.window:
            context.window.cursor_modal_restore()
        if getattr(self, "_cursor", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._cursor, "WINDOW")
            self._cursor = None
        if getattr(self, "_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            self._handle = None
        if context.area:
            context.area.tag_redraw()
        context.workspace.status_text_set(None)

    def _key(self):
        obj = self._object()
        if obj and keying.autokey_enabled():
            frame = bpy.context.scene.frame_current
            for name in self.bases:
                keying.key_bone(obj, name, frame)

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            scale = 0.1 if event.shift else 1.0
            here = self._mouse(event) - self.center2d
            if self.mode == "TRANSLATE":
                location = self._mouse_location(context, event)
                self.offset += (location - self.last_location) * scale
                self.last_location = location
            elif self.mode == "ROTATE" and self.trackball:
                # What the mouse moved this time (the window's own steps, which stay right when the
                # cursor wraps at the edge), added up; Shift slows it to a thirtieth, like Blender.
                step = Vector((event.mouse_x - event.mouse_prev_x, event.mouse_y - event.mouse_prev_y))
                self.virtual += step * (1.0 / 30.0 if event.shift else 1.0)
            elif here.length > 2.0 and self.last.length > 2.0:
                if self.mode == "ROTATE":
                    self.angle += grab.screen_angle(self.last, here) * scale
                else:
                    self.factor *= (here.length / self.last.length) ** scale
            self.last = here
            self.mouse = self._mouse(event)
            self._apply(context)
            context.area.tag_redraw()
        elif event.type == "R" and event.value == "PRESS" and not event.is_repeat and self.mode == "ROTATE":
            # Back and forth between turning about the view and the trackball, each from the start.
            self._set_trackball(context, not self.trackball)
            self.axis, self.local = None, False
            self._apply(context)
            context.area.tag_redraw()
        elif event.type in {"X", "Y", "Z"} and event.value == "PRESS":
            if self.trackball:
                self._set_trackball(context, False)  # an axis: back to turning about it
            axis = "XYZ".index(event.type)
            plane = event.shift and self.mode != "ROTATE"
            if self.axis == axis and self.plane == plane:
                if self.local:
                    self.axis, self.local = None, False  # a third press clears it
                else:
                    self.local = True
            else:
                self.axis, self.plane, self.local = axis, plane, False
            self._apply(context)
            context.area.tag_redraw()
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._key()
            self._finish(context)
            return {"FINISHED"}
        elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            self._restore()
            self._finish(context)
            return {"CANCELLED"}
        elif event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM"}:
            return {"PASS_THROUGH"}  # the view can still be moved around
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._restore()
        self._finish(context)


class LIVEIK_OT_parent_chain_transform(_TransformOperator):
    bl_idname = "live_ik.parent_chain_transform"
    bl_label = "Transform Parent Chain"
    bl_description = (
        "With a Parent Chain on, move, rotate or scale a bone of the chain together with the bones after it. "
        "X / Y / Z constrain (again for local, with Shift for the plane), Shift is for precision"
    )

    individual_ok = True

    @classmethod
    def poll(cls, context):
        return bool(driven(context))


def above_pins(context):
    """The IK chains with a pinned joint that every selected bone is above, or None.

    A bone is above a chain when it is an ancestor of the chain's first bone (the chest, the
    shoulder...). None when a Parent Chain has the selection, or any selected bone is not above
    a chain with IK on and a pinned joint: those keys are Blender's own.
    """
    obj = context.object
    if not obj or obj.type != "ARMATURE" or context.mode != "POSE" or not obj.pose or not heel.solving_enabled():
        return None
    selected = {pose_bone.name for pose_bone in context.selected_pose_bones or ()}
    if not selected or driven(context):
        return None
    bones = obj.pose.bones
    chains, above = [], set()
    for chain in obj.live_ik.chains:
        if chain.ik_enabled and chain.joint_pinned and chain.bone_0 in bones and chain.bone_1 in bones:
            chains.append(chain)
            parent = bones[chain.bone_0].parent
            while parent:
                above.add(parent.name)
                parent = parent.parent
    return chains if chains and selected <= above else None


class LIVEIK_OT_pin_transform(_TransformOperator):
    bl_idname = "live_ik.pin_transform"
    bl_label = "Transform Above Pinned Joint"
    bl_description = (
        "With the joint of an IK chain pinned, move, rotate or scale the bones it hangs from (the chest, "
        "the shoulder...) while the pinned joint stays: they can only swing about it. "
        "X / Y / Z constrain (again for local, with Shift for the plane), Shift is for precision"
    )

    @classmethod
    def poll(cls, context):
        return bool(above_pins(context))

    def _plan(self, context):
        if not above_pins(context):
            return None
        selected = [pose_bone.name for pose_bone in context.selected_pose_bones]
        active = context.active_pose_bone
        return selected, [], selected, active.name if active and active.name in selected else selected[0], None

    def _chains(self, obj):
        """The chains under the moved bones are solved by the update handler as the bones move,
        so they are only noted here, to be put back on a cancel."""
        bones = obj.pose.bones
        moved = set(self.moved)
        self.pins = []
        affected = []
        for chain in obj.live_ik.chains:
            if not (chain.ik_enabled and chain.joint_pinned and chain.bone_0 in bones and chain.bone_1 in bones):
                continue
            parent, hangs = bones[chain.bone_0].parent, False
            while parent and not hangs:
                hangs = parent.name in moved
                parent = parent.parent
            if hangs:
                affected.append((chain, grab.END))
                reach = (bones[chain.bone_1].bone.head_local - bones[chain.bone_0].bone.head_local).length
                # The bone the chain hangs from (the shoulder) is turned to keep the pin, unless it
                # is one of the bones being moved.
                holder = bones[chain.bone_0].parent.name
                if holder in moved:
                    holder = None
                else:
                    self.bases[holder] = bones[holder].matrix_basis.copy()
                self.pins.append(
                    (self.matrices[chain.bone_0].translation.copy(), Vector(chain.joint_pos), reach, holder)
                )
        return {}, affected

    def _adjust(self, change):
        """Keep the first bone of each chain one upper bone away from its pin.

        The joint is then exactly where it was put. The bone the chain hangs from (the shoulder)
        is turned about its head by the least angle that does it, so rotating the chest turns the
        shoulder with it and nothing slides. A chain with no such bone (the moved bone is the one
        it hangs from) is kept there by sliding the change instead, except for a rotation, which
        is left alone: a rotation must not move anything."""
        shift = Vector((0.0, 0.0, 0.0))
        if self.mode != "ROTATE":
            for _ in range(8):  # several chains pull the same bones: this settles on a compromise
                for root, joint, reach, holder in self.pins:
                    if holder is None:
                        here = change @ root + shift
                        away = here - joint
                        if away.length > 1e-6 and reach > 1e-6:
                            shift += joint + away * (reach / away.length) - here
            change = Matrix.Translation(shift) @ change

        extra = {}
        for root, joint, reach, holder in self.pins:
            if holder is None:
                continue
            matrix = change @ self.matrices[holder]
            head = matrix.translation
            arm = change @ root - head  # from the shoulder's head to where the chain starts
            toward = joint - head
            if arm.length < 1e-6 or toward.length < 1e-6 or reach < 1e-6:
                continue
            distance, radius = toward.length, arm.length
            axis = toward / distance
            if abs(radius - reach) <= distance <= radius + reach:
                # Where the sphere the chain can reach with the shoulder turned meets the one
                # about the pin: a circle, and the nearest point of it is the least turn.
                along = (radius * radius - reach * reach + distance * distance) / (2 * distance)
                around = max(radius * radius - along * along, 0.0) ** 0.5
                side = arm - axis * arm.dot(axis)
                side = side.normalized() if side.length > 1e-6 else axis.orthogonal().normalized()
                point = axis * along + side * around
            else:
                point = axis * radius  # out of reach: as near to the pin as it can get
            turn = arm.rotation_difference(point).to_matrix().to_4x4()
            extra[holder] = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ matrix
        return change, extra


_OPERATORS = (LIVEIK_OT_parent_chain_transform, LIVEIK_OT_pin_transform)


def register():
    for cls in _OPERATORS:
        bpy.utils.register_class(cls)
    config = bpy.context.window_manager.keyconfigs.addon
    if config:  # there is none when Blender runs without a window
        keymap = config.keymaps.new(name="Pose", space_type="EMPTY")
        for operator in _OPERATORS:
            for key, mode in (("G", "TRANSLATE"), ("R", "ROTATE"), ("S", "SCALE")):
                item = keymap.keymap_items.new(operator.bl_idname, key, "PRESS")
                item.properties.mode = mode
                _KEYMAPS.append((keymap, item))


def unregister():
    for keymap, item in _KEYMAPS:
        keymap.keymap_items.remove(item)
    _KEYMAPS.clear()
    for cls in reversed(_OPERATORS):
        bpy.utils.unregister_class(cls)
