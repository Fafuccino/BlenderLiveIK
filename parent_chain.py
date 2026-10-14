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


class LIVEIK_OT_parent_chain_transform(bpy.types.Operator):
    bl_idname = "live_ik.parent_chain_transform"
    bl_label = "Transform Parent Chain"
    bl_description = (
        "With a Parent Chain on, move, rotate or scale a bone of the chain together with the bones after it. "
        "X / Y / Z constrain (again for local, with Shift for the plane), Shift is for precision"
    )
    bl_options = {"UNDO", "BLOCKING", "GRAB_CURSOR"}

    mode: EnumProperty(
        items=[("TRANSLATE", "Move", ""), ("ROTATE", "Rotate", ""), ("SCALE", "Scale", "")],
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        return bool(driven(context))

    def _object(self):
        return bpy.data.objects.get(self.obj_name)

    def _mouse(self, event):
        return Vector((event.mouse_region_x, event.mouse_region_y))

    def _mouse_location(self, context, event):
        """The mouse in space, at the depth of the pivot."""
        coord = (event.mouse_region_x, event.mouse_region_y)
        return view3d_utils.region_2d_to_location_3d(context.region, context.region_data, coord, self.depth)

    def invoke(self, context, event):
        found = driven(context)
        if not found:
            return {"PASS_THROUGH"}
        names, selected = found
        obj = context.object
        bpy.context.view_layer.update()  # the pose bone matrices must be current
        bones = obj.pose.bones

        # The bones from the first selected one on move; the ones before it stay.
        first = min(i for i, name in enumerate(names) if name in selected)
        follow = names[first:]
        tail_parent = bones[names[-1]].parent
        if tail_parent and tail_parent.name not in names[:first]:
            follow.append(tail_parent.name)
        self.moved = list(dict.fromkeys(follow + [name for name in selected if name not in names]))
        self.pinned = names[:first]

        self.obj_name = obj.name
        self.matrices = {pose_bone.name: pose_bone.matrix.copy() for pose_bone in bones}
        self.bases = {name: bones[name].matrix_basis.copy() for name in self.moved + self.pinned}

        # The chains with IK on follow the transform: each is asked to reach where its end bone
        # was (armature space), moved by whatever happens to that bone.
        self.goals = {}
        if heel.solving_enabled():
            for index, chain in enumerate(obj.live_ik.chains):
                if chain.ik_enabled and all(name in bones for name in (chain.bone_0, chain.bone_1, chain.bone_2)):
                    self.goals[index] = heel.end_target(obj, chain)
        self.saved = grab.snapshot(obj, [(obj.live_ik.chains[index], grab.END) for index in self.goals])

        world = obj.matrix_world
        heads = [self.matrices[name].translation for name in names if name in selected]
        pivot_mode = context.scene.tool_settings.transform_pivot_point
        if pivot_mode == "CURSOR":
            self.pivot = context.scene.cursor.location.copy()
        elif pivot_mode == "ACTIVE_ELEMENT" and context.active_pose_bone:
            self.pivot = world @ self.matrices[context.active_pose_bone.name].translation
        elif first > 0:
            # The bone before the first one moved is its parent in the chain: it stays, and the
            # rest turns about the joint where they meet (that bone's head).
            self.pivot = world @ self.matrices[names[first - 1]].translation
        else:
            self.pivot = world @ (sum(heads, Vector()) / len(heads))
        self.depth = self.pivot
        # The axes of the first bone that is moved, for a local constraint.
        self.frame = (world.to_quaternion().to_matrix() @ self.matrices[names[first]].to_3x3().normalized())
        viewer = context.region_data.view_matrix.inverted().translation
        self.toward = (viewer - self.pivot).normalized()

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
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw_axes, (), "WINDOW", "POST_VIEW")
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

    def _transform(self):
        """The change, as a world-space matrix."""
        direction = self._direction()
        if self.mode == "TRANSLATE":
            return Matrix.Translation(grab.constrain(self.offset, self.axis, self.plane, direction))
        about = Matrix.Translation(self.pivot)
        if self.mode == "ROTATE":
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
        change = world.inverted() @ self._transform() @ world
        desired = {name: change @ self.matrices[name] for name in self.moved}
        desired.update({name: self.matrices[name] for name in self.pinned})
        matrix_of = place(obj, desired, self.matrices)
        if self.goals:
            bpy.context.view_layer.update()  # the solver reads the bones the chains hang from
            self._solve(obj, matrix_of)
        self._status(context)

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
        elif self.mode == "ROTATE":
            value = f"{math.degrees(self.angle):+.1f} deg"
        else:
            value = f"x{self.factor:.3f}"
        context.workspace.status_text_set(
            f"{_WORDS[self.mode]} parent chain  {value}{where}   "
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

    def _finish(self, context):
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
            elif here.length > 2.0 and self.last.length > 2.0:
                if self.mode == "ROTATE":
                    self.angle += grab.screen_angle(self.last, here) * scale
                else:
                    self.factor *= (here.length / self.last.length) ** scale
            self.last = here
            self._apply(context)
        elif event.type in {"X", "Y", "Z"} and event.value == "PRESS":
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


def register():
    bpy.utils.register_class(LIVEIK_OT_parent_chain_transform)
    config = bpy.context.window_manager.keyconfigs.addon
    if config:  # there is none when Blender runs without a window
        keymap = config.keymaps.new(name="Pose", space_type="EMPTY")
        for key, mode in (("G", "TRANSLATE"), ("R", "ROTATE"), ("S", "SCALE")):
            item = keymap.keymap_items.new(LIVEIK_OT_parent_chain_transform.bl_idname, key, "PRESS")
            item.properties.mode = mode
            _KEYMAPS.append((keymap, item))


def unregister():
    for keymap, item in _KEYMAPS:
        keymap.keymap_items.remove(item)
    _KEYMAPS.clear()
    bpy.utils.unregister_class(LIVEIK_OT_parent_chain_transform)
