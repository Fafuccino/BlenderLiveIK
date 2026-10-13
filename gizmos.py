import bpy
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector, geometry

from . import heel, keying
from .properties import pose_anchor

AXIS_COLORS = ((0.9, 0.25, 0.3), (0.5, 0.8, 0.2), (0.3, 0.5, 0.95))  # X, Y, Z like the Move tool
ARROW_SCALE = 0.75
DIAL_SCALE = 0.9
DIAL_PROPS = ("heel_pitch", "heel_roll", "heel_yaw")  # about the frame's X, Y, Z

# The small square between two arrows, in the gizmo's unit space (an arrow is 1 long).
_NEAR, _FAR = 0.3, 0.6
_PLANE_VERTS = (
    (_NEAR, _NEAR, 0), (_FAR, _NEAR, 0), (_FAR, _FAR, 0),
    (_NEAR, _NEAR, 0), (_FAR, _FAR, 0), (_NEAR, _FAR, 0),
)  # fmt: skip


def _chain(obj_name, index):
    """Looked up each time so the handlers survive collection resizes."""
    try:
        obj = bpy.data.objects[obj_name]
        return obj, obj.live_ik.chains[index]
    except (KeyError, IndexError):
        return None, None


def _anchor(obj, chain, prop):
    """Armature-space matrix the stored position is relative to."""
    if prop == "heel_pos":
        return heel.end_target(obj, chain)
    return pose_anchor(obj, chain, prop)


class _Mover:
    """Moves one point (pole, heel pivot or IK foot) by a displacement since the drag began."""

    def __init__(self, obj_name, index, kind):
        self.obj_name = obj_name
        self.index = index
        self.kind = kind  # "pole_pos", "heel_pos" or "end" (the foot or hand)
        self.start = None
        self.axes = Matrix.Identity(3)

    def object_and_chain(self):
        return _chain(self.obj_name, self.index)

    def current_axes(self):
        """Axes of the handle in armature space (columns of a 3x3 matrix): see heel.handle_axes."""
        obj, chain = self.object_and_chain()
        return heel.handle_axes(obj, chain, self.kind) if chain else Matrix.Identity(3)

    def position(self):
        """Armature-space position of the handle."""
        obj, chain = self.object_and_chain()
        if not chain:
            return Vector()
        if self.kind == "end":
            return heel.end_target(obj, chain).translation
        return _anchor(obj, chain, self.kind) @ Vector(getattr(chain, self.kind))

    def begin(self):
        if self.start is not None:
            return
        obj, chain = self.object_and_chain()
        if not chain:
            return
        self.axes = self.current_axes()
        if self.kind == "end":
            self.start = heel.end_target(obj, chain)
        else:
            self.start = Vector(getattr(chain, self.kind))

    def end(self):
        self.start = None

    def move_world(self, delta_world):
        self.begin()
        obj, chain = self.object_and_chain()
        if not chain or self.start is None:
            return
        delta = obj.matrix_world.to_3x3().inverted() @ delta_world  # to armature space
        if self.kind == "end":
            heel.move_end(obj, chain, Matrix.Translation(delta) @ self.start)
        else:
            frame = _anchor(obj, chain, self.kind).to_3x3()
            setattr(chain, self.kind, self.start + frame.inverted() @ delta)


def _world_rotation(obj):
    return obj.matrix_world.to_quaternion().to_matrix()


def _arrow_accessors(mover, axis):
    """Movement along one axis. The offset is the distance travelled since the drag started."""

    def get():
        return 0.0

    def set_(distance):
        mover.begin()
        obj, _ = mover.object_and_chain()
        if obj:
            mover.move_world((_world_rotation(obj) @ mover.axes.col[axis]) * distance)

    return get, set_


def _dial_accessors(obj_name, index, prop):
    # The dials turn the opposite way to the stored angles, so the drag matches the foot.
    def get():
        _, chain = _chain(obj_name, index)
        return -float(getattr(chain, prop)) if chain else 0.0

    def set_(value):
        _, chain = _chain(obj_name, index)
        if chain:
            setattr(chain, prop, -value)

    return get, set_


class LIVEIK_GT_plane(bpy.types.Gizmo):
    """Moves a point on the plane of two axes, like the plane handles of the Move tool."""

    bl_idname = "LIVEIK_GT_plane"
    __slots__ = ("shape", "mover", "start_hit")

    def setup(self):
        if not hasattr(self, "shape"):
            self.shape = self.new_custom_shape("TRIS", _PLANE_VERTS)
            self.mover = None
            self.start_hit = None

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def _hit(self, context, event):
        """Where the mouse ray meets the plane, in world space."""
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        normal = self.matrix_basis.col[2].to_3d().normalized()
        if abs(direction.dot(normal)) < 0.05:
            return None  # plane seen edge-on: the mouse says nothing about the position on it
        return geometry.intersect_line_plane(origin, origin + direction, self.matrix_basis.translation, normal)

    def invoke(self, context, event):
        self.start_hit = self._hit(context, event)
        if self.start_hit is None or self.mover is None:
            return {"CANCELLED"}
        self.mover.begin()
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        hit = self._hit(context, event)
        if hit is not None:
            self.mover.move_world(hit - self.start_hit)
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        if self.mover:
            self.mover.end()
        self.start_hit = None


def _push_undo():
    keying.flush_pending()  # auto keys belong to the same undo step as the move
    bpy.ops.ed.undo_push(message="LiveIK Gizmo")
    return None


class LIVEIK_GGT_handles(bpy.types.GizmoGroup):
    bl_idname = "LIVEIK_GGT_handles"
    bl_label = "LiveIK Pole / Heel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj and obj.type == "ARMATURE" and context.window_manager.live_ik_gizmo_mode in {"SETUP", "MOVE"}

    def _style(self, gz, color, alpha=0.8):
        gz.color = gz.color_highlight = color
        gz.alpha = alpha
        gz.alpha_highlight = 1.0

    def _add_position_handle(self, obj, index, kind):
        """Three arrows and three planes that move a point along its own axes."""
        for axis in range(3):
            mover = _Mover(obj.name, index, kind)
            arrow = self.gizmos.new("GIZMO_GT_arrow_3d")
            arrow.draw_style = "NORMAL"
            arrow.scale_basis = ARROW_SCALE
            arrow.line_width = 2
            arrow.use_draw_modal = True
            self._style(arrow, AXIS_COLORS[axis])
            get, set_ = _arrow_accessors(mover, axis)
            arrow.target_set_handler("offset", get=get, set=set_)
            self._handles.append(("arrow", arrow, mover, axis))

            # The plane is named after the axis it is perpendicular to, like Blender's.
            mover = _Mover(obj.name, index, kind)
            plane = self.gizmos.new(LIVEIK_GT_plane.bl_idname)
            plane.scale_basis = ARROW_SCALE
            plane.use_draw_modal = True
            plane.mover = mover
            self._style(plane, AXIS_COLORS[axis], alpha=0.5)
            self._handles.append(("plane", plane, mover, axis))

    def _add_dials(self, obj, index):
        for axis, prop in enumerate(DIAL_PROPS):
            dial = self.gizmos.new("GIZMO_GT_dial_3d")
            dial.scale_basis = DIAL_SCALE
            dial.line_width = 3
            dial.use_draw_modal = True
            self._style(dial, AXIS_COLORS[axis])
            get, set_ = _dial_accessors(obj.name, index, prop)
            dial.target_set_handler("offset", get=get, set=set_)
            self._handles.append(("dial", dial, (obj.name, index), axis))

    def setup(self, context):
        self._handles = []
        self._built = None
        self._dragging = False

    def _mode(self):
        return bpy.context.window_manager.live_ik_gizmo_mode

    def _show_heels(self):
        return bpy.context.window_manager.live_ik_show_heels

    def _key(self, obj):
        chains = tuple((c.chain_type, c.visible) for c in obj.live_ik.chains)
        return (obj.name, self._mode(), self._show_heels(), chains)

    def refresh(self, context):
        obj = context.object
        if not obj:
            return
        # Blender calls refresh() on every property notification, including the ones
        # fired while a handle is being dragged. Clearing then would free the active gizmo.
        if self._built == self._key(obj):
            return
        self.gizmos.clear()
        self._handles = []
        mode = self._mode()
        for i, chain in enumerate(obj.live_ik.chains):
            if not chain.visible:
                continue
            leg = chain.chain_type == "LEG"
            if mode == "SETUP" and leg:
                self._add_position_handle(obj, i, "heel_pos")
            elif mode == "MOVE":
                self._add_position_handle(obj, i, "end")
                self._add_position_handle(obj, i, "pole_pos")
                if leg and self._show_heels():
                    self._add_dials(obj, i)
        self._built = self._key(obj)

    def draw_prepare(self, context):
        obj = context.object
        if not obj:
            return
        if self._built != self._key(obj):
            self.refresh(context)

        # Gizmo drags change data without an undo step. Push one when the drag ends, so
        # undo reverts the move itself. (From a timer: operators cannot run while drawing.)
        dragging = any(handle[1].is_modal for handle in self._handles)
        if self._dragging and not dragging:
            bpy.app.timers.register(_push_undo, first_interval=0.0)
        self._dragging = dragging

        world = obj.matrix_world
        world_rot = _world_rotation(obj)
        dial_turning = any(kind == "dial" and gz.is_modal for kind, gz, _, _ in self._handles)
        for kind, gz, target, axis in self._handles:
            if kind == "dial":
                # While a dial is turned, only that one is shown, and it holds still. A dial
                # reads its angle from the mouse relative to its own position and orientation.
                # The foot moves while it is turned (the toes are planted, the frame follows
                # its direction), so a dial that followed it would feed that movement back
                # into the angle it reads, and the foot would keep going.
                gz.hide = dial_turning and not gz.is_modal
                if dial_turning:
                    continue
                _, chain = _chain(*target)
                state = heel.heel_frame(obj, chain) if chain else None
                if not state:
                    continue
                position, direction = state[0], state[1].col[axis]
                rot = (world_rot @ direction).to_track_quat("Z", "Y").to_matrix().to_4x4()
            else:
                mover = target
                if not gz.is_modal:
                    mover.end()  # the drag is over, whatever ended it
                if mover.start is None:
                    mover.axes = mover.current_axes()
                position = mover.position()
                axes = world_rot @ mover.axes
                if kind == "arrow":
                    rot = axes.col[axis].to_track_quat("Z", "Y").to_matrix().to_4x4()
                else:
                    # Local X and Y span the plane, Z is its normal: the axis it is named after.
                    u, v = axes.col[(axis + 1) % 3], axes.col[(axis + 2) % 3]
                    rot = Matrix((u, v, axes.col[axis])).transposed().to_4x4()
            gz.matrix_basis = Matrix.Translation(world @ position) @ rot


classes = (LIVEIK_GT_plane, LIVEIK_GGT_handles)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
