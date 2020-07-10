from utils.mjcf_utils import new_joint, array_to_string
from models.task import Task


class GraspingTask(Task):
    """
    Creates MJCF model of a grasping task.

    A gear assembly task consists of a robot picking up a cube from a table. This class combines
    the robot, the arena with table, and the objects into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)


    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.table_body = mujoco_arena.table_body
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = {}  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free", damping="0.0005"))
            self.objects[obj_name] = obj
            self.worldbody.append(obj)
