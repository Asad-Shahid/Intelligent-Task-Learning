from collections import OrderedDict
import numpy as np
import utils.transform_utils as T
from environments import MujocoEnv
from models.gripper import PandaGripper
from models.robot import Panda
from environments.action_space import ActionSpace
from controller.arm_controller import *
from collections import deque
import hjson


class PandaEnv(MujocoEnv):
    """Initializes a Panda robot environment."""

    def __init__(self, config, controller_config_file):
        """
        Args:
            All configuratin parameters are defined in config file

            controller_config_file (str): filepath to the corresponding controller config file that contains the
            associated controller parameters
        """
        self.control_freq = config.control_freq
        self.has_gripper = config.gripper_type is not None
        self.gripper_visualization = config.gripper_visualization

        # Load the appropriate controller
        self._load_controller('joint_velocity', controller_config_file)
        super().__init__(config)

    def _load_controller(self, controller_type, controller_file):
        """
        Loads controller to be used for dynamic trajectories
        Controller_type is a specified controller, and controller_params is a config file containing the appropriate
        parameters for that controller
        """
        # Load the controller config file
        try:
            with open(controller_file) as f:
                params = hjson.load(f)
        except FileNotFoundError:
            print("Controller config file '{}' not found. Please check filepath and try again.".format(controller_file))

        controller_params = params[controller_type]
        self.controller = JointVelocityController(**controller_params)

    def _load_model(self):
        """
        Loads robot and add gripper.
        """
        super()._load_model()
        # Use xml that has motor torque actuators enabled
        self.mujoco_robot = Panda(xml_path="robot/panda/robot_torque.xml")

        if self.has_gripper:
            self.gripper = PandaGripper()
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("hand", self.gripper)

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos
        self.sim.data.qvel[self._ref_joint_vel_indexes] = np.zeros(len(self._ref_joint_vel_indexes))
        if self.has_gripper:
            self.sim.data.qpos[self._ref_gripper_joint_pos_indexes] = self.gripper.init_qpos
            self.sim.data.qvel[self._ref_gripper_joint_vel_indexes] = np.zeros(len(self._ref_gripper_joint_vel_indexes))


        self.controller.reset()
        self.total_joint_torque = 0
        self.joint_torques = 0
        self.ee_v = np.zeros(6)

    def _get_reference(self):
        """
        Sets up necessary reference for robot, gripper, and objects.
        """
        super()._get_reference()
        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints) # ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints] # [0, 1, 2, 3, 4, 5, 6]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints] # [0, 1, 2, 3, 4, 5, 6]

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints) # ['finger_joint1', 'finger_joint2']
            self._ref_gripper_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints] # [7, 8]
            self._ref_gripper_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints] # [7, 8]
            self._ref_joint_gripper_actuator_indexes = [self.sim.model.actuator_name2id(actuator) for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper")] # [7, 8]
            # IDs of sites for gripper visualization
            self.eef_site_id = self.sim.model.site_name2id("grip_site") # 5


    def _pre_action(self, action, policy_step):
        """
        Overrides the superclass method to actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired
                normalized joint velocities and the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
        """

        self.policy_step = policy_step

        # Make sure action length is correct
        assert len(action) == self.dof, "environment got invalid action dimension"

        # using controller
        # Split action into joint control and peripheral (i.e.: gripper) control
        gripper_action = [],
        if self.has_gripper:
            gripper_action = action[self.controller.control_dim:]  # all indexes past controller dimension indexes
            action = action[:self.controller.control_dim]

        action = action.copy()  # ensure that we don't change the action outside of this scope
        self.controller.update_model(self.sim, joint_index=self._ref_joint_pos_indexes)
        torques = self.controller.action_to_torques(action, self.policy_step)  # this scales and clips the actions correctly
        self.total_joint_torque += np.sum(abs(torques))
        self.joint_torques = torques

        # Get gripper action
        if self.has_gripper:
            gripper_action_actual = self.gripper.format_action(gripper_action) # mirrors the commanded action i-e [0.54] ==> [-0.54 0.54]

            # rescale normalized gripper action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange[self._ref_gripper_joint_vel_indexes] # e.g.[[0. 0.04] [-0.04 -0.]]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0]) # e.g.[0.02 -0.02]
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0]) # e.g.[0.02 0.02]
            applied_gripper_action = bias + weight * gripper_action_actual
            self.sim.data.ctrl[self._ref_gripper_joint_vel_indexes] = [applied_gripper_action] #[1, -1]

        # Now,control the joints, qfrc_bias accounts for gravity compensation and other internal forces (coirolis, centrifugal)
        self.sim.data.ctrl[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes] + torques

        # velocity of hand and joints to be used in reward
        self.ee_v = np.concatenate([self.sim.data.body_xvelp[self.sim.model.body_name2id("hand")],
                 self.sim.data.body_xvelr[self.sim.model.body_name2id("hand")]])
        self.q_vel = self.sim.data.qvel[self._ref_joint_vel_indexes]

    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        ret = super()._post_action(action)
        if self.gripper_visualization:
            self._gripper_visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing robot observations
        """

        state = super()._get_observation()
        di = OrderedDict()

        # proprioceptive features
        di["joint_pos"] = np.array([self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])
        di["joint_vel"] = np.array([self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])

        # 7 x 3 (sinq, cosq, qdot) = 21
        robot_states = [np.sin(di["joint_pos"]), np.cos(di["joint_pos"]), di["joint_vel"]]

        if self.has_gripper:
            di["gripper_qpos"] = np.array([self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes])
            di["gripper_qvel"] = np.array([self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes])

            di["eef_pos"] = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('hand')])
            di["eef_quat"] = T.convert_quat(self.sim.data.get_body_xquat("hand"), to="xyzw")
            di["eef_vlin"] = np.array(self.sim.data.get_body_xvelp('hand'))
            di["eef_vang"] = np.array(self.sim.data.get_body_xvelr('hand'))

            # add in gripper information (2(gripper) + 3(end-effector pos) + 4(end-effector rot) + 3(end-effector vel) + 3(end-effector w))
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"], di["eef_vlin"], di["eef_vang"]])

        state["robot-state"] = np.concatenate(robot_states)

        return state


    @property
    def observation_space(self):
        """
        Returns dict where keys are ob names and values are dimensions.
        """
        ob_space = OrderedDict()
        observation = self._get_observation()
        ob_space['object-state'] = [observation['object-state'].size]
        ob_space['robot-state'] =  [observation['robot-state'].size]
        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        dof = self.controller.action_dim

        if self.has_gripper:
            dof += self.gripper.dof
            return dof

    @property
    def action_space(self):
        """
        Returns ActionSpec of action space, see
        action_spec.py for more documentation.
        """
        return ActionSpace(self.dof)


    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 1., 0., 1.]


    def _check_q_limits(self):
        """
        Returns True if the arm is in joint limits or very close to.
        """
        joint_limits = False
        tolerance = 0.15
        for (q, q_limits) in zip(self.sim.data.qpos[self._ref_joint_pos_indexes], self.sim.model.jnt_range[:7]):
            if (q < q_limits[0] + tolerance) or (q > abs(q_limits[1]) - tolerance):
                joint_limits = True
        return joint_limits
