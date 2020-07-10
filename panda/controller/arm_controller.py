import numpy as np
from enum import Enum
import mujoco_py
from scipy.interpolate import CubicSpline


class ControllerType(str, Enum):
    JOINT_VEL = 'joint_velocity'


class Controller():
    def __init__(self, control_max, control_min, max_action, min_action, policy_freq=250, interpolation=None):

        # Upper and lower limits to the input action (only pos/ori)
        self.control_max = control_max # array([1, 1, 1, 1, 1, 1, 1])
        self.control_min = control_min # array([-1, -1, -1, -1, -1, -1, -1])

        # Dimensionality of the action
        self.control_dim = self.control_max.shape[0] # 7

        # Limits to the policy outputs
        self.input_max = max_action #  1
        self.input_min = min_action # -1

        # This handles when the mean of max and min control is not zero -> actions are around that mean
        self.action_scale = abs(self.control_max - self.control_min) / abs(max_action - min_action) # array([1., 1., 1., 1., 1., 1., 1.])
        self.action_output_transform = (self.control_max + self.control_min) / 2.0 # array([0., 0., 0., 0., 0., 0., 0.])
        self.action_input_transform = (max_action + min_action) / 2.0 # 0

        # Frequency at which actions from the robot policy are fed into this controller
        self.policy_freq = policy_freq
        self.interpolation = interpolation # "linear"

        self.ramp_ratio = 1  # Percentage of the time between policy timesteps used for interpolation

        # Initialize the remaining attributes
        self.model_timestep = None
        self.interpolation_steps = None
        self.current_joint_velocity = None

    def reset(self):
        """
        Resets the internal values of the controller
        """
        pass

    def transform_action(self, action):
        """
        Scale the action to go to the right min and max
        """
        action = np.clip(action, self.input_min, self.input_max) # -1, 1
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def update_model(self, sim, joint_index):
        """
        Updates the state of the robot used to compute the control command
        """
        self.sim_freq = 1 / sim.model.opt.timestep
        self.interpolation_steps = np.floor(self.ramp_ratio * self.sim_freq / self.policy_freq)
        self.current_joint_velocity = sim.data.qvel[joint_index]


    def linear_interpolate(self, last_goal, goal):
        """
        Set self.linear to be a function interpolating between last_goal and goal based on the ramp_ratio
        """
        # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
        delta_x_per_step = (goal - last_goal) / self.interpolation_steps
        self.linear = np.array([(last_goal + i * delta_x_per_step) for i in range(1, int(self.interpolation_steps) + 1)])

    def action_to_torques(self, action, policy_step):
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Returns dimensionality of the actions
        """
        dim = self.control_dim
        return dim
class JointVelocityController(Controller):
    """
    Class to interprete actions as joint velocities
    """
    ##  "joint_velocity":{"control_range": [1, 1, 1, 1, 1, 1, 1], "kv": [8.0, 7.0, 6.0, 4.0, 2.0, 0.5, 0.1], "interpolation": "linear"},

    def __init__(self, control_range, kv, max_action=1, min_action=-1, interpolation=None):
        super(JointVelocityController, self).__init__(
            control_max=np.array(control_range),
            control_min=-1 * np.array(control_range),
            max_action=max_action,
            min_action=min_action,
            interpolation=interpolation)

        self.kv = np.array(kv)
        self.interpolate = True

        self.last_goal = np.zeros(self.control_dim)
        self.step = 0

    def reset(self):
        super().reset()
        self.step = 0
        self.last_goal = np.zeros(self.control_dim)

    def action_to_torques(self, action, policy_step):
        action = self.transform_action(action) # clipping is done
        if policy_step:
            self.step = 0
            self.goal = np.array((action))

            if self.interpolation == "linear":
                self.linear_interpolate(self.last_goal, self.goal)
            else:
                self.last_goal = np.array((self.goal))

        if self.interpolation == "linear":
            self.last_goal = self.linear[self.step]

            if self.step < self.interpolation_steps - 1:
                self.step += 1
        # Torques for each joint are kv*(q_dot_desired - q_dot)
        torques = np.multiply(self.kv, (self.last_goal - self.current_joint_velocity))

        return torques
