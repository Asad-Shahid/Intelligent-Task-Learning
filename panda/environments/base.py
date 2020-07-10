from collections import OrderedDict

from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml

import numpy as np
from utils import MujocoPyRenderer


REGISTERED_ENVS = {}

def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class

def get_env(name):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""

    return REGISTERED_ENVS[name]

def make(name, config=None):
    """
    Creates a new environment instance with @name and @config.
    """
    env = get_env(name)

    # get default configuration
    if config is None:
        import argparse
        import config.grasping as grasp

        parser = argparse.ArgumentParser()
        grasp.add_argument(parser)

        config, unparsed = parser.parse_known_args()

    return env(config)

class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "PandaEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """Initializes a Mujoco Environment."""

    def __init__(self, config):

        self.render_collision_mesh = config.render_collision_mesh
        self.render_visual_mesh = config.render_visual_mesh
        self.control_freq = config.control_freq
        self.horizon = config.horizon
        self.ignore_done = config.ignore_done
        self.viewer = None
        self.model = None

        # Load the model
        self._load_model()

        # Initialize the simulation
        self._initialize_sim()

        # Run all further internal (re-)initialization required
        self._reset_internal()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep #0.002 is default simulation timestep in mujoco
        self.control_freq = control_freq
        self.control_timestep = 1. / control_freq # 0.01 if control freq is 100

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation
        """
        # if we have an xml string, use that to create the sim. Otherwise, use the local model
        self.mjpy_model = self.model.get_model()

        # Create the simulation instance and run a single step to make sure changes have propagated through sim state
        self.sim = MjSim(self.mjpy_model)
        #self.sim.step()

        # Setup sim time based on control frequency
        self.initialize_time(self.control_freq)

    def reset(self):
        """Resets simulation."""
        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        """Resets simulation internal configurations."""

        # create visualization screen or renderer
        if self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

            # make sure mujoco-py doesn't block rendering frames
            self.viewer.viewer._render_every_frame = True

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state() # copy of simulator state (qpos(23), qvel(21))
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

    def _get_observation(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        if isinstance(action, list):
            action = {key: val for ac_i in action for key, val in ac_i.items()}
        if isinstance(action, dict):
            action = np.concatenate([action[key] for key in self.action_space.shape.keys()])

        self.timestep += 1
        # If the env.step frequency is lower than the mjsim timestep frequency(1/0.002), the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        policy_step = True
        for i in range(int(self.control_timestep / self.model_timestep)):
            self._pre_action(action, policy_step)
            self.sim.step() # advances simulation
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)
        return self._get_observation(), reward, done, info

    def _pre_action(self, action, policy_step=False):
        """Do any preprocessing before taking an action."""
        self.sim.data.ctrl[:] = action #sim.data.ctrl is the control signal that is sent to the actuators in the simulation

    def _post_action(self, action):
        """Do any housekeeping after taking an action."""
        reward = self.reward(action)
        info = {}
        info['episode_success'] = int(self._check_success())
        info['grasp'] = self.has_grasp
        info['phase'] = self.phase

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, info

    def reward(self, action):
        """Reward should be a function of state and action."""
        return 0

    def render(self):
        """
        Renders to an on-screen window.a
        """
        self.viewer.render()

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return False
