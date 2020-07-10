"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

from collections import defaultdict
import numpy as np


class Rollout():
    """
    Rollout storing an episode.(Stores data for all timesteps in a full episode)
    """

    def __init__(self):
        """ Initialize buffer. """
        self._history = defaultdict(list)

    def add(self, data):
        """ Add a transition @data to rollout buffer. """
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        """ Returns rollout buffer and clears buffer. """
        batch = {}
        batch['ob'] = self._history['ob']
        batch['ac'] = self._history['ac']
        batch['ac_before_activation'] = self._history['ac_before_activation']
        batch['done'] = self._history['done']
        batch['rew'] = self._history['rew']
        self._history = defaultdict(list)
        return batch

class RolloutRunner():
    """
    Run rollout given environment and policy.
    """

    def __init__(self, config, env, pi):
        """
        Args:
            config: configurations for the environment.
            env: environment.
            pi: policy.
        """

        self._config = config
        self._env = env
        self._pi = pi

    def run_episode(self, max_step=600, is_train=True):
        """
        Runs one episode and returns the rollout.

        Args:
            max_step: maximum number of steps of the rollout.
            is_train: whether rollout is for training or evaluation.
        """
        config = self._config
        device = config.device
        env = self._env
        pi = self._pi

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = defaultdict(list)
        acs = []
        done = False
        ep_len = 0
        ep_rew = 0
        ob = self._env.reset()

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        while not done and ep_len < max_step:
            # sample action from policy
            ac, ac_before_activation = pi.act(ob, is_train=is_train)

            rollout.add({'ob': ob, 'ac': ac, 'ac_before_activation': ac_before_activation})
            # joints positions
            saved_qpos.append(env.sim.get_state().qpos[:9].copy())

            # take a step
            ob, reward, done, info = env.step(ac) # e.g. info = {'episode_success': 0}
            env.render()
            rollout.add({'done': done, 'rew': reward})
            acs.append(ac)
            ep_len += 1
            ep_rew += reward

            for key, value in info.items():
                reward_info[key].append(value)

        # last frame
        rollout.add({'ob': ob})
        saved_qpos.append(env.sim.get_state().qpos[:9].copy())

        # compute average/sum of information
        ep_info = {'len': ep_len, 'rew': ep_rew, 'saved_qpos': saved_qpos} # qpos contains values for all timesteps, len and rew contain only sum
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)


        return rollout.get(), ep_info
