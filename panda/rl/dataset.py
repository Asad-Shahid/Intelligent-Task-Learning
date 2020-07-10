from collections import defaultdict
from time import time
import numpy as np

class ReplayBuffer:
    def __init__(self, keys, buffer_size, sample_func):
        self._size = buffer_size # buffer size in config

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    # store the episode
    def store_episode(self, rollout): # stores an episode given a rollout; calling each time adds a new episode
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # sample transitions
        transitions = self._sample_func(self._buffers, batch_size) # buffers contains rollout(s), batch_size is in config
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers['ac']) # no. of episodes in buffer


class RandomSampler:
    def sample_func(self, episode_batch, batch_size_in_transitions): # episode_batch is buffers
        rollout_batch_size = len(episode_batch['ac']) # no. of episodes in buffer
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) #  selects a list of episode idxs i.e. [2,0,4,0...], length equal to batch_size
        t_samples = [np.random.randint(len(episode_batch['ac'][episode_idx])) for episode_idx in episode_idxs] # [135,78,54,180,...] list

        transitions = {}
        for key in episode_batch.keys(): # selects transitions experiences corresponding to sampled timesteps from episode(s)
            # values are lists {'ob': [], 'ac': [], 'done': []}; len of each list is equal to batch_size
            transitions[key] = [episode_batch[key][episode_idx][t] for episode_idx, t in zip(episode_idxs, t_samples)]

        # selects next observations corresponding to sampled transitions from an episode(s)
        transitions['ob_next'] = [episode_batch['ob'][episode_idx][t + 1] for episode_idx, t in zip(episode_idxs, t_samples)]

        # join sequence of array values in transitions; ob values is a dict {'robot-state': array(), 'object-state': array()}, others are array
        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys}
            else:
                new_transitions[k] = np.stack(v)
        return new_transitions
