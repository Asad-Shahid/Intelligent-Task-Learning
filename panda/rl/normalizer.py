from collections import OrderedDict
import numpy as np
from utils.mpi import mpi_average


class SubNormalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf, clip_obs=np.inf):
        if isinstance(size, list): # size is ob_space dimension i-e [10], [36]
            self.size = size
        else:
            self.size = [size]
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.clip_obs = clip_obs

        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def _clip(self, v): # clips the observation in range defined by clip_obs e.g.(-200, 200)
        return np.clip(v, -self.clip_obs, self.clip_obs)

    # update the parameters of the normalizer
    def update(self, v): # takes in observation array i-e. v['robot-state']; shape is e.g. (201,36)
        v = self._clip(v)
        v = v.reshape([-1] + self.size) # size is dim. of observation e.g. [36], [10], depending on obs key
        # do the computing
        self.local_sum += v.sum(axis=0) # sum obs along first axis(timesteps); e.g. shape is 36
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0] # e.g. 201

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = mpi_average(local_sum)
        local_sumsq[...] = mpi_average(local_sumsq)
        local_count[...] = mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None): # v is observation array
        v = self._clip(v)
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)

    def state_dict(self):
        return {'sum': self.total_sum, 'sumsq': self.total_sumsq, 'count': self.total_count}

    def load_state_dict(self, state_dict):
        self.total_sum = state_dict['sum']
        self.total_sumsq = state_dict['sumsq']
        self.total_count = state_dict['count']
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))


class Normalizer:
    def __init__(self, shape, eps=1e-2, default_clip_range=np.inf, clip_obs=np.inf):
        self._shape = shape # shape = ODict([('object-state', [10]), ('robot-state', [36])])
        if not isinstance(shape, dict):
            self._shape = {'': shape}
        print('New ob_norm with shape', self._shape)

        self._keys = sorted(self._shape.keys()) # ['object-state', 'robot-state']

        self.sub_norm = {} # i-e {'object-state': SubNormalizer([10], eps, 5, 200), 'robot-state': SubNormalizer([36], eps, 5, 200)}
        for key in self._keys:
            self.sub_norm[key] = SubNormalizer(self._shape[key], eps, default_clip_range, clip_obs)

    # update the parameters of the normalizer
    def update(self, v): # v is rollout['ob']
        if isinstance(v, list):
            if isinstance(v[0], dict):
                # stacks obs arrays values acc. to keys i-e ['robot-state'] arrays for all time steps stacked together
                v = OrderedDict([(k, np.asarray([x[k] for x in v])) for k in self._keys])
            else:
                v = np.asarray(v)

        if isinstance(v, dict):
            for k, v_ in v.items():
                if k in self._keys:
                    self.sub_norm[k].update(v_)
        else:
            self.sub_norm[''].update(v)

    def recompute_stats(self):
        for k in self._keys:
            self.sub_norm[k].recompute_stats()

    # normalize the observation
    def _normalize(self, v, clip_range=None):
        if not isinstance(v, dict):
            return self.sub_norm[''].normalize(v, clip_range)

        return OrderedDict([(k, self.sub_norm[k].normalize(v_, clip_range)) for k, v_ in v.items() if k in self._keys])

    def normalize(self, v, clip_range=None):
        # v is rollout['ob'], a list where each element(ref. to timestep) is ODict([('robot-state', array()), ('obj-state', array())])   OR
        #   transitions['ob'], a dict {'robot-state': array(), 'object-state': array()} array length is equal to batch_size
        if isinstance(v, list):
            return [self._normalize(x, clip_range) for x in v] # x is a odict
        else:
            return self._normalize(v, clip_range)

    def state_dict(self):
        return OrderedDict([(k, self.sub_norm[k].state_dict()) for k in self._keys])

    def load_state_dict(self, state_dict):
        for k in self._keys:
            self.sub_norm[k].load_state_dict(state_dict[k])
