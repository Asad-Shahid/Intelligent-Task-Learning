""" Define ActionSpace class to represent action space. """

from collections import OrderedDict
import numpy as np
from utils.logger import logger


class ActionSpace(object):
    """
    Base class for action space
    This action space is used in the provided RL training code.
    """

    def __init__(self, size, minimum=-1., maximum=1.):
        """
        Loads a mujoco xml from file.

        Args:
            size (int): action dimension.
            min: minimum values for action.
            max: maximum values for action.
        """
        self.size = size
        self.shape = OrderedDict([('default', size)])

        self._minimum = np.array(minimum)
        self._minimum.setflags(write=False)

        self._maximum = np.array(maximum)
        self._maximum.setflags(write=False)

    @property
    def minimum(self):
        """
        Returns the minimum values of the action.
        """
        return self._minimum

    @property
    def maximum(self):
        """
        Returns the maximum values of the action.
        """
        return self._maximum

    def keys(self):
        """
        Returns the keys of the action space.
        """
        return self.shape.keys()

    def __repr__(self):
        template = ('ActionSpace(shape={},''minimum={}, maximum={})')
        return template.format(self.shape, self._minimum, self._maximum)

    def __eq__(self, other):
        """
        Returns whether other action space is the same or not.
        """
        if not isinstance(other, ActionSpace):
            return False
        return (self.minimum == other.minimum).all() and (self.maximum == other.maximum).all()

    def sample(self):
        """
        Returns a sample from the action space.
        """
        return np.random.uniform(low=self.minimum, high=self.maximum, size=self.size)
