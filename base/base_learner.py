from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod


class BaseLearner(metaclass=ABCMeta):
    """
    Abstract base class for learners.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def _validate(self):
        pass

    @abstractmethod
    def inference(self):
        pass
