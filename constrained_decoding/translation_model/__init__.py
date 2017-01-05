"""A translation model interface for neural MT systems"""
from abc import ABCMeta, abstractmethod


class AbstractConstrainedTM:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def start_hypothesis(self, *args, **kwargs):
        """Produce the initial hypothesis of this model"""
        pass

    @abstractmethod
    def generate(self, hyp, n_best=1):
        """Generate from the TM output vocabulary using the translation hypothesis"""
        pass

    @abstractmethod
    def generate_constrained(self, hyp):
        """Start new from hyp.constraints. New constraints must not already be used by the hypothesis"""
        pass

    @abstractmethod
    def continue_constrained(self, hyp):
        """Continue the constraint hyps that have already been started"""
        pass

