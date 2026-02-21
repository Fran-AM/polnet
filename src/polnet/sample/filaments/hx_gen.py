
import random
import numpy as np

class HxParamGen():
    """
    Class to model random distribution for helix fiber parameters
    """

    def gen_length(self, min_l, max_l):
        """
        Generate a length with a range following a uniform distribution

        :param min_l: minimum length
        :param max_l: maximum length
        :return:
        """
        assert (min_l >= 0) and (min_l <= max_l)
        return (max_l - min_l) * random.random() + min_l

    def gen_persistence_length(self, min_p):
        """
        Generate a persistence length according an exponential distribution with lambda=1 and minimum value

        :param min_p: minimum persistence value
        :return:
        """
        return min_p + np.random.exponential()

    def gen_zf_length(self, min_zf=0, max_zf=1):
        """
        Generates a z-axis factor within a range

        :param min_zf: minimum value (default 0)
        :param max_zf: maximum value (default 1)
        """
        assert (min_zf >= 0) and (max_zf <= 1)
        return (max_zf - min_zf) * random.random() + min_zf

    def gen_den_cf(self, low, high):
        """
        Generates a density correction factor within a range

        :param low: minimum value
        :param high: maximum value
        """
        assert (low >= 0) and (high >= low)
        return (high - low) * random.random() + low

class HxParamGenBranched(HxParamGen):
    """
    Class to model random distribution for helix fiber parameters with branches
    """

    def gen_branch(self, b_prob=0.5):
        """
        Generates a boolean that is True (branching) with some input probability

        :param b_prob: branching probability [0, 1) (default 0.5)
        :return: a boolean
        """
        if random.random() <= b_prob:
            return True
        else:
            return False