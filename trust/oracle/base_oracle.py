from abc import ABCMeta, abstractmethod


class TRUSTOracle:
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, num_samples, s1s2, budget):
        """Returns num_samples splits. Implemented by the underlying oracle method.

        Args:
            num_samples (int): number of sampled splits to be returned
            s1s2 (tuple): Contains s1/s2 sets of variables defining the sum-node to be split
            budget (float): time budget (s)
            iterations (int): number of iterations to run oracle method (depends on method)

        Returns:
             samples (list): List of sampled splits
        """
        return
