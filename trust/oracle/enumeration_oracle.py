from trust.oracle.base_oracle import TRUSTOracle
import itertools


class EnumerationOracle(TRUSTOracle):

    def generate(self, num_samples, s1s2, budget):
        """Returns all possible partitions, ignoring num_samples and budget.

        Args:
            num_samples (int): ignored; always returns all possible partitions
            s1s2 (tuple): Contains s1/s2 sets of variables defining the sum-node to be split
            budget (float): ignored; always returns all possible partitions

        Returns: List of splits (partitions)
        """
        s1, s2 = s1s2[0], s1s2[1]

        splits = []
        for comb in itertools.combinations(s2, len(s2) // 2):
            splits.append((comb, set(s2).difference(comb)))

        return splits