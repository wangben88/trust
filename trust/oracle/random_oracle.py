from trust.oracle.base_oracle import TRUSTOracle
import itertools
import random


class RandomOracle(TRUSTOracle):

    def generate(self, num_samples, s1s2, budget):
        """Returns a random subset of all possible partitions, ignoring num_samples and budget.

        Args:
            num_samples (int): number of random partitions to return
            s1s2 (tuple): Contains s1/s2 sets of variables defining the sum-node to be split
            budget (float): ignored; random sampling is fast

        Returns: List of splits (partitions)
        """
        # Note: Currently does not check for duplicates.
        s1, s2 = s1s2[0], s1s2[1]
        s2_list = list(s2)

        splits = []
        for _ in range(num_samples):
            random.shuffle(s2_list)
            splits.append((s2_list[:len(s2_list) // 2], s2_list[len(s2_list) // 2:]))

        return splits