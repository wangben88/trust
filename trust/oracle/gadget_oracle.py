from sumu.gadget import Gadget, ScoreTRUST

import random
import numpy as np
from .base_oracle import TRUSTOracle

# Note on indices: In general, we use "relative" and "absolute" indices for variables:
# "absolute" are the original indices
# "relative" for s2 are the indices within s2. So, for instance, if s2 = [2, 3, 5, 8], variable 3 has absolute index 3
# but relative index 1.

def filter_s2(pars, s2):
    s2_absolute_parents = set(pars).intersection(set(s2))
    s2_relative_parents = (s2.index(abs_par) for abs_par in s2_absolute_parents)
    return s2_relative_parents

def calc_iters(n_particles, s2_d):
    """Default heuristic for number of iterations of Gadget MCMC to perform, given s1s2

    Args:
        n_particles (int): number of samples to return
        s2_d (int): dimension of s2

    Returns: number of iterations to apply
    """
    return max(1000, 32 * n_particles * 2**(s2_d))

def partition_split(partition):
    """Splits a partition into two halves (with equal number of variables on each side).

    Args:
        partition: Partition of set of variables, expressed as list of list

    Returns:
        partition1: first half of partition
        partition2: second half of partition
        order: ordering of variables consistent with the partition
    """
    partition_size = sum([len(part) for part in partition])

    partition1, partition2 = [], []
    partition1_size, partition2_size = 0, 0
    leftside = True
    for part in partition:
        if leftside and (partition1_size + len(part) >= partition_size//2):
            shuf_part_list = random.sample(list(part), len(part))
            part1 = set(shuf_part_list[:partition_size//2 - partition1_size])
            part2 = set(shuf_part_list[partition_size//2 - partition1_size:])
            if part1:
                partition1.append(part1)
                partition1_size += len(part1)
            if part2:
                partition2.append(part2)
                partition2_size += len(part2)
            leftside=False
        elif leftside:
            partition1.append(part)
            partition1_size += len(part)
        else:
            partition2.append(part)
            partition2_size += len(part)

    order = []
    for part in partition1:
        order += list(part)
    for part in partition2:
        order += list(part)

    return partition1, partition2, order


class GadgetOracle(TRUSTOracle):
    def __init__(self, X_train, K=None):
        """Performs the necessary precomputation to use Gadget as an oracle.

        Args:
            X_train (np.array): [..., d] matrix containing training data
            K (int): number of candidate parents per node
        """
        super().__init__()
        if K is not None:
            self.candidates, self.score, self.score_array = \
                Gadget(data=X_train, mcmc={"n_dags": 10000},
                       run_mode={"name": "budget", "params": {"t": 300}},
                       cons={"K": K}).return_cand_parents_and_score()


        else:
            self.candidates, self.score, self.score_array = \
                Gadget(data=X_train,
                       mcmc={"n_dags": 10000}).return_cand_parents_and_score()

        self.N, self.d = X_train.shape

    def return_relative_score(self, s1s2):
        s1, s2 = s1s2[0], s1s2[1]
        score_TRUST = ScoreTRUST(C=self.score.C, c_c_score=self.score.c_c_score,
                                                          c_r_score=self.score.c_r_score, s1=s1,
                                                          s2=s2, d=self.d)
        return score_TRUST

    def generate(self, num_samples, s1s2, budget, init_partition=None):
        """Returns num_samples orderings, using Gadget as oracle.

        Args:
            num_samples (int): number of sampled orderings to be returned
            s1s2 (tuple): Contains s1/s2 sets of variables defining the sum-node to be split
            budget (float): time budget (s)
            #iterations (int): Number of iterations of Gadget MCMC to apply
            init_partition: optionally, initialization for Gadget MCMC

        Returns: List of sampled splits (partitions)
        """
        s1, s2 = s1s2[0], s1s2[1]

        # Construct score and candidate parent sets for s1s2
        absolute_to_relative = {abso: rela for rela, abso in enumerate(s2)}
        if init_partition is None:
            init_partition_rel = None
        else:
            init_partition_rel = [{absolute_to_relative[abso] for abso in part} for part in init_partition]


        score_TRUST = ScoreTRUST(C=self.score.C, c_c_score=self.score.c_c_score,
                                                          c_r_score=self.score.c_r_score, s1=s1,
                                                          s2=s2, d=self.d)

        candidates_restricted_s2 = {v: filter_s2(self.candidates[node], s2) for v, node in enumerate(s2)}

        # Set up and run Gadget MCMC
        # Partitions here refer to Gadget partitions, NOT partitions into two sets in TRUST oracle
        dummy_data = np.empty((self.N, len(s2)))

        rm = {"name": "budget_mcmconly", "params": {"t": budget}}
        mcmc = {"n_dags": num_samples}

        partitions = Gadget(data=dummy_data, mcmc=mcmc,
                            run_mode=rm,
                            trust={"s1": s1, "s2": s2, "sample": False,
                                    "d": self.d, "candidates": candidates_restricted_s2,
                                    "score": score_TRUST,
                                    "init_part": init_partition_rel}).sample()
        partitions_absolute = [[{s2[v] for v in part} for part in partition] for partition in partitions]

        splits = [] # these are the "partitions" into two sets for TRUST oracle
        for partition in partitions_absolute:
            _, _,  top_order = partition_split(partition)

            split_children = (top_order[:len(top_order) // 2],
                              top_order[len(top_order) // 2:])

            splits.append(split_children)

        return splits