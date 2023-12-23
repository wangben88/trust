import numpy as np
import random

class MarginalEvidence:
    """Represents evidence on the parents of each variable in the DAG.

    For each node, a list of literals is stored, where a literal is a positive integer corresponding to another node.

    TODO: implement negative literals
    """
    def __init__(self, d):
        self.d = d
        self.evidence = [[] for _ in range(d)]

    def __iter__(self):
        return self.evidence

    def __getitem__(self, item):
        return self.evidence[item]

    def set_node_evidence(self, node, literals):
        self.evidence[node] = literals

    def add_node_evidence(self, node, literal):
        self.evidence[node].append(literal)

    def remove_node_evidence(self, node, literal):
        if literal in self.evidence[node]:
            self.evidence[node].remove(literal)
        else:
            raise AssertionError

    def rand_marginal(self, G, n):
        """Generate evidence randomly (for purposes of testing).

        Args:
            G (np.array): [d, d] graph to extract edges from
            n: Number of edges to randomly select
        """
        row_indices, column_indices = np.where(G == 1)
        if n > len(row_indices):
            raise ValueError

        sampled_row_and_column_idxs = random.sample(list(zip(row_indices, column_indices)), n)
        for sampled_row_idx, sampled_column_idx in sampled_row_and_column_idxs:
            self.add_node_evidence(node=sampled_column_idx, literal=sampled_row_idx)

