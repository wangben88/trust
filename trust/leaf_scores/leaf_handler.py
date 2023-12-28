from oracle.gadget.sumu.gadget import Score
from LeafScore import LeafScore
from tqdm import tqdm

class LeafHandler(Score):
    """Computes and stores precomputation results for leaf nodes in OrderSPNs.
    """
    def __init__(self, *, C, c_r_score, c_c_score, score_array):
        """Initializes LeafHandler class.

        Note that in this class, we refer to 'absolute' and 'relative' indices for variables. Absolute indices refer
        to the index of a variable, while relative indices are specific to a specific parent set (tuple) and indicate
        the position of that variable within the tuple.

        For example, if variable 6 had a candidate parent tuple C[6] = (3, 4, 8), then the indices would be:
        Abs | Relative
        3 | 0
        4 | 1
        8 | 2

        Args:
            C (dict): Maps variables to their candidate parent tuples
            c_r_score (np.array): Candidate set scores
            c_c_score (np.array): Candidate complement scores
            score_array (list):
        """

        super().__init__(C=C, c_r_score=c_r_score, c_c_score=c_c_score)
        self.C = C
        self.n = len(self.C)
        K = len(self.C[0])
        # Create leaf score object for each variable from 1 to n.
        self.leaf_scores = [LeafScore(K, node_score) for node_score in score_array]

    def precompute_sum_and_max(self, log=False):
        """Precomputes sum and max arrays for each LeafScore.

        Args:
            log (bool): whether to log progress
        """
        if log:
            for leaf_score in tqdm(self.leaf_scores):
                leaf_score.precompute_sum_and_max()
        else:
            for leaf_score in self.leaf_scores:
                leaf_score.precompute_sum_and_max()


    def _compute_relative(self, v, A, A_prime_comp):
        """Computes relative indices for the input absolute index arrays.

        Args:
            v: Variable to compute the relative indices for
            A (set): Set of absolute indices (corresponding to A_v in Appendix of paper)
            A_prime_comp (set): Set of absolute indices (corresponding to compleemnt of A'_v in Appendix of paper)

        Returns:
            A_relative (list): List of relative indices for A
            A_prime_relative (list): List of relative indices for A' (i.e. NOT in A_prime_comp)
        """
        A_relative = []
        A_prime_relative = []
        for relative_index, absolute_index in enumerate(self.C[v]):
            if absolute_index in A:
                A_relative.append(relative_index)
            if absolute_index not in A_prime_comp:
                A_prime_relative.append(relative_index)

        return A_relative, A_prime_relative

    def f_func(self, v, A, A_prime_comp):
        """Returns the precomputed value of f (see Appendix of paper).

        Args:
            v: Variable index
            A (set): Set of allowed (absolute) indices
            A_prime_comp (set): Complement of set of disallowed (absolute) indices

        Returns:
            value: f_i(A, A')
        """
        A_relative, A_prime_relative = self._compute_relative(v, A, A_prime_comp)

        return self.leaf_scores[v].sum(A_relative, A_prime_relative)

    def f_max_func(self, v, A, A_prime_comp):
        """Returns the precomputed value of f_max (see Appendix of paper).

        Args:
            v: Variable index
            A (set): Set of allowed (absolute) indices
            A_prime_comp (set): Complement of set of disallowed (absolute) indices

        Returns:
            parent_set: Set of parents corresponding to maximum (absolute indices)
            value: f_{max, i}(A, A')
        """
        A_relative, A_prime_relative = self._compute_relative(v, A, A_prime_comp)

        max_score, relative_parent_set = self.leaf_scores[v].max(A_relative, A_prime_relative)
        absolute_parent_set = tuple(self.C[v][relative_parent] for relative_parent in relative_parent_set)

        return absolute_parent_set, max_score

