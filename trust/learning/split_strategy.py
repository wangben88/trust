class ThresholdStrategy:
    """Implements a strategy for structure learning of OrderSPNs, where we have two distinct splitting methods: a strong
    and a weak oracle. The strong oracle is assumed to be more expensive, but higher quality.

    Supports two methods for deciding whether to use the strong or weak oracle. The first (default) uses the strong
    oracle whenever the dimension (i.e. size of S_2) is greater than a certain threshold. The second uses the strong
    oracle whenever the time budget allocated to the splitting exceeds a certain threshold.

    """
    def __init__(self, strong_oracle, weak_oracle, min_dimension=5, min_time=None):
        """Initializes splitting strategy.

        Args:
            strong_oracle: (stronger) oracle to be used above min_dimension
            weak_oracle: (weaker) oracle to be used below min_dimension
            min_dimension: dimension to switch oracles
            min_time: if min_dimension not set, use minimum time to switch oracles
        """
        self.strong_oracle = strong_oracle
        self.weak_oracle = weak_oracle
        self.min_dimension = min_dimension
        self.min_time = min_time

    def generate(self, num_samples, s1s2, budget, **kwargs):
        """Apply splitting strategy to a given OrderSPN node.

        Args:
            num_samples (int): Number of splits required
            s1s2 (tuple): Tuple (s1, s2) for the OrderSPN node.
            budget (float): Time budget (s) for this split
            **kwargs: Additional arguments to be passed to the specific oracle

        Returns:
            samples (list): List of sampled splits
        """
        s2 = s1s2[1]

        if self.min_dimension is None:
            if budget >= self.min_time:
                return self.strong_oracle.generate(num_samples, s1s2, budget, **kwargs)
            else:
                return self.weak_oracle.generate(num_samples, s1s2, budget, **kwargs)
        else:
            if len(s2) >= self.min_dimension:
                return self.strong_oracle.generate(num_samples, s1s2, budget, **kwargs)
            else:
                return self.weak_oracle.generate(num_samples, s1s2, budget, **kwargs)