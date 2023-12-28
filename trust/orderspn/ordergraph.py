import numpy as np
from tqdm import tqdm
from trust.utils.misc import HiddenPrints

class OrderGraph():
    """Class representing structure of an OrderSPN.

    Consists of lists of sum and product layers, which alternate. Each layer consists of a list of nodes, which are
    specified differently for sums and products as explained below.

    """
    def __init__(self, items, seed=1235):

        self.items = tuple(sorted(items))

        # Each sum_layer is a list with tuple elements (S_1, S_2)
        self.sum_layers = []
        # Each prod_layer is a list with tuple (of tuple) elements ((S_1, S_{21}), (S_1 U S_{21}, S_{22}))
        self.prod_layers = []

        # Each prod_to_sum_map is a list representing mapping from node in a product layer, to its parent
        # in its parent sum layer.
        self.prod_to_sum_layers_map = []

        self._rand_state = np.random.RandomState(seed)

        # Root layer, consisting of root sum node, which is a tuple (parents, children)
        self.sum_layers.append( [((), self.items)] )

    def random_multi_split(self, expand_factors):
        """
        Construct graph by introducing expand_factors[i] child product nodes for each sum node in sum_layer[i], and
        split each product node into two sum nodes. The partitions at each sum node are chosen by random.

        Args:
            expand_factors (list): List of expansion factors for each layer
        """

        for expand_factor in expand_factors:
            prev_sum_layer = self.sum_layers[-1]
            new_prod_layer = []
            prod_to_sum_layer_map = []

            # Add new product layer
            for sum_idx, sum_node in enumerate(prev_sum_layer):
                for _ in range(expand_factor):
                    permuted_children = list(self._rand_state.permutation(list(sum_node[1])))

                    # Left and right parts; note that if len is odd, this puts more on the right
                    split_children = (permuted_children[:len(permuted_children)//2],
                                      permuted_children[len(permuted_children)//2:])

                    new_prod_layer.append(
                        (
                            (sum_node[0], tuple(sorted(split_children[0]))),
                            (tuple(sorted(sum_node[0]+tuple(split_children[0]))), tuple(sorted(split_children[1])))
                        )
                    )
                prod_to_sum_layer_map += [sum_idx] * expand_factor
            self.prod_layers.append(new_prod_layer)
            self.prod_to_sum_layers_map.append(prod_to_sum_layer_map)

            # Add new sum layer
            new_sum_layer = []
            for prod_node in new_prod_layer:
                new_sum_layer.append(prod_node[0])
                new_sum_layer.append(prod_node[1])
            self.sum_layers.append(new_sum_layer)

        # Start from the bottom of SPN
        self.sum_layers = list(reversed(self.sum_layers))
        self.prod_layers = list(reversed(self.prod_layers))
        self.prod_to_sum_layers_map = list(reversed(self.prod_to_sum_layers_map))


    def split_using_samples(self, expand_factors, budget, strategy, suppress_prints=True,
                            log=False, **kwargs):
        """
        Construct graph by introducing expand_factors[i] child product nodes for each sum node in sum_layer[i], using
        a specific splitting strategy specified as a parameter

        Args:
            expand_factors (list): List of expansion factors for each layer
            budget (float): the time budget (s) for constructing the graph
            strategy: Strategy for splitting OrderSPN nodes (e.g. ThresholdStrategy)
            suppress_prints (bool): Whether to suppress prints of the oracle method (which is called many times)
            log (bool): Whether to log progress
            kwargs: Any other arguments to the oracle method
        """
        budget_per_layer = budget/len(expand_factors)
        for layer_idx, expand_factor in enumerate(expand_factors):
            if log:
                print(f"Layer {layer_idx}")
            prev_sum_layer = self.sum_layers[-1]
            new_prod_layer = []

            prod_to_sum_layer_map = []

            budget_per_node = budget_per_layer/len(prev_sum_layer)
            iterable = enumerate(tqdm(prev_sum_layer)) if log else enumerate(prev_sum_layer)
            for sum_idx, sum_node in iterable:
                if suppress_prints:
                    with HiddenPrints():
                        splits = strategy.generate(num_samples=expand_factor, budget=budget_per_node, s1s2=sum_node,
                                                   **kwargs)
                else:
                    splits = strategy.generate(num_samples=expand_factor, budget=budget_per_node, s1s2=sum_node,
                                               **kwargs)
                for split in splits:
                    new_prod_layer.append(
                                (
                                    (sum_node[0], tuple(sorted(split[0]))),
                                    (tuple(sorted(sum_node[0] + tuple(split[0]))), tuple(sorted(split[1])))
                                )
                            )
                prod_to_sum_layer_map += [sum_idx] * len(splits)
            self.prod_layers.append(new_prod_layer)
            self.prod_to_sum_layers_map.append(prod_to_sum_layer_map)

            # Add new sum layer
            new_sum_layer = []
            for prod_node in new_prod_layer:
                new_sum_layer.append(prod_node[0])
                new_sum_layer.append(prod_node[1])
            self.sum_layers.append(new_sum_layer)

        # Start from the bottom of SPN
        self.sum_layers = list(reversed(self.sum_layers))
        self.prod_layers = list(reversed(self.prod_layers))
        self.prod_to_sum_layers_map = list(reversed(self.prod_to_sum_layers_map))


