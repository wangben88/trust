import numpy as np
import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import sys, os
import scipy.stats as st
from torch_scatter import scatter, scatter_max
from torch_scatter.composite import scatter_logsumexp
from trust.utils.misc import HiddenPrints

from torch.optim import Adam

EPS = 1e-15

class NodeLayer(torch.nn.Module, ABC):
    def __init__(self):
        super(NodeLayer, self).__init__()

    @abstractmethod
    def initialize(self):
        pass

class OrderSumLayer(NodeLayer):
    """Implements a sum layer in an OrderSPN.

    Also stores the connections to the child layer. If child_to_sum is None, the children are assumed to be regularly
    assigned, e.g. if self.num = 10 and self.child_num = 30, then children [0, 1, 2] are connected to node 0,
    [3, 4, 5] to node 1, etc. If child_to_sum is specified, this maps a child node to its parent sum node.
    """
    def __init__(self, num, child_num, device, child_to_sum=None):
        """
        Args:
            num: the number of sum nodes in this layer
            child_num: the number of sum nodes in the last layer
            child_to_sum: mapping from children to sum nodes in this layer. If None, the children are assumed to be
                regularly assigned.
        """
        super(OrderSumLayer, self).__init__()
        self.num = num
        self.child_num = child_num
        self.device = device

        self.child_per_node = self.child_num // self.num
        if child_to_sum is not None:
            self.child_to_sum = torch.tensor(child_to_sum, device=self.device)
        else:
            self.child_to_sum = child_to_sum

    def initialize(self, params=None, equal_prob=False):
        """Initializes the parameters of the sum layer.

        The number of parameters is equal to the number of nodes in the child layer (self.child_num), as each child
        has one parent, with the corresponding edge being labelled with a weight.

        Args:
            params (Tensor): optionally, provide a parameter initialization
            equal_prob (bool): if True, set all parameters to have equal value; else draw values randomly from uniform
                distribution.
        """
        if params:
            self.logparams = params
        else:
            if equal_prob:
                self.logparams = nn.Parameter(
                torch.log(torch.ones([self.child_num], device=self.device)/ self.child_per_node))
            else:
                self.logparams = nn.Parameter(
                    torch.log(torch.ones([self.child_num], device=self.device).uniform_() / self.child_per_node))

    def forward_no_log(self, input, verbose=False):
        """Performs forward computation through the layer.
        Args:
            input (Tenosr): (batch, self.child_num) the input to the layer

        Return:
            node_output (Tensor): (batch, self.num): the output of nodes in this layer
        """
        batch, _ = input.size()

        if self.child_to_sum is None:
            input = input.view(batch, self.num, self.child_per_node)
            #logparams = self.get_logparams()
            params = torch.exp(self.get_normalized_logparams().view(self.num, self.child_per_node))

            node_output = (input * params).sum(dim=-1)
        else:
            node_output = scatter(dim=1, index=self.child_to_sum, src=input.view(batch, -1) * torch.exp(self.get_normalized_logparams()), reduce="sum")

        if verbose:
            print(node_output)
        return node_output

    def forward(self, input, verbose=False):
        """Performs forward computation through the layer (in log-domain).
        Args:
            input (Tensor): (batch, self.child_num): the input to the layer

        Return:
            node_output (Tensor): (batch, self.num): the output of nodes in this layer
        """
        batch, _ = input.size()

        if self.child_to_sum is None:
            input = input.view(batch, self.num, self.child_per_node)
            logparams = self.get_normalized_logparams().view(self.num, self.child_per_node)

            node_output = torch.logsumexp(input + logparams, dim=-1)
        else:
            node_output = scatter_logsumexp(dim=-1, index=self.child_to_sum, src=input.view(batch, -1) + self.get_normalized_logparams())
        if verbose:
            print(node_output)
        return node_output

    def entropy(self, input):
        """Compute entropy of sum layer nodes, given entropy of child layer nodes.

        Args:
            input (Tensor): (batch, self.child_num) entropies of child nodes

        Returns:
            output (Tensor): (batch, self.num) entropies of sum nodes
        """
        if self.child_to_sum is None:
            logparams = self.get_normalized_logparams().view(self.num, self.child_per_node)
            params = torch.exp(logparams)
            wlogw = (params * logparams).sum(dim=-1).unsqueeze(0) # unsqueeze batch dim
        else:
            logparams = self.get_normalized_logparams()
            params = torch.exp(logparams)
            wlogw = scatter(dim=-1, index=self.child_to_sum, src=(logparams * params), reduce="sum").unsqueeze(0)

        wt_sum = self.forward_no_log(input)
        return wt_sum - wlogw

    def max(self, scores, indices):
        """Performs node-wise maximizations through the layer.
        Args:
            scores: array of dim (batch_size, self.child_num) representing  scores from child layer
            indices: array of dim (batch_size, self.child_num, num_assignments) containing, for each node in the
                child layer, the indices (num_assignments) of the leaf nodes corresponding to the maximization

        Returns:
            scores: array of dim (batch_size, self.num) representing output scores from this layer
            indices: array of dim (batch_size, self.num, num_assignments) containing, for each node in the
                this layer, the indices (num_assignments) of the leaf nodes corresponding to the maximization
        """
        batch_size, _, num_assignments = indices.shape

        combined_scores = scores + self.get_normalized_logparams() # (batch_size, self.child_num)

        # (batch_size, self.num), (batch_size, self.num)
        # child_absolute_indices expresses which of the nodes in the child layer have been selected.
        if self.child_to_sum is None:
            child_to_sum = torch.repeat_interleave(torch.arange(self.num, device=self.device), self.child_per_node)
            scores_out, child_absolute_indices = scatter_max(dim=-1, index=child_to_sum, src=combined_scores)
        else:
            scores_out, child_absolute_indices = scatter_max(dim=-1, index=self.child_to_sum, src=combined_scores)

        # The indices array records, for each node in the child layer, the corresponding leaf layer node(s). Thus
        # we now index the indices array with child_absolute_indices
        # (batch_size, self.num, num_assignments)
        max_as_index = torch.stack(num_assignments * [child_absolute_indices], dim=2)

        # (batch_size, self.num, num_assignments)
        leaf_indices = torch.gather(input=indices, dim=1, index=max_as_index)

        return scores_out, leaf_indices

    def forward_bace(self, input):
        """Performs forward computation of the Bayesian averaged causal effect (BACE)
        Args:
            input (Tensor): (d*d, self.child_num): BACE for child layer nodes

        Return:
            node_output (Tensor): (d*d, self.num): BACE for this layer nodes
        """
        # forward_no_log works for sum layers
        return self.forward_no_log(input)

    def get_normalized_logparams(self):
        """Computes and returns normalized version of logparams.

        Returns:
            logparams: normalized version of self.logparams
        """
        if self.child_to_sum is None:
            logparams = self.logparams.view(self.num, self.child_per_node)
            logparams = logparams - torch.logsumexp(logparams, dim=-1, keepdim=True)
            logparams = logparams.flatten()
        else:
            logparams_norm = scatter_logsumexp(dim=-1, index=self.child_to_sum, src=self.logparams)
            logparams_norm = torch.gather(input=logparams_norm, dim=-1, index=self.child_to_sum)
            logparams = self.logparams - logparams_norm
        return logparams

    def sample_next_layer_indices(self, layer_indices):
        """Given 1d Tensor of current layer indices, return samples of next (child) layer indices according to the
        probability distribution (parameters) at each sum node.

        Args:
            layer_indices (1d Tensor): Tensor containing indices for current layer, corresponding to nodes from which
            we want to sample nodes from the next layer. Repeated indices are permitted.

        Returns:
            nextlayer_index_samples: 1d Tensor containing indices for next layer
        """
        # Given 1d tens of current layer indices, return samples of next layer indices
        if self.child_to_sum is None:
            params = torch.exp(self.get_normalized_logparams().view(self.num, self.child_per_node))
            params_per_idx = params[layer_indices, :]

            distrn = torch.distributions.Categorical(probs=params_per_idx)
            samples = distrn.sample()
            base_nextlayer_index = layer_indices * self.child_per_node
            nextlayer_index_samples = base_nextlayer_index + samples
        else:

            sum_to_first_child = [None]*self.num
            for child_pos in reversed(range(self.child_num)):
                sum_to_first_child[self.child_to_sum[child_pos]] = child_pos
            sum_to_first_child.append(self.child_num)

            next_layer_candidate_indices = torch.cat([
                torch.arange(self.child_num, device=self.device)[sum_to_first_child[sum_idx]: sum_to_first_child[sum_idx+1]]
                for sum_idx in layer_indices #range(self.num)
            ]) # (sum of child_i, where child_i is the number of children of the ith node in layer_indices)
            sample_indices = torch.cat([
                torch.full(size=torch.Size([sum_to_first_child[sum_idx+1] - sum_to_first_child[sum_idx]]), fill_value=sample_idx, device=self.device)
                for sample_idx, sum_idx in enumerate(layer_indices)
            ])
            # for efficiency, use gumbel-max trick
            samples = torch.distributions.Gumbel(loc=0.0, scale=1.0).sample(next_layer_candidate_indices.shape).to(self.device)
            unnorm_probs = (self.logparams[next_layer_candidate_indices] - self.logparams[next_layer_candidate_indices].max()).exp()
            _, raw_indices = scatter_max(dim=0, index=sample_indices, src=samples + unnorm_probs)
            nextlayer_index_samples = next_layer_candidate_indices[raw_indices]

        return nextlayer_index_samples




class OrderProdLayer(NodeLayer):
    """Represents OrderSPN product layer."""
    def __init__(self, num, device):
        super(OrderProdLayer, self).__init__()

        self.num = num
        self.device = device

    def initialize(self, params=None):
        pass

    def forward(self, input):
        """Performs forward computation of the product layer.

        Args:
            input: child layer values

        Returns:
            output: product layer values
        """
        node_output = input[:, np.arange(0, input.shape[1]-1, 2)] + input[:, np.arange(1, input.shape[1], 2)]
        return node_output

    def entropy(self, input):
        """Given entropy of child layer nodes, returns entropy of product layer nodes.

        Args:
            input (Tensor): entropy of child layer nodes

        Returns:
            output (Tensor): entropy of product layer nodes
        """
        return self.forward(input)

    def sample(self, input):
        """Given samples from the child layer nodes, produces sample for the product layer nodes.

        Each product node has precisely two children, so this corresponds to merging the samples of those two
        children.

        Args:
            input (Tensor): samples from input layer nodes; shape (batch, 2*self.num, d, d)

        Returns:
            output (Tensor): samples from the product layer nodes; shape (batch, self.num, d, d)
        """
        node_output = np.concatenate([input[:, np.arange(0, input.shape[1] - 1, 2), :, :],
                                      input[:, np.arange(1, input.shape[1], 2), :, :]],
                                     axis=2)
        return node_output

    def max(self, scores, indices):
        """Performs (passes through) node-wise maximizations through the layer.
        Args:
            scores: array of dim (batch_size, 2*self.num) representing input scores from child layer
            indices: array of dim (batch_size, 2*self.num, num_assignments) containing, for each node in the
                child layer, the indices (num_assignments) of the leaf nodes corresponding to the maximization

        Returns:
            scores: array of dim (batch_size, self.num) representing output scores from this layer
            indices: array of dim (batch_size, self.num, num_assignments) containing, for each node in the
                this layer, the indices (num_assignments) of the leaf nodes corresponding to the maximization
        """
        batch_size, layer_size, num_assignments = indices.shape
        scores = self.forward(scores)
        indices = torch.cat([indices[:, torch.arange(0, layer_size - 1, 2, device=self.device), :],
                                  indices[:, torch.arange(1, layer_size, 2, device=self.device), :]],
                                     dim=2)
        return scores, indices

    def forward_bace(self, input, layer_details):
        """Performs forward computation of the Bayesian averaged causal effect (BACE).

        For each product node, we combine the BACE matrices for its children.
        Args:
            input (Tensor): (d*d, 2*self.num): BACE for child layer nodes

        Return:
            node_output (Tensor): (d*d, self.num): BACE for this layer nodes
        """
        d = int(math.sqrt(input.shape[0]))
        prev_layer_size = input.shape[1]
        input_bace_tensor = input.view(d, d, prev_layer_size).cpu()  # move to cpu as following is not tensorized
        # tensorization is difficult as the dimensions of s1,s21,s22 can differ across different nodes.

        output_bace_tensor = torch.zeros((d, d, self.num))

        for index in range(self.num):
            left_index = 2*index
            right_index = 2*index + 1
            out_node_bace = output_bace_tensor[:, :, index]
            in_node_left_bace = input_bace_tensor[:, :, left_index]
            in_node_right_bace = input_bace_tensor[:, :, right_index]

            s1 = layer_details[index][0][0]
            s21 = layer_details[index][0][1]
            s22 = layer_details[index][1][1]


            # (0) fill in diagonal
            out_node_bace.fill_diagonal_(1)

            # (1) i \in S_{22} and j \in S_{21}; BACE = 0

            # (2) (a) i \in S_1 \cup S_{21} and j \in S_{21}
            #     (b) i \in S_{22} and j \in S_{22}
            i_indices, j_indices = torch.meshgrid(torch.LongTensor(s1 + s21), torch.LongTensor(s21), indexing="ij")
            out_node_bace[i_indices, j_indices] = in_node_left_bace[i_indices, j_indices]

            i_indices, j_indices = torch.meshgrid(torch.LongTensor(s22), torch.LongTensor(s22), indexing="ij")
            out_node_bace[i_indices, j_indices] = in_node_right_bace[i_indices, j_indices]


            # (3) i \in S_1 \cup S_{21} and j \in S_{22}

            i_left_indices, j_left_indices = torch.meshgrid(torch.LongTensor(s1 + s21), torch.LongTensor(s21), indexing="ij")
            i_right_indices, j_right_indices = torch.meshgrid(torch.LongTensor(s21), torch.LongTensor(s22),
                                                            indexing="ij")
            i_indices, j_indices = torch.meshgrid(torch.LongTensor(s1 + s21), torch.LongTensor(s22), indexing="ij")
            out_node_bace[i_indices, j_indices] = torch.mm(in_node_left_bace[i_left_indices, j_left_indices],
                                                          in_node_right_bace[i_right_indices, j_right_indices])

        output_bace_tensor = output_bace_tensor.view(d*d, self.num).to(self.device)
        return output_bace_tensor



    def get_next_layer_indices(self, layer_indices):
        """Utility function: returns (indices of) children of given product nodes

        Args:
            layer_indices (Tensor): list of product layer indices

        Returns:
            left_indices, right_indices (Tensor): list of left and right child indices
        """
        left_indices = layer_indices * 2
        right_indices = layer_indices * 2 + 1
        return left_indices, right_indices


class OrderLeafLayer(NodeLayer):
    def __init__(self, leaf_graph_layer, device, leaf_handler):
        """Initializes the leaf layer (all leaf nodes).

        Args:
            leaf_graph_layer (list): List of leaf nodes (from OrderGraph)
            device (str): Torch device
            leaf_handler (LeafHandler): object for computing properties of leaf node distributions (sums, etc.)
        """
        super(OrderLeafLayer, self).__init__()

        self.leaf_graph_layer = leaf_graph_layer
        self.device = device
        self.leaf_handler = leaf_handler

    def initialize(self, params=None):
        pass

    def summed_score(self, index):
        """Returns (precomputed) summed score for a given leaf node.

        This is the sum of scores for all possible parent sets for this leaf node.

        Args:
            index (int): index of leaf node within layer

        Returns:
            sc (float): summed score
        """
        leaf_node = self.leaf_graph_layer[index]
        if leaf_node[1]:
            child = leaf_node[1][0]
            parents = leaf_node[0]
            sc = self.leaf_handler.sum(v=child, U=set(parents))
        else:
            sc = 0  # log_prob 0

        return sc

    def sample_from_leaf(self, index):
        """Samples parent set from the leaf node

        Args:
            index (int): index of leaf node within layer

        Returns:
            sample: if the leaf (S_1, S_2) has a variable (i.e. |S_2| >= 1), then returns (S_2, parent_set) where
            parent_set is the sampled parents; otherwise if not (|S_2| = 0), then returns 'NA'
        """
        leaf_node = self.leaf_graph_layer[index]
        if leaf_node[1]:
            child = leaf_node[1][0]
            parents = leaf_node[0]
            # NOTE: no need to restrict parent set to candidates as gadgetscore does it for us already
            pset = self.leaf_handler.sample_pset(v=child, U=set(parents))[0]

            return (pset[0], list(pset[1]))
        else:
            return 'NA'

    def full_summed_scores(self):
        """Returns Tensor of summed scores for each node in the leaf layer

        Returns:
            all_summed_scores (Tensor): Tensor of summed scores for each node in the leaf layer
        """
        all_summed_scores = []
        for index in range(len(self.leaf_graph_layer)):
            all_summed_scores.append(self.summed_score(index))
        return torch.Tensor(all_summed_scores).to(self.device).unsqueeze(0) # unsqueeze for batch dimension

    def marg_summed_scores(self, evidence):
        """Returns Tensor of summed scores for each node in the leaf layer, where the score is summed over all parent
        sets consistent with the given evidence.

        Args:
            evidence (MarginalEvidence): Evidence on parents of each variable (i.e. partial instantiation)

        Returns:
            all_summed_scores (Tensor): Tensor of summed scores for each node in the leaf layer
        """
        all_summed_scores = []
        for leaf_node in self.leaf_graph_layer:
            sc = 0
            if leaf_node[1]:
                child = leaf_node[1][0]
                parents = leaf_node[0]
                evidence_child = evidence[child]

                if set(evidence_child).issubset(set(parents)):
                    sc = self.leaf_handler.f_func(v=child, A=frozenset(evidence_child),
                                                  A_prime_comp=frozenset(parents)) - \
                         self.leaf_handler.f_func(v=child, A=frozenset(),
                                                  A_prime_comp=frozenset(parents))
                else:
                    sc = -np.inf

            all_summed_scores.append(sc)
        return torch.Tensor(all_summed_scores).to(self.device).unsqueeze(0)  # unsqueeze for batch dimension

    def max_score(self, evidence):
        """Returns Tensor of max scores for each node in the leaf layer, where the max score is taken over all parent
        sets consistent with the given evidence.

        Args:
            evidence (MarginalEvidence): Evidence on parents of each variable (i.e. partial instantiation)

        Returns:
            all_summed_scores (Tensor): Tensor of summed scores for each node in the leaf layer
        """

        all_max_scores = []
        all_max_assignments = []
        for leaf_node in self.leaf_graph_layer:
            sc = 0
            child = 'NA'
            ass = tuple()
            if leaf_node[1]:
                child = leaf_node[1][0]
                parents = leaf_node[0]
                evidence_child = evidence[child]

                if set(evidence_child).issubset(set(parents)):
                    ass, sc = self.leaf_handler.f_max_func(v=child, A=frozenset(evidence_child),
                                              A_prime_comp=frozenset(parents))
                    sc = sc - self.leaf_handler.f_func(v=child, A=frozenset(),
                                              A_prime_comp=frozenset(parents))
                else:
                    sc = -np.inf
                    ass = tuple()
            all_max_scores.append(sc)
            all_max_assignments.append((child, ass))
            # returns tensor of scores (batch_size, layer_size)
            # and assignments/graph columns for each leaf in the layer
        return (torch.Tensor(all_max_scores).to(self.device).unsqueeze(0), # unsqueeze for batch dimension
                all_max_assignments)

    def bace(self, X, bge_model, sample_size=1, params_per_sample=1):
        """Approximates the Bayesian averaged causal effect (BACE) matrix for each leaf node.

        Draws sample_size samples of parent sets, then params_per_sample parameters for the Bayesian network given the
        parent sets, then exactly computes the (averaged) causal effect for each set of parameters, then averages.

        Args:
            X (Tensor): (batch_size, d) training data
            bge_model (BGe): BGe model
            sample_size (int): Number of sampled parent sets to draw; increasing this will increase accuracy, at the
                cost of time.
            params_per_sample (int): Number of sampled parameters to draw; increasing this will increase accuracy, at
                the cost of time.

        Returns:
            bace (Tensor): (d*d, self.num) BACE matrix for each leaf node in the layer. The matrix is flattened to a
                single dimension in order to use the batched OrderSPN operations.
        """
        N = X.shape[0]
        d = X.shape[1]
        R = bge_model.calc_R(X)
        num_leaves = len(self.leaf_graph_layer)

        bace_input = np.zeros((d, d, num_leaves))
        for index in range(num_leaves):
            column_sum = np.zeros(d)
            leaf_node = self.leaf_graph_layer[index]
            if leaf_node[1]:
                child = leaf_node[1][0]
                for _ in range(sample_size):
                    _, parents = self.sample_from_leaf(index)
                    # print(child, " ", parents)
                    # print()
                    parents_mask = np.zeros(d, dtype=np.bool_)
                    parents_mask[parents] = True
                    if np.any(parents_mask):
                        l = np.sum(parents_mask) + 1
                        parents_child_mask = np.copy(parents_mask)
                        parents_child_mask[child] = True

                        R22 = R[child, child]
                        R12 = R[parents_mask, child]
                        R21 = R[child, parents_mask]
                        R11 = R[parents_mask, :][:, parents_mask]

                        loc = np.linalg.inv(R11) @ R12
                        deg_free = bge_model.alpha_w + N - d + l
                        shape = np.linalg.inv(
                            deg_free /
                            (R22 - R21 @ np.linalg.inv(R11) @ R12
                             ) *
                            R11

                        )

                        dist = st.multivariate_t(loc=loc, shape=shape, df=deg_free)
                        bs = np.reshape(np.array(dist.rvs(params_per_sample)), (params_per_sample, -1))  # ensure is 2-dimensional
                        b_avg = np.mean(bs, axis=0)
                        column_sum[parents_mask] += b_avg

                column_avg = column_sum/sample_size
                bace_input[:, child, index] = column_avg
            else:
                pass # if no child, no need to update

        bace_input[np.arange(d), np.arange(d), :] = 1  # the ACE for a node on itself is 1

        batched_bace_input = torch.Tensor(bace_input).to(self.device).view(d*d, num_leaves) # flatten the matrix to a single dimension

        return batched_bace_input




class OrderSPN(nn.Module, ABC):
    """Class implementing OrderSPNs.

    OrderSPNs deal with tensors of dimension (batch_size, layer_size), where layer_size is the number of
    nodes in the layer.
    """

    def __init__(self, graph_sum_layers, graph_prod_layers, leaf_handler, device, prod_to_sum_layers_map=None):
        """Construct OrderSPN based on layers from an OrderGraph.

        Args:
            graph_sum_layers (list): List of sum layers from the bottom of the SPN (including leaf layer)
            graph_prod_layers (list): list of product layers from the bottom of the SPN
            leaf_handler (LeafHandler): object for computing properties of leaf node distributions (sums, etc.)
            prod_to_sum_layers_map (list): List (over sum layers), with each entry being a mapping from product nodes
                in the child layer to the sum nodes. If not specified, mapping assumed to be regular.
                e.g. for sum layer with 3 nodes and child product layer with 6 nodes, could have:
                     [0, 0, 1, 1, 2, 2] (regular) or [0, 1, 1, 2, 2, 2]
            device (str): PyTorch device used for Tensor computation
        """
        super().__init__()

        self.graph_sum_layers = graph_sum_layers
        self.graph_prod_layers = graph_prod_layers
        self.prod_to_sum_layers_map = prod_to_sum_layers_map
        assert(len(graph_sum_layers) == len(graph_prod_layers) + 1)
        if prod_to_sum_layers_map is not None:
         assert(len(prod_to_sum_layers_map) == len(graph_prod_layers))
        self.num_combined_layers = len(graph_prod_layers)
        self.leaf_layer = OrderLeafLayer(leaf_graph_layer=graph_sum_layers[0],
                                         device=device,
                                         leaf_handler=leaf_handler)

        self.device = device

        # Layers, minus the "input" sum layer in graph_sum_layers
        sum_layers = []
        prod_layers = []
        for i in range(len(graph_prod_layers)):
            prod_layers.append(OrderProdLayer(num=len(self.graph_prod_layers[i]),
                                              device=device))
            if prod_to_sum_layers_map is not None:
                sum_layers.append(OrderSumLayer(num=len(self.graph_sum_layers[i+1]),
                                                child_num=len(self.graph_prod_layers[i]),
                                                child_to_sum=self.prod_to_sum_layers_map[i],
                                                device=device))
            else:
                sum_layers.append(OrderSumLayer(num=len(self.graph_sum_layers[i + 1]),
                                                child_num=len(self.graph_prod_layers[i]),
                                                device=device))

        # Interleave layers
        self.layers = [arb_layers[layer_num] for layer_num in range(self.num_combined_layers)
                       for arb_layers in (prod_layers, sum_layers)]

        self.net = None

        # For dealing with leaf distributions appropriately
        self.leaf_handler = leaf_handler

    def initialize(self, equal_prob=False):
        """Initializes parameters of OrderSPN.

        Args:
            equal_prob (bool): if True, set all parameters to have equal value; else draw values randomly from uniform
                distribution.
        """
        for layer in self.layers:
            if isinstance(layer, OrderSumLayer) and equal_prob:
                layer.initialize(equal_prob=equal_prob)
            else:
                layer.initialize()

        self.nn = nn.ModuleList(self.layers)

    #################################### Core OrderSPN operations ##################################################

    def forward(self, input):
        """Perform a forward (bottom-up) computation through the OrderSPN, taking in input from the leaf layer and
        outputting the result at the root node. Computations are performed in the log-domain.

        Args:
            input (Tensor): (batch_size, leaf_layer_size). This corresponds to the values of leaf layer, and is fed into
            the first product layer.

        Returns
            output (Tensor): (batch_size,); contains the output at the root node.
        """
        output = input.clone()
        for layer in self.layers:
            output = layer.forward(output)
        return output.squeeze()

    def mpe_forward(self, input):
        """Performs MPE computation for a deterministic OrderSPN. In particular, given input leaf layer, traverses the
        layers bottom-up, maintaining the maximizing score at each node in the layer, and the indices of the leaf nodes
        corresponding to that assignment.

        Args:
            input (Tensor): (batch_size, leaf_layer_size). This corresponds to the leaf layer, and is fed into
            the first product layer.

        Returns:
            scores (Tensor): (batch_size,) corresponds to score of maximizing assignment
            indices (Tensor): (batch_size, d), where the second dimension indexes the leaf node corresponding to
            the parent set of the i^th variable in the DAG.
        """

        scores = input.clone()
        batch_size, layer_size = scores.shape
        indices = torch.stack(batch_size*[torch.arange(layer_size, device=self.device)])
        # Size (batch_size, layer_size), describes the indices

        indices = indices.unsqueeze(2)
        # Now (batch_size, layer_size, 1), where 1 will be the dimension of assignments
        # As we go up the layers, layer_size will shrink, while the last dimension will increase in size (as each node
        # will have a greater scope and thus correspond to more leaf nodes).
        for layer in self.layers:
            scores, indices = layer.max(scores, indices)

        return scores.squeeze(1), indices.squeeze(1)

    def bace_forward(self, input):
        """Performs Bayesian ACE computation for a deterministic OrderSPN. In particular, given input leaf layer,
        traverses the layers bottom-up, maintaining the Bayesian ACE matrix at each node in each layer.

        Args:
            input (Tensor): (d*d, leaf_layer_size). This corresponds to the BACE matrices for leaf layer, where
            the first dimension is the flattened matrix and the second dimension enumerates the nodes of the layer.

        Returns:
            bace (Tensor): (d, d). This corresponds to the BACE matrix for the root, i.e. the circuit.
        """

        output = input.clone()

        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, OrderProdLayer):
                prod_layer_index = layer_index//2
                output = layer.forward_bace(output, self.graph_prod_layers[prod_layer_index])
            else:
                output = layer.forward_bace(output)

        return output.squeeze()




    def forward_vi(self, input):
        """Perform a forward (bottom-up) computation through the OrderSPN, taking in input from the leaf layer and
        outputting the result at the root node. Computations are performed in the log-domain for product nodes, but
        not for sum-nodes. This is necessary, in particular, for variational inference ELBO computations.

        Args:
            input (Tensor): (batch_size, leaf_layer_size). This corresponds to the leaf layer, and is fed into
            the first product layer.

        Returns:
            output (Tensor): (batch_size,) contains the output at the root node.
        """
        output = input.clone()
        for layer in self.layers:
            if isinstance(layer, OrderProdLayer):
                output = layer.forward(output)
            elif isinstance(layer, OrderSumLayer):
                output = layer.forward_no_log(output)
        return output.squeeze()
    
    def learn_spn_adam(self, lr=0.1, epochs=700):
        """Learns the parameters of the OrderSPN by maximizing the ELBO. The Adam optimizer is used.

        Args:
            lr (float): Learning rate
            epochs (int): Number of iterations for the optimizer
        """
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)

        input = self.leaf_layer.full_summed_scores()

        for epoch in range(epochs):
            self.zero_grad()

            loss = -self.forward_vi(input) - self.entropy()
            loss.backward()

            optimizer.step()

        return -loss.cpu().detach().numpy()  # ELBO

    def learn_spn(self):
        """Learns the parameters of the OrderSPN by maximizing the ELBO using closed form optimization.

        Returns:
            ELBO (float): the ELBO of the OrderSPN after learning parameters
        """
        with torch.no_grad():
            input = self.leaf_layer.full_summed_scores()
            output = input.clone()
            softmax = torch.nn.Softmax(dim = 0)
            for layer in self.layers:
                if isinstance(layer, OrderProdLayer):
                    output = layer.forward(output) # Add ELBOs together in the product node
                elif isinstance(layer, OrderSumLayer):
                    weight_entropy = torch.zeros([layer.num], dtype = output.dtype) # weight entropy calculates the - sum over w_i log w_i term
                    if self.prod_to_sum_layers_map is not None:
                        for sum_node in range(layer.num):
                            indices = ((layer.child_to_sum == sum_node).nonzero(as_tuple=True)[0])
                            layer.logparams[indices] = torch.log(softmax(output[0, indices]))
                            weight_entropy[sum_node] = -torch.sum(layer.logparams[indices] * torch.exp(layer.logparams[indices]))
                    else:
                        for sum_node in range(layer.num):
                            begin_index = sum_node * layer.child_per_node
                            end_index = (sum_node + 1) * layer.child_per_node
                            layer.logparams[begin_index:end_index] = torch.log(softmax(output[0, begin_index:end_index])) # Softmax of the child inputs produces the optimal weights
                            weight_entropy[sum_node] = -torch.sum(layer.logparams[begin_index:end_index] * torch.exp(layer.logparams[begin_index:end_index]))
                    output = layer.forward_no_log(output) + weight_entropy #layer.forward_no_log produces the weighted sum of ELBO and then add the weight entropy to get new ELBO
            loss = -self.forward_vi(input) - self.entropy()
        return -loss.cpu().detach().numpy()  # ELBO

    ####################################### OrderSPN queries ###############################################

    def entropy(self):
        """Computes entropy of the (deterministic) OrderSPN.

        Returns:
            output (float): Entropy of OrderSPN
        """
        output = torch.zeros(len(self.graph_sum_layers[0])).to(self.device).unsqueeze(0)
        for layer in self.layers:
            output = layer.entropy(output)
        return output.squeeze()

    def sample(self, num_samples, d):
        """Samples graphs from the OrderSPN by iteratively sampling nodes top-down from the root.

        Args:
            num_samples (int): Number of graph samples to return
            d (int): Dimension (number of variables) in graph
        """
        # Step 0: Sample iteratively top-down through the SPN

        # 1st dim is samples, 2nd dim is indices within that sample
        active_layer_indices = torch.zeros(torch.Size((num_samples, 1)), dtype=torch.int64).to(self.device)
        for layer in reversed(self.layers): # start from the top
            if isinstance(layer, OrderSumLayer):
                flat_active_layer_indices = torch.flatten(active_layer_indices)
                flat_nextlayer_indices = layer.sample_next_layer_indices(flat_active_layer_indices)
                active_layer_indices = flat_nextlayer_indices.view(num_samples, -1)
            elif isinstance(layer, OrderProdLayer):
                flat_active_layer_indices = torch.flatten(active_layer_indices)
                flat_left_indices, flat_right_indices = layer.get_next_layer_indices(flat_active_layer_indices)
                left_indices, right_indices = flat_left_indices.view(num_samples, -1), flat_right_indices.view(num_samples, -1)
                active_layer_indices = torch.cat([left_indices, right_indices], dim=1)

        # Step 1: Sample from the relevant leaves of the SPN

        graph_samples = []
        for leaf_sample in active_layer_indices:
            graph_sample = []
            for leaf_index in leaf_sample:
                node_sample = self.leaf_layer.sample_from_leaf(leaf_index)
                if node_sample != 'NA':
                    graph_sample.append(node_sample)
            graph_samples.append(graph_sample)

        # Step 2: Convert the representation to an adjacency matrix
        G_samples = [np.zeros((d, d)) for _ in range(num_samples)]
        for i in range(num_samples):
            for col in graph_samples[i]:
                col_idx = col[0]
                col_entry = col[1]
                if (col_entry):
                    G_samples[i][col_entry, col_idx] = 1

        return G_samples

    def marginal(self, evidence):
        """Computes the marginal probability of given evidence in the OrderSPN.

        Args:
            evidence (MarginalEvidence): Evidence on the graph edges

        Returns:
            p (float): probability of evidence
        """
        self.eval()

        marg_input = self.leaf_layer.marg_summed_scores(evidence)
        p = self.forward(marg_input)

        return p

    def conditional(self, evidence1, evidence2):
        """Computes the conditional probability of evidence2 given evidence1 in the OrderSPN. Note that evidence2 must
        contain evidence1.

        e.g. for p(c'|c), evidence1 is c, evidence2 is c' AND c.

        Args:
            evidence1 (MarginalEvidence): Evidence on the graph edges corresponding to the condition
            evidence2 (MarginalEvidence): Evidence on the graph edges corresponding to the condition AND marginal

        Returns:
            p (float): conditional probability
        """

        return self.marginal(evidence2) - self.marginal(evidence1)

    def mpe(self, evidence, d):
        """Computes the conditional MPE (most likely graph) for the OrderSPN given some evidence.

        Args:
            evidence (MarginalEvidence): condition
            d (int): number of graph nodes

        Returns:
            score (float): probability of most likely graph
            G (Tensor): (d, d) most likely graph
        """
        self.eval()

        marg_input, max_assignments = self.leaf_layer.max_score(evidence)
        scores, indices = self.mpe_forward(marg_input)

        # Step 2: Convert the representation to an adjacency matrix
        score, leaf_index_list = scores.squeeze(), indices.squeeze()
        G = np.zeros((d, d))
        for leaf_idx in leaf_index_list:
            (child, parent_set) = max_assignments[leaf_idx]
            if child != 'NA':
                for parent in parent_set:
                    G[parent, child]=1

        return score, G

    def bace(self, X, bge_model):
        """Computes the Bayesian averaged causal effect (BACE) matrix for the OrderSPN.

        Args:
            X (Tensor): (batch_size, d) training data
            bge_model (BGe): BGe model

        Returns:
            bace (Tensor): (d, d) BACE matrix
        """
        self.eval()

        bace_input = self.leaf_layer.bace(X, bge_model)
        bace_output = self.bace_forward(bace_input)

        d = int(math.sqrt(bace_output.shape[0]))

        return bace_output.view(d, d)


