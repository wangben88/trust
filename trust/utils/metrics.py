import numpy as np
import sklearn
import causaldag as cd
import scipy.stats as st

def auroc(edge_probs, edge_true):
    """Computes AUROC (area under the receiver operating characteristic curve) for DAG.

    Args:
        edge_probs (np.array): (d, d) Marginal probabilities for each edge
        edge_true (np.array): (d, d) Adjacency matrix (true edges, 1 or 0)

    Returns:
        auroc (float): AUROC over all d(d-1) edges
    """
    edge_probs_flat = edge_probs.flatten()
    edge_true_flat = edge_true.flatten()

    fpr, tpr, _ = sklearn.metrics.roc_curve(edge_true_flat, edge_probs_flat)
    auroc = sklearn.metrics.auc(fpr, tpr)

    return auroc

def pdag_shd(g_samples, g_true):
    """Compute the Expected Structural Hamming Distance between the true CPDAG and the CPDAGs corresponding to the
    sampled graphs.

    Args:
        g_samples (list): List of sampled DAGs from structure learning method
        g_true (np.array): (d, d) Adjacency matrix of true DAG
    """
    shd = 0
    ground_truth_dag = cd.DAG.from_amat(g_true)
    ground_truth_pdag = ground_truth_dag.cpdag()
    for g in g_samples:
        est_dag = cd.DAG.from_amat(g)
        est_pdag = est_dag.cpdag()
        shd += ground_truth_pdag.shd(est_pdag)
    return shd/len(g_samples)

def mll(g_samples, data, model):
    """Computes marginal log-likelihood of data given the DAG structure, averaged over all DAGs in the sample.

    Args:
        g_samples (list): List of sampled DAGs from structure learning method
        data (np.array): (..., d) Data to compute the MLL for
        model: Model for the likelihood (e.g. BGe)

    Returns:
        mll (float): Averaged marginal log-likelihood
    """
    mll = 0
    for g_sample in g_samples:
        kk = model.mll(g_sample, data)
        mll += kk
    return mll/len(g_samples)



def pairwise_linear_ce(g_samples, data, bge_model, params_per_graph=10):
    """Returns the pairwise (linear) causal effect, averaged over the DAG samples.

     For a given DAG, computes causal effects between pairs of variables, by sampling and averaging over parameters of
     the Bayesian net (given the DAG structure and data). For each pair of variables (i, j), this is averaged over
     all DAGs in the sample to return a matrix of averaged pairwise causal effects.

    Args:
        g_samples (list): List of sampled DAGs from structure learning method
        data (np.array): (..., d) Data for computing/sampling parameters
        bge_model (BGe): BGe model for the likelihood
        params_per_graph (int): How many times to sample the parameters (for each graph)

    Returns:
        avg_effects (np.array): (d, d) matrix of averaged pairwise causal effects
    """

    R = bge_model.calc_R(data)
    N, d = data.shape

    B = [[] for _ in range(d)]
    for G_sample in g_samples:
        for i in range(d):
            parents_mask = G_sample[:, i].astype(bool)
            if np.any(parents_mask):
                l = np.sum(parents_mask) + 1
                parents_child_mask = np.copy(parents_mask)
                parents_child_mask[i] = True

                R22 = R[i, i]
                R12 = R[parents_mask, i]
                R21 = R[i, parents_mask]
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
                bs = dist.rvs(params_per_graph)
                for b in bs:
                    column = np.zeros(d)
                    column[parents_mask] = b
                    B[i].append(column)
            else:
                for _ in range(params_per_graph):
                    B[i].append(np.zeros(d))

    B = np.array(B)  # (d-col, num_total_samples, d-row)
    B = np.swapaxes(np.swapaxes(B, 0, 1), 1, 2)

    effects = [np.linalg.inv(np.eye(d) - B_sample) for B_sample in B]
    avg_effects = np.mean(np.array(effects), axis=0)

    return avg_effects

def pairwise_linear_ce_given_params(edge_weights):
    """Returns the pairwise causal effect given the matrix of edge weights.

    Args:
        edge_weights (np.array): (d, d) Weights of the linear model

    Returns:
        effects (np.array): (d, d) matrix of pairwise causal effects
    """
    d = edge_weights.shape[0]
    effects = np.linalg.inv(np.eye(d) - edge_weights)

    return effects

def pairwise_linear_ce_mse(effects_pred, effects_true):
    """Returns the mean squared error of pairwise causal effects, between predicted and true effects. The average
    error is taken over all variable pairs (i, j).

    Args:
        effects_pred (np.array): (d, d) Predicted pairwise causal effects
        effects_true (np.array): (d, d) True pairwise causal effects

    Returns:
        mse (float): Scalar averaged mean squared error
    """

    mse = np.mean((effects_pred - effects_true) ** 2)

    return mse







