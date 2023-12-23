import networkx as nx
import numpy as np
from numpy.random import default_rng
import random
import jax.numpy as jnp
import jax

def make_erdosrenyi_graph(d, edges_per_node, seed=None):
    """Generates a random Erdos-Renyi directed acyclic graph.

    Args:
        d (int): Number of nodes in graph
        edges_per_node (int): Expected number of edges per node
        seed (int): Random seed

    Returns:
        G (np.array): (d, d) adjacency array representing the random graph
    """
    rng = default_rng(seed=seed)

    #p = edges_per_node / d
    p = min((edges_per_node * d) / (d * (d - 1) / 2), 0.5)
    #G = nx.erdos_renyi_graph(d, p, seed, directed=True)
    G_array = rng.binomial(n=1, p=p, size=(d, d))#nx.adjacency_matrix(G).toarray()

    # Make acyclic
    G_array = np.tril(G_array, k=-1)

    # Randomly permute nodes
    P = rng.permutation(np.eye(d))
    G_array_permuted = P.T @ G_array @ P

    return G_array_permuted

def make_linear_model(G, rng, weight_mean=0.0, weight_sd=1.0):
    """Given a DAG representing the graphical model, generate random weights of a linear model for each edge in the DAG.

    Args:
        G (np.array): (d, d) array representing the adjacency matrix
        rng: NumPy Generator
        weight_mean (float): Mean weight for edges
        weight_sd (float): Standard deviation of weight for edges

    Returns:
        weights: (d, d) array of weights
    """

    B = rng.normal(weight_mean, weight_sd, size=G.shape)
    B_masked = B * G

    return B_masked

def generate_linear_data(n_samples, B, rng, noise_sd=0.316):
    """Generates data from a Linear Gaussian model.

    Args:
        n_samples (int): Number of samples of data to generate
        B (np.array): (d, d) array of edge weights
        rng: NumPy Generator
        noise_sd (float/np.array): Either a scalar or a 1d array of noise standard deviations for each variable

    Returns:
         data (np.array): (n, d) representing the samples
    """
    d = B.shape[0]

    eps = rng.normal(loc=0, scale=noise_sd, size=(n_samples, d))

    return eps @ np.linalg.inv(np.eye(d) - B)