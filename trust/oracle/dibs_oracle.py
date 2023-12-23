import os

os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import jax.numpy as jnp
from jax import random

import numpy as np

from oracle.dibs.dibs.inference import JointDiBS, MarginalDiBS
from oracle.dibs.dibs.kernel import JointAdditiveFrobeniusSEKernel, FrobeniusSquaredExponentialKernel
from oracle.dibs.dibs.eval.target import make_linear_gaussian_model,  make_linear_gaussian_equivalent_model
from oracle.dibs.dibs.utils.graph import elwise_acyclic_constr_nograd as constraint
from oracle.dibs.dibs.config.example import DiBSExampleSettings, DiBSMarginalExampleSettings
from oracle.dibs.dibs.eval.target import Target

import itertools

from trust.oracle.base_oracle import TRUSTOracle

def top_sorting(G, s1s2):
    """Returns a topological sorting of nodes in s2, according to the DAG G.
    """
    _, s2 = s1s2[0], s1s2[1]
    # Preprocess: only care about ordering within s2 nodes
    G = np.copy(G)
    G = G[:, s2][s2, :]  # only care about s2 rows and columns

    d = G.shape[1]
    S = []
    L = []
    for i in range(d):
        if np.all(G[:, i] == 0):
            S.append(i)

    if (len(S) == 0):
        raise Exception

    while S:
        n = S.pop(np.random.randint(len(S))) # random choice
        L.append(s2[n])    # returns original index, rather than index within s2
        for m in range(d):
            if G[n, m] == 1:
                G[n, m] = 0
                if np.all(G[:, m] == 0):
                    S.append(m)

    return L

def build_marginal_target(G, W, X, graph_prior_str = 'er'):
    n, d = X.shape
    graph_prior_str = graph_prior_str
    key = random.PRNGKey(123)
    key, subk = random.split(key)

    target = make_linear_gaussian_equivalent_model(key=subk, n_vars=d, graph_prior_str=graph_prior_str)

    x = jnp.array(X)
    g = jnp.array(G)
    th = jnp.array(W)
    target = Target(
        passed_key=target.passed_key,
        graph_model=target.graph_model,
        generative_model=target.generative_model,
        inference_model=target.inference_model,
        n_vars=target.n_vars,
        n_observations=target.n_observations,
        n_ho_observations=target.n_ho_observations,
        g=g,
        theta=th,
        x=x,
        x_ho=x,
        x_interv=target.x_interv,
    )

    return target

def build_joint_target(G, W, X, graph_prior_str='er'):
    n, d = X.shape
    graph_prior_str = graph_prior_str
    key = random.PRNGKey(1239)
    key, subk = random.split(key)

    target = make_linear_gaussian_model(key=subk, n_vars=d, graph_prior_str=graph_prior_str)
    x = jnp.array(X)
    g = jnp.array(G)
    th = jnp.array(W)

    target = Target(
        passed_key=target.passed_key,
        graph_model=target.graph_model,
        generative_model=target.generative_model,
        inference_model=target.inference_model,
        n_vars=target.n_vars,
        n_observations=target.n_observations,
        n_ho_observations=target.n_ho_observations,
        g=g,
        theta=th,
        x=x,
        x_ho=x,
        x_interv=target.x_interv,
    )

    return target


def build_dibs_marginal_model(target, d, s1s2=None, hparams=DiBSMarginalExampleSettings()):
    model = target.inference_model

    no_interv_targets = jnp.zeros(d).astype(bool)  # observational data

    def log_prior(single_w_prob):
        """log p(G) using edge probabilities as G"""
        return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_marginal_target(single_w):
        """log p(theta, D | G) =  log p(theta | G) + log p(D | G, theta)"""
        log_marg_lik = model.log_marginal_likelihood_given_g(w=single_w, data=jnp.array(target.x),
                                                             interv_targets=no_interv_targets)
        return log_marg_lik

    # initialize kernel and algorithm
    kernel = FrobeniusSquaredExponentialKernel(
        h=hparams.h_latent
    )

    dibs = MarginalDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_marginal_prob=log_marginal_target,
            alpha_linear=hparams.alpha_linear,
            s1s2=s1s2
    )

    return dibs

def build_dibs_joint_model(target, d, s1s2=None, hparams=DiBSExampleSettings()):
    model = target.inference_model

    no_interv_targets = jnp.zeros(d).astype(bool)  # observational data

    def log_prior(single_w_prob):
        """log p(G) using edge probabilities as G"""
        return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_joint_target(single_w, single_theta, rng):
        """log p(theta, D | G) =  log p(theta | G) + log p(D | G, theta)"""
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=jnp.array(target.x),
                                       interv_targets=no_interv_targets)
        return log_prob_theta + log_lik


    # initialize kernel and algorithm
    kernel = JointAdditiveFrobeniusSEKernel(
        h_latent=hparams.h_latent,
        h_theta=hparams.h_theta)

    dibs = JointDiBS(
        kernel=kernel,
        target_log_prior=log_prior,
        target_log_joint_prob=log_joint_target,
        alpha_linear=hparams.alpha_linear,
        s1s2=s1s2)

    return dibs


class DibsOracle(TRUSTOracle):
    def __init__(self, G, W, X_train, graph_prior_str = 'er', use_marginal=True, hparams=DiBSMarginalExampleSettings()):
        """Builds the DiBS target.

        Args:
            G (np.array): [d, d] directed adjacency matrix representing graph
            W (np.array): [d, d] directed weight matrix
            X_train (np.array): [..., d] matrix containing training data
            graph_prior_str (string): string describing prior used in DiBS, 'er' by default
            use_marginal (bool): whether to use marginal DiBS or joint DiBS. marginal DiBS by default
        """
        super().__init__()

        # Convert data into Target
        if use_marginal:
            self.target = build_marginal_target(G, W, X_train, graph_prior_str)
        else:
            self.target = build_joint_target(G, W, X_train, graph_prior_str)

        self.use_marginal = use_marginal

        self.hparams = hparams


    def generate(self, num_samples, s1s2, budget): # iterations=1000,
        """Returns num_samples splits, using DiBS as oracle.

        Args:
            num_samples (int): number of sampled orderings to be returned
            s1s2 (tuple): Contains s1/s2 sets of variables defining the sum-node to be split
            budget (float): Time budget
            hparams (dict): Hyperparameters for DiBS

        Returns:
            splits (list): List of sampled splits (partitions)
        """
        n, d = self.target.x.shape
        key = random.PRNGKey(123)
        key, subk = random.split(key)
        if s1s2 is not None:
            s1s2_arr = (jnp.array(s1s2[0], dtype=jnp.int64), jnp.array(s1s2[1], dtype=jnp.int64))  # convert to jnp arrays
        n_particles = num_samples  # use all DiBS particles as samples

        # (1) Build prior and likelihood + prepare dibs

        # Building the target (jit compilation) each time generate is run results in significant overhead. Unfortunately
        # this may be unavoidable as the s1s2 will change every time we want to generate samples.

        if self.use_marginal:
            dibs = build_dibs_marginal_model(self.target, d, s1s2_arr, self.hparams)
        else:
            dibs = build_dibs_joint_model(self.target, d, s1s2_arr, self.hparams)

        # (2) Initialize particles and Run DiBS
        key, subk = random.split(key)
        if self.use_marginal:
            init_particles_z = dibs.sample_initial_random_particles(
                key=subk, n_particles=n_particles, n_vars=d)
            key, subk = random.split(key)
            particles_z = dibs.sample_particles(key=subk, budget=budget,
                                                init_particles_z=init_particles_z)
        else:
            init_particles_z, init_particles_theta = dibs.sample_initial_random_particles(
                key=subk, model=self.target.inference_model, n_particles=n_particles, n_vars=d)
            key, subk = random.split(key)
            particles_z, particles_theta = dibs.sample_particles(key=subk, budget=budget,
                                                                 init_particles_z=init_particles_z,
                                                                 init_particles_theta=init_particles_theta)

        # (3) Postprocessing
        particles_g = dibs.particle_to_g_lim(particles_z)

        # Remove cyclic graphs
        is_dag = constraint(particles_g, d) == 0
        particles_g = particles_g[is_dag, :, :]

        # Pad entries if there were cyclic graphs (this is fairly rare)
        if (len(particles_g) < num_samples):
            particles_g = list(itertools.islice(itertools.cycle(particles_g), num_samples))

        splits = []
        for G_sample in particles_g:
            top_order = top_sorting(G_sample, s1s2=s1s2)
            split_children = (top_order[:len(top_order) // 2],
                              top_order[len(top_order) // 2:])
            splits.append(split_children)

        return splits