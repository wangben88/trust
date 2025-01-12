import math
import torch
import numpy as np
import argparse

from trust.learning.learn_trust import learn_ordergraph, learn_orderspn
from trust.oracle.gadget_oracle import GadgetOracle
from trust.oracle.dibs_oracle import DibsOracle
from trust.oracle.enumeration_oracle import EnumerationOracle
from trust.oracle.random_oracle import RandomOracle
from trust.utils.generation import generate_linear_data, make_linear_model, make_erdosrenyi_graph
from trust.learning.split_strategy import ThresholdStrategy
from trust.utils.bge import BGe
from trust.utils.metrics import auroc, mll, pairwise_linear_ce, pairwise_linear_ce_mse, pdag_shd
from trust.leaf_scores.leaf_handler import LeafHandler
from trust.orderspn.evidence import MarginalEvidence
from trust.utils.misc import HiddenPrints

from dibs.config.example import DiBSMarginalExampleSettings
from sumu.gadget import Gadget

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--num-nodes", type=int, default=16, help="Number of variables/nodes")
    parser.add_argument("-e", "--edges-per-node", type=int, default=2, help="Average number of edges per node in the random Erdos-Renyi graph")
    parser.add_argument("-tr", "--train-size", type=int, default=100, help="Number of training samples")
    parser.add_argument("-te", "--test-size", type=int, default=1000, help="Number of test samples")
    parser.add_argument("-wm", "--weight-mean", type=float, default=0.0, help="Mean edge weight for linear model")
    parser.add_argument("-exp", "--expansion-factors", type=int, nargs='+', default=[64, 16, 6, 2], help="Expansion factors for each layer")
    parser.add_argument("-so", "--strong-oracle", type=str, default="gadget", help="Strong oracle: \"gadget\", \"dibs\", or \"random\"")
    parser.add_argument("-wo", "--weak-oracle", type=str, default="enumeration", help="Weak oracle: \"enumeration\"")
    parser.add_argument("-md", "--min-dimension", type=int, default=5, help="Minimum dimension to use the strong oracle for")
    parser.add_argument("-cp", "--num-candidate-parents", type=int, default=16, help="Number of candidate parents to consider for each node")
    parser.add_argument("-dev", "--device", type=str, default="cpu", help="device, e.g. \"cuda\" or \"cpu\"")
    parser.add_argument("-tb", "--total-time-budget", type=int, default=300, help="Total time budget")
    parser.add_argument("-pr", "--precomputation-ratio", type=float, default=0.2, help="Precomputation ratio")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    num_candidate_parents = min(args.num_candidate_parents, args.num_nodes - 1)
    assert(len(args.expansion_factors) == math.ceil(math.log2(args.num_nodes)))

    # Generate data
    rng = np.random.default_rng(args.seed)
    G = make_erdosrenyi_graph(d=args.num_nodes, edges_per_node=args.edges_per_node)
    B = make_linear_model(G, rng, weight_mean=args.weight_mean)
    X_train = generate_linear_data(args.train_size, B, rng)
    X_test = generate_linear_data(args.test_size, B, rng)

    # Perform precomputation of parent set scores
    with HiddenPrints():
        candidates, score, score_array = Gadget(data=X_train, 
                                                mcmc={"n_dags": 10000},
                                                run_mode={"name": "budget", "params": {"t": args.total_time_budget*args.precomputation_ratio}},
                                                cons={"K": num_candidate_parents}).return_cand_parents_and_score()
    
    lh = LeafHandler(C=candidates, c_r_score=score.c_r_score, c_c_score=score.c_c_score, score_array=score_array)
    lh.precompute_sum_and_max(log=True)

    # Learn OrderSPN
    with HiddenPrints():
        if args.strong_oracle == "gadget":
            strong_oracle = GadgetOracle(X_train, K=num_candidate_parents)
        elif args.strong_oracle == "dibs":
            strong_oracle = DibsOracle(G, B, X_train)
        
        if args.weak_oracle == "enumeration":
            weak_oracle = EnumerationOracle()
        elif args.weak_oracle == "random":
            weak_oracle = RandomOracle()
        
        strategy = ThresholdStrategy(strong_oracle, weak_oracle, min_dimension=args.min_dimension)
    
    og = learn_ordergraph(args.num_nodes, strategy, args.expansion_factors, 
                          time_budget=args.total_time_budget*(1-args.precomputation_ratio), seed=args.seed,
                      suppress_prints=True, log=True)
    
    ospn = learn_orderspn(og, device=args.device, leaf_function=lh)
    print('Loss: ', ospn.learn_spn())


    G_samples = ospn.sample(1000, args.num_nodes)
    bge_model = BGe(d=args.num_nodes, alpha_u=1)

    # Compute marginal edge probabilities
    marg_details = MarginalEvidence(args.num_nodes)
    pairwise_edge_probs = np.zeros((args.num_nodes, args.num_nodes))
    for j in range(args.num_nodes):
        for i in range(args.num_nodes):
            if (j != i) and (i not in marg_details[j]):
                marg_details.add_node_evidence(j, i)
                pairwise_edge_probs[i][j] = np.exp(ospn.marginal(marg_details).cpu().detach().numpy())
                marg_details.remove_node_evidence(j, i)

    # Compute AUROC
    trust_auroc = auroc(pairwise_edge_probs, np.copy(G))

    # Compute KL-divergence
    trust_kl = mll(np.copy(G_samples), X_test, bge_model)

    # Compute BACE matrix; compare approximate (sampling-based) and exact methods
    approx_avg_pairwise_effects = pairwise_linear_ce(np.copy(G_samples), X_train, bge_model)
    exact_avg_pairwise_effects = ospn.bace(X_train, bge_model).cpu().detach().numpy()

    approx_trust_mse = pairwise_linear_ce_mse(approx_avg_pairwise_effects, B)
    exact_trust_mse = pairwise_linear_ce_mse(exact_avg_pairwise_effects, B)

    # Compute expected SHD
    trust_shd = pdag_shd(np.copy(G_samples), np.copy(G))

    print('Metrics')
    print('--------------------------------------')
    print(f'SHD |  {trust_shd:4.1f},')
    print(f'AUROC| {trust_auroc:5.2f},')
    print(f'MLL| {trust_kl:4.1f},')
    print(f'CE_MSE| {approx_trust_mse:5.5f},')
    print(f'CE_MSE_exact| {exact_trust_mse:5.5f},')


        