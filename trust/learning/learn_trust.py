
from trust.orderspn.ordergraph import OrderGraph
from trust.orderspn.orderspn import OrderSPN


def learn_ordergraph(d,
                     strategy,
                     layer_to_expand_factor,
                     time_budget,
                     seed,
                     suppress_prints,
                     log):
    """Learn an OrderGraph (OrderSPN structure learning).

    Args:
        d (int): Dimension of problem (number of variables)
        strategy: Strategy for splitting OrderSPN nodes (e.g. ThresholdStrategy)
        layer_to_expand_factor (list): list of expansion factors (K) for each layer
        time_budget (float): Time budget (s) for structure learning
        seed (int): Random seed
        suppress_prints (bool): whether to suppress printing from the splitting strategy/oracle
        log (bool): whether to log progress

    Returns:
        og: learned OrderGraph
    """
    og = OrderGraph(list(range(d)), seed=seed)

    og.split_using_samples(expand_factors=layer_to_expand_factor, budget=time_budget, # or just pass time budget
                           strategy=strategy, suppress_prints=suppress_prints, log=log)

    return og

def learn_orderspn(og,
                   device,
                   leaf_function,
                   use_adam=False,
                   lr=0.1,
                   epochs=700,
                   use_childsum_mapping=False,
                   ):
    """Learn parameters of OrderSPN (parameter learning), which by default uses closed form optimization.

    Args:
        og (OrderGraph): structure of OrderSPN
        device (str): pytorch device (cuda or cpu)
        leaf_function (LeafHandler): implements function f from precomputation, outputting leaf values (derived from data)
        use_adam (bool): whether to use Adam optimizer instead of the default
        lr (float): Learning rate (only used if using Adam)
        epochs (int): Number of iterations for the optimizer (only used if using Adam)

    Returns:
        ospn: learned OrderSPN
    """
    if use_childsum_mapping:
        ospn = OrderSPN(og.sum_layers, og.prod_layers, device=device, leaf_handler=leaf_function,
                        prod_to_sum_layers_map=og.prod_to_sum_layers_map)
    else:
        ospn = OrderSPN(og.sum_layers, og.prod_layers, device=device, leaf_handler=leaf_function)
    ospn.initialize()

    elbo = ospn.learn_spn(use_adam=use_adam, lr=lr, epochs=epochs)

    return ospn
