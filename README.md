# TRUST: Tractable Uncertainty for Structure Learning

Official implementation of the ICML 2022 paper "Tractable Uncertainty for Structure Learning".

TRUST is a Bayesian structure learning method that approximately infers a posterior over Bayesian network structures
given data. The posterior distribution over graphs (structures) is represented as an OrderSPN model, which is a variant of
the sum-product network (SPN) for distributions over orderings/graphs. 

The distinguishing feature of OrderSPNs is their
ability to perform tractable **exact inference** (i.e., without sampling) for a number of useful queries, such as the marginal probability of an
edge, or the Bayesian model averaged causal effect.

    @inproceedings{wang2022tractable,
    title={Tractable uncertainty for structure learning},
    author={Wang, Benjie and Wicker, Matthew R and Kwiatkowska, Marta},
    booktitle={International Conference on Machine Learning},
    pages={23131--23150},
    year={2022},
    organization={PMLR}
    }

## Installation Instructions

Install Poetry (python-poetry.org), then run the following commands from the base directory:

    poetry build
    poetry install

To run code, first activate the environment using

    poetry shell

Install torch-scatter in the virtual environment:

    pip install torch-scatter

## Getting Started

The best place to start is with the `example.ipynb` notebook in the `experiments` directory. This provides an example of the entire pipeline, including data generation, learning the OrderSPN, and then performing inference on the learned posterior distribution.

To reproduce the experimental results in the paper, use the `run_trust.py` script, with the appropriate options. For example, to reproduce the $d=32$ results using the Gadget oracle, run the following command from the root directory:

    python experiments/run_trust.py -d 32 -exp 32 8 2 6 2 -tb 1000

## Directory structure

```bash
└── experiments
└── oracle
│   └── dibs
│   └── gadget
└── trust
│   └── leaf_scores
│   └── learning
│   └── oracle
│   └── orderspn
│   └── utils
```

* experiments: Contains code and notebooks for running experiments
* oracle: Contains code for the oracle methods (dibs and gadget) used in the TRUST method. The code has been modified in order to implement the functionality necessary (c.f. READMEs in the respective directories for details)
* trust: Contains source code for TRUST.
    * leaf_scores: C++ code for efficient storage and computation of leaf distributions/scores
    * learning: Scripts for learning OrderSPNs
    * oracle: Wrappers for oracle methods
    * orderspn: Main code implementing OrderSPNs
    * utils: Miscellaneous utilities

## Contact
If you have any questions, please send an email to:

benjiewang at cs dot ucla dot edu

