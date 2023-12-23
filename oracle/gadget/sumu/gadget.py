"""The module implements the algorithm Gadget as first detailed in
:footcite:`viinikka:2020a`.
"""

import copy
import os
import sys
import time

import numpy as np
import scipy as sp

try:
    import plotext as plt

    plot_trace = plt.__version__ == "4.1.3"
except ImportError:
    plot_trace = False

from . import validate
from .candidates import candidate_parent_algorithm as cpa
from .data import Data
from .mcmc import MC3, PartitionMCMC
from .scorer import BDeu, BGe
from .stats import Stats, stats
from .utils.bitmap import bm, bm_to_ints, bm_to_np64
from .utils.io import read_candidates
from .utils.math_utils import comb, subsets
from .weight_sum import CandidateComplementScore, CandidateRestrictedScore


class Defaults:

    # default parameter values used by multiple classes

    def __init__(self):
        self.default = {
            "run_mode": {"name": "normal"},
            "mcmc": {
                "n_indep": 1,
                "iters": 320000,
                "burn_in": 0.5,
                "n_dags": 10000,
                "move_weights": [1, 1, 2],
            },
            "mc3": {"name": "linear", "M": 16},
            "score": lambda discrete: {"name": "bdeu", "params": {"ess": 10}}
            if discrete
            else {"name": "bge"},
            "prior": {"name": "fair"},
            "cons": {
                "max_id": -1,
                "K": lambda n: min(n - 1, 16),
                "d": lambda n: min(n - 1, 3),
                "pruning_eps": 0.001,
                "score_sum_eps": 0.1,
            },
            "candp": {
                "name": "greedy",
                "params": {"k": 6, "criterion": "score"},
            },
            "catc": {"tolerance": 2 ** -32, "cache_size": 10 ** 7},
            "logging": {
                "logfile": sys.stdout,
                "stats_period": 15,
                "tracefile": None,
                "overwrite": False,
            },
        }

    def __call__(self):
        return self.default

    def __getitem__(self, key):
        return self.default[key]


class GadgetParameters:
    def __init__(
        self,
        *,
        data,
        run_mode=dict(),
        mcmc=dict(),
        mc3=dict(),
        score=dict(),
        prior=dict(),
        cons=dict(),
        candp=dict(),
        catc=dict(),
        logging=dict(),
    ):
        # Save parameters initially given by user.
        # locals() has to be the first thing called in __init__.
        self.init = dict(**locals())
        del self.init["self"]
        del self.init["data"]

        self.data = Data(data)
        self.default = Defaults()()
        self.p = copy.deepcopy(self.init)

        self._populate_default_parameters()
        self._complete_user_given_parameters()
        self._validate_parameters()
        if self.p["run_mode"]["name"] == "normal":
            self._adjust_inconsistent_parameters()
        if self.p["run_mode"]["name"] == "budget":
            self._set_budget_based_parameters()
        if self.p["run_mode"]["name"] == "budget_mcmconly":  ## CHANGED
            self._set_budget_based_parameters_mcmconly()

    def _validate_parameters(self):
        # only validating possible user given candidate parents for now
        if "name" not in self.p["candp"] and "path" not in self.p["candp"]:
            validate.candidates(self.p["candp"])

    def _populate_default_parameters(self):
        # Some defaults are defined as functions of data.
        # Evaluate the functions here.
        self.default["cons"]["K"] = self.default["cons"]["K"](self.data.n)
        self.default["cons"]["d"] = self.default["cons"]["d"](self.data.n)
        self.default["score"] = self.default["score"](self.data.discrete)

    def _complete_user_given_parameters(self):
        for k in self.p:
            if (
                "name" in self.p[k]
                and self.p[k]["name"] != self.default[k]["name"]
            ):
                continue
            if validate.candidates_is_valid(self.p[k]):
                continue
            self.p[k] = dict(self.default[k], **self.p[k])
            for k2 in self.p[k]:
                if type(self.p[k][k2]) == dict:
                    self.p[k][k2] = dict(self.default[k][k2], **self.p[k][k2])

    def _adjust_inconsistent_parameters(self):
        iters = self.p["mcmc"]["iters"]
        M = self.p["mc3"].get("M", 1)
        burn_in = self.p["mcmc"]["burn_in"]
        n_dags = self.p["mcmc"]["n_dags"]
        self.p["mcmc"]["iters"] = iters // M * M
        self.p["mcmc"]["n_dags"] = min(
            (iters - int(iters * burn_in)) // M, n_dags
        )
        self.adjusted = (
            self.p["mcmc"]["iters"] != iters,
            self.p["mcmc"]["n_dags"] != n_dags,
        )

    def _set_budget_based_parameters(self):
        self.gb = GadgetBudget(self.data, self.p["run_mode"]["params"]["t"])
        params_to_predict = ["d", "K"]
        # dict of preset values for params if any
        preset_params = dict(
            i
            for i in ((k, self.init["cons"].get(k)) for k in params_to_predict)
            if i[1] is not None
        )

        # params needs to be copied because of the remove below
        for k in list(params_to_predict):
            if k in preset_params:
                self.p["cons"][k] = self.gb.get_and_pred(k, preset_params[k])
                params_to_predict.remove(k)
        for k in params_to_predict:
            self.p["cons"][k] = self.gb.get_and_pred(k)

        candp_is_given = self.init["candp"] != dict()
        candp_is_greedy = (
            candp_is_given
            and self.init["candp"]["name"] == Defaults()["candp"]["name"]
        )
        candp_params_is_given = (
            candp_is_given and "params" in self.init["candp"]
        )
        candp_k_is_preset = (
            candp_is_greedy
            and candp_params_is_given
            and "k" in self.init["candp"]["params"]
        )
        if candp_k_is_preset:
            self.gb.preset.add("candp")
        else:
            if not candp_params_is_given:
                self.p["candp"]["params"] = self.default["candp"]["params"]
            self.p["candp"]["params"]["t_budget"] = int(
                self.gb.budget["candp"]
            )

    def _set_budget_based_parameters_mcmconly(self):
        self.gb = GadgetBudget(self.data, self.p["run_mode"]["params"]["t"])

    def __getitem__(self, key):
        return self.p[key]


class GadgetBudget:
    """Class for predicting run times."""

    def __init__(
        self,
        data,
        t_budget,
        share={"candp": 1 / 9, "crs": 1 / 9, "ccs": 1 / 9, "mcmc": 2 / 3},
    ):
        # The preferred order is: predict d, predict K,
        # remaining precomp budget to candp.
        self.t0 = time.time()
        self.n = data.n
        self.data = data
        self.share = share
        # NOTE: After get_d and get_K the sum of budgets might exceed
        #       total, since remaining budget is adjusted upwards if previous
        #       phase is not predicted to use all of its budget.
        self.budget = dict()
        self.budget["total"] = t_budget
        self.predicted = {phase: 0 for phase in share}
        self.used = {phase: 0 for phase in share}
        self._not_done = set(share)
        self._not_done.remove("mcmc")
        self._update_precomp_budgets()
        self.preset = set()

    def _update_precomp_budgets(self):
        precomp_budget_left = self.budget["total"]
        precomp_budget_left *= 1 - self.share["mcmc"]
        precomp_budget_left -= sum(self.predicted.values()) - sum(
            self.used.values()
        )
        normalizer = sum(self.share[phase] for phase in self._not_done)
        for phase in self._not_done:
            self.budget[phase] = (
                self.share[phase] / normalizer * precomp_budget_left
            )

    def get_and_pred(self, param, preset_value=None):
        # if preset_value given only the time use is predicted
        if param == "d":
            return self.get_and_pred_d(preset_value)
        elif param == "K":
            return self.get_and_pred_K(preset_value)

    def get_and_pred_d(self, d_preset=None):
        phase = "ccs"
        t0 = time.time()
        K_max = 25
        C = {
            v: tuple([u for u in range(self.n) if u != v])
            for v in range(self.n)
        }
        C = {v: C[v][:K_max] for v in C}
        ls = LocalScore(data=self.data)
        d = 0
        t_d = 0
        t_budget = self.budget[phase]

        if d_preset is None:
            d_cond = (
                lambda: d < self.n - 1
                and t_d * comb(self.n, d + 1) / comb(self.n, d) * self.n
                < t_budget
            )
        else:
            self.preset.add("d")
            d_cond = lambda: d <= d_preset - 1

        while d_cond():
            d += 1
            t_d = time.time()
            # This does not take into acount time used by initializing
            # IntersectSums, as it seemed negligible.
            ls.complement_psets_and_scores(0, C, d)
            t_d = time.time() - t_d
        self.predicted["ccs"] = t_d * self.n
        self.used["ccs"] = time.time() - t0
        self._not_done.remove(phase)
        self._update_precomp_budgets()
        return d

    def get_and_pred_K(self, K_preset=None):
        phase = "crs"
        t0 = time.time()
        K_high = min(self.n - 1, 13)
        K_low = max(K_high - 8, 1)
        t_budget = self.budget[phase]
        X = np.zeros((K_high - K_low + 1, 3))
        i = 0
        for K in range(K_low, K_high + 1):
            params = {
                "cons": {"K": K},
                "candp": {"name": "rnd"},
                "logging": {"logfile": None},
            }
            g = Gadget(data=self.data, **params)
            g._find_candidate_parents()
            t_score = time.time()
            g._precompute_scores_for_all_candidate_psets()
            t_score = time.time() - t_score
            t = time.time()
            g._precompute_candidate_restricted_scoring()
            t = time.time() - t
            X[i] = np.array([1, K ** 2 * 2 ** K, t])
            i += 1
            # NOTE: It really should not be necessary to call this: Apparently
            #       some problem in how Cython calls (or does not call) the
            #       destructor. See CandidateRestrictedScore.cpp.
            g.c_r_score.reset_cout()

        t_score = t_score / 2 ** K_high
        a, b = np.linalg.lstsq(X[:, :-1], X[:, -1], rcond=None)[0]
        t_pred = 0
        K = K_high

        if K_preset is None:
            K_cond = lambda: K < self.n and t_pred < t_budget
        else:
            K_cond = lambda: K < K_preset + 1
            K = K_preset
            self.preset.add("K")

        while K_cond():
            K += 1
            t_pred = a + b * K ** 2 * 2 ** K + 2 ** K * t_score
        K -= 1

        self.predicted["crs"] = a + b * K ** 2 * 2 ** K + 2 ** K * t_score
        self.used["crs"] = time.time() - t0
        self._not_done.remove(phase)
        self._update_precomp_budgets()
        return K

    def left(self):
        return self.budget["total"] - (time.time() - self.t0)


class Logger:
    def __init__(self, *, logfile, mode="a", overwrite=False):

        # NOTE: mode needs to be "a" as otherwise CandidateRestrictedScore
        #       writing to same file breaks things. If overwrite is True
        #       the file needs to be first deleted then.

        self._silent = False
        # No output.
        if logfile is None:
            self._silent = True
            self._logfile = open(os.devnull, "w")
            self._logfilename = ""
        # Output to file.
        elif type(logfile) == str:
            if os.path.isfile(logfile):
                if not overwrite:
                    raise FileExistsError(f"{logfile} exists.")
                else:
                    os.remove(logfile)
            self._logfile = open(logfile, mode)
            self._logfilename = self._logfile.name
        # Output to stdout.
        else:
            self._logfile = logfile
            self._logfilename = ""

    def __call__(self, string):
        print(string, file=self._logfile)
        self._logfile.flush()

    def dict(self, data):
        def pretty_dict(d, n=0, string=""):
            for k in d:
                if type(d[k]) in (dict, Stats):
                    string += f"{' '*n}{k}\n"
                else:
                    string += f"{' '*n}{k}: {d[k]}\n"
                if type(d[k]) in (dict, Stats):
                    string += pretty_dict(d[k], n=n + 2)
            return string

        print(pretty_dict(data), file=self._logfile)
        self._logfile.flush()

    def numpy(self, array, fmt="%.2f"):
        np.savetxt(self._logfile, array, fmt=fmt)
        self._logfile.flush()

    def br(self, n=1):
        print("\n" * (n - 1), file=self._logfile)
        self._logfile.flush()


class GadgetLogger(Logger):
    """Stuff for printing stuff."""

    def __init__(self, gadget):
        super().__init__(
            logfile=gadget.p["logging"]["logfile"],
            overwrite=gadget.p["logging"]["overwrite"],
        )
        self._running_sec_num = 0
        self._linewidth = 80  # max(80, 12 + 6 * gadget.p["mc3"]["M"] - 1)
        self.g = gadget

    def h(self, title):
        self._running_sec_num += 1
        end = "." * (self._linewidth - len(title) - 4)
        title = f"{self._running_sec_num}. {title} {end}"
        print(title, file=self._logfile)
        self.br()
        self._logfile.flush()

    def periodic_stats(self, header=False):
        msg_tmpl = "{:<12.12}" + " {:<5.5}" * self.g.p["mc3"]["M"]
        temps = [1.0]
        temps_labels = [1.0]
        if self.g.p["mc3"]["M"] > 1:
            temps = sorted(list(stats["mcmc"].keys()), reverse=True)
            temps_labels = [round(t, 2) for t in temps]
        moves = stats["mcmc"][1.0].keys()

        def print_stats_title():
            msg = "Periodic statistics on:\n"
            msg += (
                "1. Cumulative acceptance probability by move "
                "and inverse temperature.\n"
            )
            if plot_trace:
                msg += (
                    "2. Root-partition score traces "
                    "for each independent chain.\n"
                )
            else:
                msg += (
                    "2. Last root-partition score "
                    "for each independent chain.\n"
                )
            print(msg, file=self._logfile)
            self._logfile.flush()

        if header:
            print_stats_title()

        msg = msg_tmpl.format("move", *temps_labels)
        msg += "\n" + "-" * self._linewidth
        print(msg, file=self._logfile)

        for m in moves:
            ar = [stats["mcmc"][i][m]["accept_ratio"] for i in temps]
            ar = [round(r, 2) if type(r) == float else "" for r in ar]
            msg = msg_tmpl.format(m, *ar)
            print(msg, file=self._logfile)
        if self.g.p["mc3"]["M"] > 1:
            ar = stats["mc3"]["accept_ratio"]
            ar = [round(r, 2) if not np.isnan(r) else "" for r in ar] + [""]
            msg = msg_tmpl.format("MC^3", *ar)
            print(msg, file=self._logfile)
        print(file=self._logfile)
        self._logfile.flush()

    def run_stats(self):
        w_iters = str(max(len("iters"), len(str(stats["iters"]["total"]))) + 2)
        w_seconds = str(len(str(int(stats["t"]["mcmc"]))) + 2)
        msg_title_tmpl = (
            "{:<20}{:<" + w_iters + "}{:<13}{:<" + w_seconds + "}{:<9}{:<13}"
        )
        msg_tmpl = (
            "{:<20}{:<"
            + w_iters
            + "}{:<13.3}{:<"
            + w_seconds
            + "}{:<9.3}{:<13.3}"
        )
        msg = (
            msg_title_tmpl.format(
                "phase", "iters", "iters/total", "s", "s/total", "s/iter"
            )
            + "\n"
        )
        msg += "-" * self._linewidth + "\n"

        phases = ["burn-in", "after burn-in"]
        if self.g.p["mc3"]["name"] == "adaptive":
            phases = ["adaptive tempering"] + phases

        for phase in phases:
            msg += (
                msg_tmpl.format(
                    phase,
                    stats["iters"][phase],
                    stats["iters"][phase] / stats["iters"]["total"],
                    round(stats["t"][phase]),
                    stats["t"][phase] / stats["t"]["mcmc"],
                    (
                        stats["t"][phase] / stats["iters"][phase]
                        if stats["iters"][phase] > 0
                        else "-"
                    ),
                )
                + "\n"
            )
        msg += msg_tmpl.format(
            "mcmc total",
            stats["iters"]["total"],
            1.0,
            round(stats["t"]["mcmc"]),
            1.0,
            stats["t"]["mcmc"] / stats["iters"]["total"],
        )
        print(msg, file=self._logfile)
        print(file=self._logfile)
        self._logfile.flush()

    def progress(self, t, t_elapsed):
        if self.g.p["run_mode"]["name"] == "normal":
            progress = round(
                100 * t / (self.g.p["mcmc"]["iters"] // self.g.p["mc3"]["M"])
            )
            progress = str(progress)
            print(
                f"Progress: {progress}% ({t*self.g.p['mc3']['M']} iterations)",
                file=self._logfile,
            )
            self._logfile.flush()
        elif self.g.p["run_mode"]["name"] == "budget":
            progress = round(100 * t_elapsed / self.g.p.gb.budget["mcmc"])
            progress = str(progress)
            print(
                f"Progress: {progress}% ({t*self.g.p['mc3']['M']} iterations)",
                file=self._logfile,
            )
        elif self.g.p["run_mode"]["name"] == "anytime":
            print(
                f"Progress: {t*self.g.p['mc3']['M']} iterations",
                file=self._logfile,
            )

    def r_scores(self, t, R_scores):
        msg = "Last root-partition scores: " + " ".join(
            str(int(score)) for score in R_scores[t % 1000]
        )
        print(msg, file=self._logfile)
        self.br()
        self._logfile.flush()

    def plot_score_trace(self, t, M, R_scores):

        r = R_scores.shape[0]  # 1000
        plt.clear_plot()
        for i in range(self.g.p["mcmc"]["n_indep"]):
            if t < r:
                plt.scatter(
                    R_scores[:t, i], label=str(i), color=i + 1, marker="•"
                )
            else:
                plt.scatter(
                    R_scores[np.r_[(t % r) : r, 0 : (t % r)], i],
                    label=str(i),
                    color=i + 1,
                    marker="dot",
                )
        plt.plotsize(80, 20)
        plt.yfrequency(4)
        if t < 1000:
            xticks = [int(w * t) for w in np.arange(0, 1 + 1 / 3, 1 / 3)]
            xlabels = [str(round(x * M / 1000, 1)) + "k" for x in xticks]
        else:
            xticks = np.array([0, 333, 666, 999])
            xlabels = [
                str(round((x + t) * M / 1000, 1)) + "k"
                for x in -1 * xticks[::-1]
            ]
        plt.xticks(xticks, xlabels)
        plt.canvas_color("default")
        plt.axes_color("default")
        plt.ticks_color("default")
        print(plt.build(), file=self._logfile)
        print(file=self._logfile)
        self._logfile.flush()


class Gadget:
    """Class implementing the Gadget pipeline for MCMC sampling from
    the structure posterior of DAG models. The user interface consists
    of:

    1. The constructor for setting all the parameters.
    2. :py:meth:`.sample()` method which runs the MCMC chain and
       returns the sampled DAGs and their scores.

    All the constructor arguments are keyword arguments, i.e., the
    **data** argument should be given as ``data=data``, etc. Only the
    data argument is required; other arguments have some more or less
    sensible defaults.

    There is a lot of parameters that can be adjusted. To make
    managing the parameters easier, they are grouped into dict-objects
    around some common theme, except the **data** argument which
    accepts any valid constructor argument for a :py:class:`.Data`
    object.

    The (nested) lists in the following description reflect the
    structure of the dict objects. For example, to set the equivalent
    sample size for BDeu score to some value :math:`a`, you should
    construct the object as

    >>> Gadget(data=data, score={"name": "bdeu", "params": {"ess": a}}).

    To only adjust some parameter within a dict-argument while keeping
    the others at default, it suffices to set the one parameter. For
    example, to set the number of candidate parents :math:`K` to some
    value :math:`k`, you should construct the object as

    >>> Gadget(data=data, cons={"K": k}).

    In this documentation nested parameters are referenced as
    **outer:inner**, e.g., the ``ess`` parameter can be referenced as
    **score:params:ess**.

    - **run_mode**: Which mode to run Gadget in.

      - **name**: Name of the mode: ``normal``, ``budget`` or ``anytime``.

        - **Default**: ``normal``.

        ``normal``: All parameters are set manually.

        ``budget``: Gadget is run until a given time budget is used
        up. **cons:K**, **cons:d**, **mcmc:iters** and **candp** are set
        automatically, so that approximately one third of the budget is
        used on precomputations and the rest on MCMC sampling. The
        precomputation budget is split between

        - (1) finding candidate parents;
        - (2) precomputing candidate restricted scoring structures;
        - (3) precomputing complementary scoring structures.

        The time required by the third phase is factorial in **cons:d**
        (there are approximately :math:`\\binom{n}{d}` scores complementary
        to those restricted to candidate parents), so the amount of
        additional time required going from :math:`d` to :math:`d+1` can be
        very large. Therefore, as a first step :math:`d` is set to a value
        with which phase (3) is predicted to use at most :math:`1/3` of the
        precomputation budget (i.e., :math:`1/9` of the total). Then the
        remaining precomputation budget is adjusted to be the original
        subtracted by the predicted time use for phase (3) and the (small)
        amount of time required for the prediction itself.

        As a second step **cons:K** is set to a value with which phase (2)
        is predicted to use at most :math:`1/2` of the remaining
        precomputation budget. Again, the predicted time use and the amount
        of time required for the prediction of this phase is subtracted
        from the remaining precomputation budget.

        Then, the candidate parent selection algorithm (**candp**) is set
        to ``greedy-lite``, and its parameter :math:`k` is dynamically set
        during the running of the algorithm to a value for which
        :math:`k-1` is predicted to overuse the remaining precomputation
        budget.

        Finally, the MCMC phase uses the amount of budget that remains. The
        **mcmc:burn_in** parameter in this mode sets fraction of *time* to
        be used on the burn-in phase, rather than the fraction of
        iterations.

        Overrides **mcmc:iters**, **cons:K**, **cons:d** and **candp**.

        - **params**:

          - **t**: The time budget in seconds.

        ``budget_mcmconly``: Similar to ``budget``, but used in conjunction with
        using_precomp_score to set a budget only for MCMC (no precomputation).

        - **params**:

          - **t**: The time budget in seconds.

        ``anytime``: If ran in this mode the first CTRL-C after calling
        sample() stops the burn-in phase and starts sampling DAGs, and the
        second CTRL-C stops the sampling. DAG sampling first accumulates up to
        2 * **mcmc**:**n_dags** - 1 DAGs with thinning 1 (i.e., a DAG is
        sampled for each sampled root-partition), then each time the number of
        DAGs reaches 2 x **mcmc**:**n_dags** the thinning is doubled and every
        2nd already sampled DAG is deleted. Overrides **mcmc**:**iters** and
        **mcmc**:**burn_in**.

    - **mcmc**: General Markov Chain Monte Carlo arguments.

      - **n_indep**: Number of independent chains to run (each multiplied by
        **mc3**).  DAGs are sampled evenly from each.

        **Default**: 4.

      - **iters**: The total number of iterations across all the Metropolis
        coupled chains, i.e., if the number of coupled chains is :math:`k`
        then each runs for **iters/k** iterations. If the given **iters**
        is not a multiple of the number of chains it is adjusted downwards.

        **Default**: 320000.

      - **mc3**: The number of of Metropolis coupled chains. The
        temperatures of the chains are spread evenly between uniform
        and the target distribution.

        **Default**: 16.

      - **burn_in**: Ratio of how much of the iterations to use for burn-in
        (0.5 is 50%).

        **Default**: 0.5.

      - **n_dags**: Number of DAGs to sample. The maximum number of
        DAGs that can be sampled is **iters/mc3*(1-burn_in)**; if the given
        **n_dags** is higher than the maximum, it is adjusted
        downwards.

        **Default**: 10000.

    - **score**: The score to use.

      - **name**: Name of the score.

        **Default**: ``bdeu`` (i.e., Bayesian Dirichlet equivalent
        uniform) for discrete data, and ``bge`` (i.e., Bayesian
        Gaussian equivalent) for continuous data.

      - **params**: A dict of parameters for the score.

        **Default**: ``{"ess": 10}`` for ``bdeu``.

    - **prior**: Modular structure prior to use.

      - **name**: Structure prior: *fair* or *unif*
        :footcite:`eggeling:2019`.

        **Default**: fair.

    - **cons**: Constraints on the explored DAG space.

      - **K**: Number of candidate parents per node.

        **Default**: :math:`\min(n-1, 16)`, where :math:`n` is the number
        of nodes.

      - **d**: Maximum size of parent sets that are not subsets of the
        candidate parents.

        **Default**: :math:`\min(n-1, 3)`, where :math:`n` is the number of
        nodes.

      - **max_id**: Maximum size of parent sets that are subsets of
        candidates. There should be no reason to change this from
        the default.

        **Default**: -1, i.e., unlimited.

      - **pruning_eps**: Allowed relative error for a root-partition
        node score sum. Setting this to some value :math:`>0` allows
        some candidate parent sets to be pruned, expediting parent
        set sampling.

        **Default**: 0.001.

      - **score_sum_eps**: Tolerated relative error when computing
        score sums from parent sets that are not subsets of the
        candidate parents.

        **Default**: 0.1.

    - **candp**: Algorithm to use for finding candidate parents.

      - **name**: Name of the algorithm.

        **Default**: ``greedy-lite``.

      - **params**: A dict of parameters for the algorithm.

        **Default**: ``{"k": 6}``. The default algorithm
        :py:func:`~sumu.candidates.greedy_lite` has one parameter,
        :math:`k`, determining the number of parents to add during
        the last iteration of the algorithm. The candidate selection
        phase can be made faster by incrementing this value.

      - **path**: Path to precomputed file storing the candidate
        parents. The format is such that the row number determines the
        node in question, and on each row there are the :math:`K`
        space separated candidate parents. If path is given no
        computations are done.

        **Default**: ``None``.

    - **catc**: Parameters determining how catastrofic cancellations are
      handled. Catastrofic cancellation occurs when a score sum
      :math:`\\tau_i(U,T)` computed as :math:`\\tau_i(U) - \\tau_i(U
      \setminus T)` evaluates to zero due to numerical reasons.

      - **tolerance**: how small should the absolute difference
        between two log score sums be in order for the subtraction
        to be determined to lead to catastrofic cancellation.

        **Default**: :math:`2^{-32}`.

      - **cache_size**: Maximum amount of score sums that cannot be
        computed through subtraction to be stored separately. If there is a
        lot of catastrofic cancellations, setting this value high can
        have a big impact on memory use.

        **Default**: :math:`10^7`

    - **logging**: Parameters determining the logging output during
      running of the sampler.

      - **stats_period**: Interval in seconds for printing more statistics.

        **Default**: 15.

      - **logfile**: File path to print the output to. To suppress all
        output set this to ``None``.

        **Default**: ``sys.stdout``.

      - **tracefile**: File path to write the root-partition scores to, of
        each independent chain, for analyzing mixing and convergence.

        **Default**: ``None``.

      - **overwrite**: If ``True`` both **logfile** and **tracefile** are
        overwritten if they exist.

        **Default**: ``False``.

    - **trust** Custom parameters for implementation of s1/s2 functionality, as oracle for TRUST. If specified,
    must provide precomputed scores/candidate parents. Instead of just returning DAGs, partitions of the variables
    will be returned.

      - **s1**: List of variables corresponding to s1
      - **s2**: List of variables corresponding to s2
      - **d**: Dimension of full graph (for conversion purposes)
      - **sample**: Whether to sample graphs or just return partitions
      - **score**: Score object representing precomputed scores. relative -> absolute
      - **candidates**: Candidate parents. relative -> relative for s2

    """

    def __init__(
        self,
        *,
        data,
        run_mode=dict(),
        mcmc=dict(),
        mc3=dict(),
        score=dict(),
        prior=dict(),
        cons=dict(),
        candp=dict(),
        catc=dict(),
        logging=dict(),
        trust=dict(),
    ):

        # locals() has to be the first thing called in __init__.
        user_given_parameters = locals()

        if trust:
            self.s2 = trust["s2"]
            self.s1 = trust["s1"]
            self.full_d = trust["d"]
            self.custom_return_dags = trust["sample"]
            self.precomp_score = trust["score"]
            self.using_precomp_score = True
            self.candidates = trust["candidates"]
            self.init_partition = trust["init_part"]

            self.C = self.candidates

            del user_given_parameters["trust"]
            del user_given_parameters["self"]
            self.p = GadgetParameters(**user_given_parameters)
            self.data = self.p.data
        else:
            self.using_precomp_score = False
            del user_given_parameters["trust"]
            del user_given_parameters["self"]
            self.p = GadgetParameters(**user_given_parameters)
            self.data = self.p.data


        self.log = GadgetLogger(self)
        log = self.log
        self.trace = Logger(
            logfile=self.p["logging"]["tracefile"],
            overwrite=self.p["logging"]["overwrite"],
        )

        log.h("PROBLEM INSTANCE")
        log.dict(self.data.info)
        log.br()

        log.h("RUN PARAMETERS")
        log.dict(self.p.p)
        if self.p["run_mode"]["name"] == "normal":
            if any(self.p.adjusted):
                log("WARNING")
            if self.p.adjusted[0]:
                log("iters adjusted downwards: needs to be multiple of mc3.")
            if self.p.adjusted[1]:
                log(
                    "n_dags adjusted downwards: "
                    "max is (iters * (1 - burn_in)) / mc3."
                )
            if any(self.p.adjusted):
                log.br()
        log.br()


    def return_cand_parents_and_score(self):
        """Performs the precomputation step and returns candidate parents and scores for each node. Does not continue
        with MCMC sampling.

        Returns:
            C: set of candidate parents for each node
            combined_score: Score object containing all information
            score_array: raw array of scores
        """
        log = self.log

        log.h("FINDING CANDIDATE PARENTS")
        stats["t"]["C"] = time.time()
        self._find_candidate_parents()
        stats["t"]["C"] = time.time() - stats["t"]["C"]
        log.numpy(self.C_array, "%i")
        if (
                self.p["run_mode"]["name"] == "budget"
                and "candp" not in self.p.gb.preset
                and self.p["candp"]["name"] == Defaults()["candp"]["name"]
        ):
            log.br()
            log(f"Adjusted for time budget: k = {stats['C']['k']}")
            log(
                "time budgeted: "
                f"{round(self.p['candp']['params']['t_budget'])}s"
            )
        log.br()
        log(f"time used: {round(stats['t']['C'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS")
        stats["t"]["crscore"] = time.time()
        self._precompute_scores_for_all_candidate_psets()
        # saving score array
        score_array = np.copy(self.score_array)
        self._precompute_candidate_restricted_scoring()
        stats["t"]["crscore"] = time.time() - stats["t"]["crscore"]
        if self.p["run_mode"]["name"] == "budget":
            log(f"time predicted: {round(self.p.gb.predicted['crs'])}s")
        log(f"time used: {round(stats['t']['crscore'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS")
        stats["t"]["ccscore"] = time.time()
        self._precompute_candidate_complement_scoring()
        stats["t"]["ccscore"] = time.time() - stats["t"]["ccscore"]
        if self.p["run_mode"]["name"] == "budget":
            log(f"time predicted: {round(self.p.gb.predicted['ccs'])}s")
        log(f"time used: {round(stats['t']['ccscore'])}s")
        log.br(2)

        combined_score = Score(
                C=self.C, c_r_score=self.c_r_score, c_c_score=self.c_c_score
            )

        return self.C, combined_score, score_array

    def sample(self):

        log = self.log

        # skip precomputation if precomputed scores already provided
        if not self.using_precomp_score:
            log.h("FINDING CANDIDATE PARENTS")
            stats["t"]["C"] = time.time()
            self._find_candidate_parents()
            stats["t"]["C"] = time.time() - stats["t"]["C"]
            log.numpy(self.C_array, "%i")
            if (
                self.p["run_mode"]["name"] == "budget"
                and "candp" not in self.p.gb.preset
                and self.p["candp"]["name"] == Defaults()["candp"]["name"]
            ):
                log.br()
                log(f"Adjusted for time budget: k = {stats['C']['k']}")
                log(
                    "time budgeted: "
                    f"{round(self.p['candp']['params']['t_budget'])}s"
                )
            log.br()
            log(f"time used: {round(stats['t']['C'])}s")
            log.br(2)

            log.h("PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS")
            stats["t"]["crscore"] = time.time()
            self._precompute_scores_for_all_candidate_psets()
            self._precompute_candidate_restricted_scoring()
            stats["t"]["crscore"] = time.time() - stats["t"]["crscore"]
            if self.p["run_mode"]["name"] == "budget":
                log(f"time predicted: {round(self.p.gb.predicted['crs'])}s")
            log(f"time used: {round(stats['t']['crscore'])}s")
            log.br(2)

            log.h("PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS")
            stats["t"]["ccscore"] = time.time()
            self._precompute_candidate_complement_scoring()
            stats["t"]["ccscore"] = time.time() - stats["t"]["ccscore"]
            if self.p["run_mode"]["name"] == "budget":
                log(f"time predicted: {round(self.p.gb.predicted['ccs'])}s")
            log(f"time used: {round(stats['t']['ccscore'])}s")
            log.br(2)

        log.h("RUNNING MCMC")
        stats["t"]["mcmc"] = time.time()
        self._mcmc_init() ##CHANGED##
        stats["t"]["adaptive tempering"] = time.time() - stats["t"]["mcmc"]
        if self.p["run_mode"]["name"] == "anytime":
            self._mcmc_run_anytime()
        else:
            if self.p["mc3"]["name"] == "adaptive":
                self._mcmc_run(t_elapsed_init=stats["t"]["adaptive tempering"])
            else:
                self._mcmc_run()
        stats["t"]["mcmc"] = time.time() - stats["t"]["mcmc"]
        log(f"time used: {round(stats['t']['mcmc'])}s")
        log.br(2)

        log.h("RUN STATISTICS")
        log.run_stats()
        log(f"no. dags sampled: {len(self.sampled_partitions)}")

        if self.using_precomp_score:
            if self.custom_return_dags:
                return self.sampled_partitions, self.dags
            else:
                return self.sampled_partitions

        chain_info = dict()
        for i, c_i in enumerate(self.mcmc):
            if self.p["mc3"]["M"] > 1:
                chain_info[i] = {
                    "inv_temperatures": [c.inv_temp for c in c_i.chains]
                }

        return self.dags, dict(
            parameters=self.p.p,
            scores=self.dag_scores,
            candidates=self.C,
            chains=chain_info,
            stats=stats,
        )

    def _find_candidate_parents(self):
        self.l_score = LocalScore(
            data=self.data,
            score=self.p["score"],
            maxid=self.p["cons"]["max_id"],
        )

        if "path" in self.p["candp"]:
            self.C = read_candidates(self.p["candp"]["path"])

        elif "name" in self.p["candp"]:
            self.C, stats["C"] = cpa[self.p["candp"]["name"]](
                self.p["cons"]["K"],
                scores=self.l_score,
                data=self.data,
                params=self.p["candp"].get("params"),
            )
        else:
            self.C = self.p["candp"]

        self.C_array = np.empty(
            (self.data.n, self.p["cons"]["K"]), dtype=np.int32
        )

        for v in self.C:
            self.C_array[v] = np.array(self.C[v])

    def _precompute_scores_for_all_candidate_psets(self):
        self.score_array = self.l_score.candidate_scores(self.C_array)

    def _precompute_candidate_restricted_scoring(self):
        self.c_r_score = CandidateRestrictedScore(
            score_array=self.score_array,
            C=self.C_array,
            K=self.p["cons"]["K"],
            cc_tolerance=self.p["catc"]["tolerance"],
            cc_cache_size=self.p["catc"]["cache_size"],
            pruning_eps=self.p["cons"]["pruning_eps"],
            logfile=self.log._logfilename,
            silent=self.log._silent,
        )
        del self.score_array

    def _precompute_candidate_complement_scoring(self):
        self.c_c_score = None
        if self.p["cons"]["K"] < self.data.n - 1 and self.p["cons"]["d"] > 0:
            # NOTE: CandidateComplementScore gives error if K = n-1,
            #       and is unnecessary.
            # NOTE: Does this really need to be reinitialized?
            self.l_score = LocalScore(
                data=self.data,
                score=self.p["score"],
                maxid=self.p["cons"]["d"],
            )
            self.c_c_score = CandidateComplementScore(
                localscore=self.l_score,
                C=self.C,
                d=self.p["cons"]["d"],
                eps=self.p["cons"]["score_sum_eps"],
            )
            del self.l_score

    def _mcmc_init(self):

        if self.using_precomp_score:
            self.score = self.precomp_score
        else:
            self.score = Score(
                C=self.C, c_r_score=self.c_r_score, c_c_score=self.c_c_score
            )

        self.mcmc = list()

        R = None
        if self.using_precomp_score:
            R = self.init_partition
        for i in range(self.p["mcmc"]["n_indep"]):
            if self.p["mc3"]["name"] == "adaptive":
                self.log("Adaptive tempering")
                self.mcmc.append(
                    MC3.adaptive(
                        PartitionMCMC(
                            self.C,
                            self.score,
                            self.p["cons"]["d"],
                            move_weights=self.p["mcmc"]["move_weights"],
                            R=R
                        ),
                        stats=stats,
                        log=self.log,
                    )
                )
                self.p["mc3"]["M"] = len(self.mcmc[0].chains)
                self.log.br()

            elif self.p["mc3"]["M"] == 1:
                self.mcmc.append(
                    PartitionMCMC(
                        self.C,
                        self.score,
                        self.p["cons"]["d"],
                        move_weights=self.p["mcmc"]["move_weights"],
                        stats=stats,
                        R=R
                    )
                )

            elif self.p["mc3"]["M"] > 1:
                inv_temps = MC3.get_inv_temperatures(
                    self.p["mc3"]["name"], self.p["mc3"]["M"]
                )
                self.mcmc.append(
                    MC3(
                        [
                            PartitionMCMC(
                                self.C,
                                self.score,
                                self.p["cons"]["d"],
                                inv_temp=inv_temps[i],
                                move_weights=self.p["mcmc"]["move_weights"],
                                stats=stats,
                                R=R
                            )
                            for i in range(self.p["mc3"]["M"])
                        ],
                        stats=stats,
                    )
                )

    def _mcmc_run(self, t_elapsed_init=0):

        r = 1000  # max number of iterations to plot in score trace

        self.dags = list()
        self.dag_scores = list()

        if self.using_precomp_score:
            self.sampled_partitions = list()

        R_scores = np.zeros((r, self.p["mcmc"]["n_indep"]))

        timer = time.time()
        first = True

        if self.p["run_mode"]["name"] == "normal":
            iters_burn_in = int(
                self.p["mcmc"]["iters"]
                / self.p["mc3"]["M"]
                * self.p["mcmc"]["burn_in"]
            )
            iters_burn_in = int(iters_burn_in)
            iters_dag_sampling = (
                self.p["mcmc"]["iters"] // self.p["mc3"]["M"] - iters_burn_in
            )
            if self.p["mc3"]["name"] == "adaptive":
                iters_burn_in -= int(
                    stats["iters"]["adaptive tempering"] / self.p["mc3"]["M"]
                )
            burn_in_cond = lambda: t < iters_burn_in
            mcmc_cond = lambda: t < iters_dag_sampling
            dag_sample_cond = (
                lambda: t
                >= iters_dag_sampling / self.p["mcmc"]["n_dags"] * dag_count
            )

        elif self.p["run_mode"]["name"] == "budget" or self.p["run_mode"]["name"] == "budget_mcmconly": ## CHANGED
            self.p.gb.budget["mcmc"] = self.p.gb.left()
            t_b_burnin = self.p.gb.budget["mcmc"] * self.p["mcmc"]["burn_in"]
            burn_in_cond = lambda: t_elapsed < t_b_burnin
            mcmc_cond = (
                lambda: dag_count < self.p["mcmc"]["n_dags"]
                and t_elapsed < t_b_mcmc
            )
            dag_sample_cond = lambda: dag_count < t_elapsed / t_per_dag

        t = 0
        t_elapsed = t_elapsed_init
        t0 = time.time()
        while burn_in_cond():
            for i in range(self.p["mcmc"]["n_indep"]):
                R, R_score = self.mcmc[i].sample()
                R_scores[t % r, i] = R_score
            if t > 0 and t % (r - 1) == 0:
                self.trace.numpy(R_scores)
            if time.time() - timer > self.p["logging"]["stats_period"]:
                timer = time.time()
                self.log.periodic_stats(first)
                self.log.progress(t, time.time() - t0)
                if plot_trace:
                    self.log.br()
                    self.log.plot_score_trace(t, self.p["mc3"]["M"], R_scores)
                else:
                    self.log.r_scores(t, R_scores)
                first = False

            t += 1
            t_elapsed = t_elapsed_init + time.time() - t0

        stats["t"]["burn-in"] = time.time() - t0
        if self.p["run_mode"]["name"] == "budget" or self.p["run_mode"]["name"] == "budget_mcmconly": ## CHANGED
            t_b_mcmc = self.p["run_mode"]["params"]["t"] - (
                time.time() - self.p.gb.t0
            )
            t_per_dag = t_b_mcmc / self.p["mcmc"]["n_dags"]

        self.log("Sampling DAGs...")
        self.log.br(2)

        dag_count = 0
        iters_burn_in = t
        t = 0
        t_elapsed = 0
        t0 = time.time()
        while mcmc_cond():
            if time.time() - timer > self.p["logging"]["stats_period"]:
                timer = time.time()
                self.log.periodic_stats(first)
                self.log.progress(
                    t + iters_burn_in,
                    time.time() - t0 + stats["t"]["burn-in"],
                )
                if plot_trace:
                    self.log.br()
                    self.log.plot_score_trace(
                        t + iters_burn_in, self.p["mc3"]["M"], R_scores
                    )
                else:
                    self.log.r_scores(t + iters_burn_in, R_scores)
                first = False
            if dag_sample_cond():
                for i in range(self.p["mcmc"]["n_indep"]):
                    dag_count += 1
                    R, R_score = self.mcmc[i].sample()
                    R_scores[(t + iters_burn_in) % 1000, i] = R_score
                    if ((not self.using_precomp_score) or self.custom_return_dags):
                        dag, score = self.score.sample_DAG(R)
                        self.dags.append(dag)
                        self.dag_scores.append(score)
                    if self.using_precomp_score:
                        self.sampled_partitions.append(R)
            else:
                for i in range(self.p["mcmc"]["n_indep"]):
                    R, R_score = self.mcmc[i].sample()
                    R_scores[(t + iters_burn_in) % r, i] = R_score
            if t > 0 and t % (r - 1) == 0:
                self.trace.numpy(R_scores)

            t += 1
            t_elapsed = time.time() - t0
        stats["t"]["after burn-in"] = t_elapsed

        self.log.periodic_stats(first)

        stats["iters"]["burn-in"] = iters_burn_in * self.p["mc3"]["M"]
        stats["iters"]["after burn-in"] = t * self.p["mc3"]["M"]
        stats["iters"]["total"] = (iters_burn_in + t) * self.p["mc3"]["M"]
        if "adaptive tempering" in stats["iters"]:
            stats["iters"]["total"] += stats["iters"]["adaptive tempering"]

    def _mcmc_run_anytime(self):

        r = 1000  # max number of iterations to plot in score trace

        self.dags = list()
        self.dag_scores = list()

        R_scores = np.zeros((r, self.p["mcmc"]["n_indep"]))

        timer = time.time()
        t0 = timer
        first = True

        try:
            t_b = -1
            while True:
                t_b += 1
                for i in range(self.p["mcmc"]["n_indep"]):
                    R, R_score = self.mcmc[i].sample()
                    R_scores[t_b % r, i] = R_score
                if t_b > 0 and t_b % (r - 1) == 0:
                    self.trace.numpy(R_scores)
                if time.time() - timer > self.p["logging"]["stats_period"]:
                    timer = time.time()
                    self.log.periodic_stats(first)
                    self.log.progress(t_b, 0)
                    if plot_trace:
                        self.log.br()
                        self.log.plot_score_trace(t_b, R_scores)
                    else:
                        self.log.r_scores(t_b, R_scores)
                    first = False
        except KeyboardInterrupt:
            stats["t"]["burn-in"] = time.time() - t0
            stats["iters"]["burn-in"] = t_b

        self.log("Sampling DAGs...")
        self.log.br(2)

        try:
            t0 = time.time()
            thinning = 1
            dag_count = 0
            t = -1
            while True:
                t += 1
                if time.time() - timer > self.p["logging"]["stats_period"]:
                    timer = time.time()
                    self.log.periodic_stats(first)
                    self.log.progress(t_b + t, 0)
                    if plot_trace:
                        self.log.br()
                        self.log.plot_score_trace(t + t_b, R_scores)
                    else:
                        self.log.r_scores(t + t_b, R_scores)
                    first = False
                    msg = "{} DAGs with thinning {}."
                    self.log(msg.format(len(self.dags), thinning))
                    self.log.br()
                if t > 0 and t % thinning == 0:
                    for i in range(self.p["mcmc"]["n_indep"]):
                        dag_count += 1
                        R, R_score = self.mcmc[i].sample()
                        R_scores[(t + t_b) % 1000, i] = R_score
                        if ((not self.using_precomp_score) or self.custom_return_dags):
                            dag, score = self.score.sample_DAG(R)
                            self.dags.append(dag)
                            self.dag_scores.append(score)
                            if dag_count == 2 * self.p["mcmc"]["n_dags"]:
                                self.dags = self.dags[0::2]
                                dag_count = len(self.dags)
                                thinning *= 2
                        if self.using_precomp_score:
                            self.sampled_partitions.append(R)
                else:
                    for i in range(self.p["mcmc"]["n_indep"]):
                        R, R_score = self.mcmc[i].sample()
                        R_scores[(t + t_b) % r, i] = R_score

                if t % (r - 1) == 0:
                    self.trace.numpy(R_scores)

        except KeyboardInterrupt:
            stats["t"]["after burn-in"] = time.time() - t0
            stats["iters"]["after burn-in"] = t
            stats["iters"]["total"] = (
                stats["iters"]["burn-in"] + stats["iters"]["after burn-in"]
            )

        if first:
            self.log.periodic_stats(first)


class LocalScore:
    """Class for computing local scores given input data.

    Implemented scores are BDeu and BGe. The scores by default use the "fair"
    modular structure prior :footcite:`eggeling:2019`.

    """

    def __init__(
        self,
        *,
        data,
        score=None,
        prior=Defaults()["prior"],
        maxid=Defaults()["cons"]["max_id"],
    ):
        self.data = Data(data)
        self.score = score
        if score is None:
            self.score = Defaults()["score"](self.data.discrete)
        self.prior = prior
        self.priorf = {"fair": self._prior_fair, "unif": self._prior_unif}
        self.maxid = maxid
        self._precompute_prior()

        if self.data.N == 0:
            self.scorer = EmptyDataScore()

        elif self.score["name"] == "bdeu":
            self.scorer = BDeu(
                data=self.data.data,
                maxid=self.maxid,
                ess=self.score["params"]["ess"],
            )

        elif self.score["name"] == "bge":
            self.scorer = BGe(data=self.data, maxid=self.maxid)

        self.t_scorer = 0
        self.t_prior = 0

    def _prior_fair(self, indegree):
        return self._prior[indegree]

    def _prior_unif(self, indegree):
        return 0

    def _precompute_prior(self):
        if self.prior["name"] == "fair":
            self._prior = np.zeros(self.data.n)
            self._prior = -np.array(
                list(
                    map(
                        np.log,
                        [
                            float(comb(self.data.n - 1, k))
                            for k in range(self.data.n)
                        ],
                    )
                )
            )

    def local(self, v, pset):
        """Local score for input node v and pset, with score function
        self.scoref.

        This is the "safe" version, raising error if queried with invalid
        input.  The unsafe self._local will just segfault."""
        if v in pset:
            raise IndexError(
                "Attempting to query score for (v, pset) where v \in pset"
            )
        # Because min() will raise error with empty pset
        if v in range(self.data.n) and len(pset) == 0:
            return self._local(v, pset)
        if min(v, min(pset)) < 0 or max(v, max(pset)) >= self.data.n:
            raise IndexError(
                "Attempting to query score for (v, pset) "
                "where some variables don't exist in data"
            )
        return self._local(v, pset)

    def _local(self, v, pset):
        # NOTE: How expensive are nested function calls?
        return self.scorer.local(v, pset) + self.priorf[self.prior["name"]](
            len(pset)
        )

    def score_dag(self, dag):
        dag = validate.dag(dag)
        return sum([self.local(v, np.array(list(pset))) for v, pset in dag])

    def clear_cache(self):
        self.scorer.clear_cache()

    def candidate_scores(self, C=None):
        # There should be an option to return this for a given node
        if C is None:
            C = np.array(
                [
                    np.array([j for j in range(self.data.n) if j != i])
                    for i in range(self.data.n)
                ],
                dtype=np.int32,
            )
        prior = np.array([bin(i).count("1") for i in range(2 ** len(C[0]))])
        prior = np.array(
            list(map(lambda k: self.priorf[self.prior["name"]](k), prior))
        )
        return self.scorer.candidate_score_array(C) + prior

    def complement_psets_and_scores(self, v, C, d):
        psets, scores, pset_len = self.scorer.complement_psets_and_scores(
            v, C, d
        )
        prior = np.array(
            list(map(lambda k: self.priorf[self.prior["name"]](k), pset_len))
        )
        return psets, scores + prior

    def all_scores_dict(self, C=None):
        # NOTE: Not used in Gadget pipeline, but useful for example
        #       when computing input data for aps.
        scores = dict()
        if C is None:
            C = {
                v: tuple(sorted(set(range(self.data.n)).difference({v})))
                for v in range(self.data.n)
            }
        for v in C:
            tmp = dict()
            for pset in subsets(
                C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]
            ):
                tmp[frozenset(pset)] = self._local(v, np.array(pset))
            scores[v] = tmp
        return scores


class EmptyDataScore:
    def __init__(self, **kwargs):
        pass

    def local(self, v, pset):
        return 0

    def candidate_score_array(self, C):
        return np.zeros((len(C), 2 ** len(C[0])))

    def clear_cache(self):
        pass

    def complement_psets_and_scores(self, v, C, d):
        n = len(C)
        k = (n - 1) // 64 + 1
        pset_tuple = list(
            filter(
                lambda ss: not set(ss).issubset(C[v]),
                subsets([u for u in C if u != v], 1, d),
            )
        )
        pset_len = np.array(list(map(len, pset_tuple)), dtype=np.int32)
        pset_bm = list(
            map(lambda pset: bm_to_np64(bm(set(pset)), k), pset_tuple)
        )
        scores = np.array([self.local(v, pset) for pset in pset_tuple])
        return np.array(pset_bm), scores, pset_len


class Score:  # should be renamed to e.g. ScoreHandler
    def __init__(self, *, C, c_r_score, c_c_score):

        self.C = C
        self.n = len(self.C)
        self.c_r_score = c_r_score
        self.c_c_score = c_c_score

    def sum(self, v, U, T=set()):
        """Returns the sum of scores for node v over the parent sets that
        1. are subsets of U;
        2. and, if T is not empty, have at least one member in T.

        The sum is computed over first the scores restricted to candidate
        parents (self.C), and then the result is augmented by scores
        complementary to those restricted to the candidate parents, until
        some predefined level of error.

        Args:
           v (int): Label of the node whose local scores are summed.
           U (set): Parent sets of scores to be summed are the subsets of U.
           T (set): Parent sets must have at least one member in T
                    (if T is not empty).

        Returns:
            Sum of scores (float).

        """

        U_bm = bm(U.intersection(self.C[v]), idx=self.C[v])
        # T_bm can be 0 if T is empty or does not intersect C[v]
        T_bm = bm(T.intersection(self.C[v]), idx=self.C[v])
        if len(T) > 0:
            if T_bm == 0:
                W_prime = -float("inf")
            else:
                W_prime = self.c_r_score.sum(v, U_bm, T_bm)
        else:
            W_prime = self.c_r_score.sum(v, U_bm)
        if self.c_c_score is None or U.issubset(self.C[v]):
            # This also handles the case U=T={}
            return W_prime
        if len(T) > 0:
            return self.c_c_score.sum(v, U, T, W_prime)  # [0]
        else:
            # empty pset handled in c_r_score
            return self.c_c_score.sum(v, U, U, W_prime)  # [0]

    def sample_pset(self, v, U, T=set()):

        U_bm = bm(U.intersection(self.C[v]), idx=self.C[v])
        T_bm = bm(T.intersection(self.C[v]), idx=self.C[v])

        if len(T) > 0 and T_bm == 0 and self.c_c_score is None:
            raise RuntimeError(
                "Cannot meet constraints if d=0 (c_c_score is None) "
                "and T does not intersect C[v]"
            )

        if len(T) > 0:
            if T_bm == 0:
                w_crs = -float("inf")
            else:
                w_crs = self.c_r_score.sum(v, U_bm, T_bm, isum=True)
        else:
            w_crs = self.c_r_score.sum(v, U_bm)

        w_ccs = -float("inf")
        if self.c_c_score is not None and not U.issubset(self.C[v]):
            if len(T) > 0:
                w_ccs = self.c_c_score.sum(v, U, T)
            else:
                # Empty pset is handled in c_r_score
                w_ccs = self.c_c_score.sum(v, U, U)

        if (
            self.c_c_score is None
            or -np.random.exponential() < w_crs - np.logaddexp(w_ccs, w_crs)
        ):
            # Sampling from candidate psets.
            pset, family_score = self.c_r_score.sample_pset(
                v, U_bm, T_bm, w_crs - np.random.exponential()
            )
            family = (v, set(self.C[v][i] for i in bm_to_ints(pset)))

        else:
            # Sampling from complement psets.
            if len(T) > 0:
                pset, family_score = self.c_c_score.sample_pset(
                    v, U, T, w_ccs - np.random.exponential()
                )
            else:
                pset, family_score = self.c_c_score.sample_pset(
                    v, U, U, w_ccs - np.random.exponential()
                )

            family = (v, set(pset))

        return family, family_score

    def sample_DAG(self, R):
        DAG = list()
        DAG_score = 0
        for v in range(self.n):
            for i in range(len(R)):
                if v in R[i]:
                    break
            if i == 0:
                family = (v, set())
                family_score = self.sum(v, set(), set())
            else:
                U = set().union(*R[:i])
                T = R[i - 1]
                family, family_score = self.sample_pset(v, U, T)
            DAG.append(family)
            DAG_score += family_score
        return DAG, DAG_score


class ScorePythonImpl():
    def __init__(self, *, C, score_array):
        self.C = C
        self.n = len(self.C)
        self.n = len(self.C)
        self.score_array = score_array

    def precompute(self):
        # Usual tau
        K = len(self.C[0])
        self.tau = np.copy(self.score_array)  # Standard score tau(U)
        for v in range(self.n):
            for i in range(K):
                newtau = np.copy(self.tau[v, :])
                for idx in range(2 ** K):
                    bit_idx = format(idx, '0'+str(K)+'b')
                    if bit_idx[i] == '1':
                        idx_minusi = idx - 2**(K - 1 - i)
                        newtau[idx] = np.logaddexp(self.tau[v, idx_minusi], self.tau[v, idx])


                self.tau[v, :] = newtau

    def sum(self, v, U):
        U_bm = bm(U)
        return self.tau[v, U_bm]



class ScoreTRUST(Score):
    """Extends the Score object to handle s1/s2 sets. In particular, changes sum method to restrict the parent sets
    appropriately.
    """
    def __init__(self, *, C, c_r_score, c_c_score, s1, s2, d):
        super().__init__(C=C, c_r_score=c_r_score, c_c_score=c_c_score)
        # C, c_r_score, c_c_score are all absolute
        self.n = len(s2) # so that sample_DAG only finds parents for the vars in s2

        self.s1 = s1
        self.s2 = s2
        self.d = d


    def sum(self, v, U, T=set()):
        """Returns the sum of scores for node v over the parent sets that
        1. are subsets of U OR in self.s1;
        2. and, if T is not empty, have at least one member in T.

        Takes as input **relative** indices for nodes; that is, their index in the s2 array. For example, if
        s2 = [1, 4, 7, 9], then the **absolute** node 7 would have **relative** index 2.

        Args:
           v (int): Label of the node whose local scores are summed.
           U (set): Parent sets of scores to be summed are the subsets of U.
           T (set): Parent sets must have at least one member in T
                    (if T is not empty).

        Returns:
            Sum of scores (float).

        """
        v_absolute = self.s2[v]
        U_absolute = set(self.s2[u] for u in U)
        U_absolute_with_s1 = U_absolute.union(self.s1)
        T_absolute = set(self.s2[t] for t in T)

        return super().sum(v_absolute, U_absolute_with_s1, T_absolute)
