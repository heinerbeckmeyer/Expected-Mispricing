# %%
from ray.tune import Callback
import numpy as np


# %% Callback class:
class MyCallback(Callback):
    def __init__(
        self,
        mode: str,
        stopper,
        exp_logfile: str,
        exp_top_models: int = 10,
        exp_num_results: int = 100,
        exp_grace_period: int = 0,
        trial_grace_period: int = 0,
        exp_tol: float = 0.0,
    ):
        self._exp_logfile = exp_logfile
        self._mode = mode
        self._exp_top_models = exp_top_models
        self._exp_grace_period = exp_grace_period
        self._trial_grace_period = trial_grace_period
        self._exp_num_results = exp_num_results
        self._exp_tol = exp_tol

        self._stopper = stopper

        self.done_trials = []
        self.done = 0
        self.exp_no_imp = 0
        self.exp_top_mean = 10000000.0
        self.exp_top_values = []
        self.mean_improvements = []
        self.scores = {}

    def experiment_logger(self, config, trial_id, iterations, done):
        with open(self._exp_logfile, "a+") as log:
            print("===== Trial with id %s done at epoch %d." % (trial_id, iterations), file=log)
            for key in config:
                print(key + ":", file=log)
                print(config[key], file=log)
            print("\n\n", file=log)
            print("%d trials done." % done, file=log)
            print(
                "Trials without improvements: %d/%d." % (self.exp_no_imp, self._exp_num_results), file=log,
            )
            print("Last %d mean improvements (in percent):" % self._exp_top_models, file=log)
            print(
                "   ".join("%.6f" % i for i in self.mean_improvements[-self._exp_top_models :]), file=log,
            )
            print("Mean of top %d trials: %.6f." % (self._exp_top_models, self.exp_top_mean), file=log)
            print("\n\n", file=log)

    def on_trial_complete(self, iteration: int, trials, trial, **info):
        # NOTE: experiment-level stopping does not work when resuming experiment,
        # since old trials will not be found.

        iterations = len(self.scores[trial])

        done = 0
        done_trials = []
        for t in trials:
            if t.status in ["TERMINATED"]:
                done += 1
                if self._mode == "min":
                    try:
                        best = min(self.scores[t][self._trial_grace_period :])
                    except KeyError:
                        best = 1_000_000
                        print(
                            "It seems like the experiment was resumed. Old trial information is not retained."
                            + "Experiment-level stopping may therefore not work."
                        )
                else:
                    try:
                        best = max(self.scores[t][self._trial_grace_period :])
                    except KeyError:
                        best = -1_000_000
                        print(
                            "It seems like the experiment was resumed. Old trial information is not retained."
                            + "Experiment-level stopping may therefore not work."
                        )
                done_trials.append(best)
        new_done_trials = done - self.done  # how many trials completed in between calls to function.
        self.done = done

        if self._mode == "min":
            exp_top_values = sorted(done_trials)[: self._exp_top_models]
        else:
            exp_top_values = sorted(done_trials)[-self._exp_top_models :]

        # get mean score.
        mean_finished_trials = np.mean(exp_top_values)
        self.mean_improvements.append(
            (self.exp_top_mean - mean_finished_trials) / self.exp_top_mean
        )  # mean improv in %

        # check for mean improvements.
        if self.mean_improvements[-1] <= self._exp_tol:
            self.exp_no_imp += new_done_trials
        else:
            self.exp_no_imp = 0
        self.exp_top_mean = mean_finished_trials

        # ---- logging:
        self.experiment_logger(trial.config, trial, iterations, done)

        if self.exp_no_imp >= self._exp_num_results:
            with open(self._exp_logfile, "a+") as log:
                print("\n\n\n*** Stopping experiment early from callbacks.py ***", file=log)
            self._stopper.stop_all(override=True)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if trial not in self.scores.keys():
            self.scores[trial] = []
        self.scores[trial].append(result["score"])


# %%
