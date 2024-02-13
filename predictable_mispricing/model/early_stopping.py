# %%
from ray.tune import Stopper
import numpy as np
from collections import defaultdict
from typing import Dict


# %%
class ExperimentAndTrialPlateauStopper(Stopper):
    def __init__(
        self,
        metric: str,
        mode: str,
        epochs: int,
        experiment_stopper: bool,
        trial_stopper: bool,
        trial_logfile: str,
        exp_logfile: str,
        exp_top_models: int = 10,
        exp_num_results: int = 100,
        exp_grace_period: int = 0,
        exp_tol: float = 0.0,
        trial_num_results: int = 5,
        trial_grace_period: int = 5,
        trial_tol: float = 0.0,
    ):
        """Creates the EarlyStopping object.

        Stops the entire experiment or single trials when the
        metric has plateaued for more than the given amount of
        iterations specified in the patience parameter.
            ---- This stopping mechanism uses the *mean improvement*.

        Also stops trials early if the target metric has stopped
        improving for 'num_results' iterations.

        Args:
            metric (str): The metric to be monitored.
            top (int): The number of best model to consider.
            max_iter_no_change (int): Max SGD iterations without a change in
                mean best score of "top" trials.
            mode (str): The mode to select the top results.
                Can either be "min" or "max".
            patience (int): Number of epochs to wait for
                a change in the top models.
            num_results (int): Number of results to consider for stdev
                calculation.
            grace_period (int): Minimum number of timesteps before a trial
                can be early stopped
            tol (float): Tolerance for stopping early after n_iter_no_change
                iterations without score improvements

        Raises:
            ValueError: If the mode parameter is not "min" nor "max".
            ValueError: If the top parameter is not an integer
                greater than 1.
            ValueError: If the standard deviation parameter is not
                a strictly positive float.
            ValueError: If the patience parameter is not
                a strictly positive integer.
            ValueError: If the n_iter_no_change parameter is not
                a strictly positive integer.
        """
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be" " either min or max.")
        if not isinstance(exp_top_models, int) or exp_top_models <= 1:
            raise ValueError("Top results to consider must be" " a positive integer greater than one.")
        if not isinstance(exp_grace_period, int) or exp_grace_period < 0:
            raise ValueError("Patience must be" " a strictly positive integer.")
        if not isinstance(trial_tol, float):
            raise ValueError("The tol must be" " a float number.")
        if not isinstance(exp_tol, float):
            raise ValueError("The tol must be" " a float number.")

        # -----
        self._experiment_stopper = experiment_stopper
        self._trial_stopper = trial_stopper
        self._trial_logfile = trial_logfile
        self._exp_logfile = exp_logfile

        self._mode = mode
        self._metric = metric
        self._epochs = epochs

        # experiment level stuff
        self._exp_top_models = exp_top_models
        self._exp_grace_period = exp_grace_period
        self._exp_num_results = exp_num_results
        self._exp_tol = exp_tol
        self.done_trials = []
        self.exp_no_imp = 0
        self.exp_top_mean = 1000000.0
        self.exp_top_values = []
        self.mean_improvements = []
        self.stopped = False

        # trial stuff
        self._num_results = trial_num_results
        self._grace_period = trial_grace_period
        self._tol = trial_tol
        self.scores = defaultdict(list)

    def __call__(self, trial_id: str, result: Dict):
        """Return a boolean representing if the tuning for a trial has to stop."""
        self.scores[trial_id].append(result[self._metric])
        iterations = len(self.scores[trial_id])

        # If still in grace period, do not stop yet
        if iterations <= self._grace_period:
            self.trial_logger(result, trial_id, -10000, 0, iterations)
            return False

        # ----- obtain best score for current trial:
        if self._mode == "min":
            best = min(self.scores[trial_id][self._grace_period :])
        else:
            best = max(self.scores[trial_id][self._grace_period :])

        # ----- trial-level improvement:
        no_improvement = 0
        ii = np.argmin(self.scores[trial_id][self._grace_period :])
        for i in range(ii + self._grace_period, len(self.scores[trial_id])):
            if self._mode == "min":
                if (self.scores[trial_id][i] + self._tol) >= best:
                    no_improvement += 1
            else:
                if (self.scores[trial_id][i] + self._tol) <= best:
                    no_improvement += 1
        no_improvement -= 1  # best score

        # ----- trial logging:
        self.trial_logger(result, trial_id, best, no_improvement, iterations)

        # -------- stopping trial:
        if self._trial_stopper:

            # If not enough results yet, do not stop yet
            if iterations < self._num_results:
                return False

            # early stop ---- prepare experiment-level stopping here:
            if no_improvement >= self._num_results:  # no improvements

                self.done_trials.append(best)
                if self._mode == "min":
                    self.exp_top_values = sorted(self.done_trials)[: self._exp_top_models]
                else:
                    self.exp_top_values = sorted(self.done_trials)[-self._exp_top_models :]

                # check for improvements of the mean score:
                mean_finished_trials = np.mean(self.exp_top_values)
                self.mean_improvements.append(
                    (self.exp_top_mean - mean_finished_trials) / self.exp_top_mean
                )  # mean improv in %
                # check if mean imp. (in % !) is greater than tol
                if self.mean_improvements[-1] <= self._exp_tol:
                    self.exp_no_imp += 1
                else:
                    self.exp_no_imp = 0
                self.exp_top_mean = mean_finished_trials

                # ----- experiment logging:
                self.experiment_logger(result, trial_id, best, iterations)

                return True

        else:
            # prepare experiment-level stopping here:
            if iterations == (self._epochs + 1):  # end of training (max epochs)
                self.done_trials.append(best)
                if self._mode == "min":
                    self.exp_top_values = sorted(self.done_trials)[: self._exp_top_models]
                else:
                    self.exp_top_values = sorted(self.done_trials)[-self._exp_top_models :]

                # check for improvements of the mean score:
                mean_finished_trials = np.mean(self.exp_top_values)
                self.mean_improvements.append(
                    (self.exp_top_mean - mean_finished_trials) / self.exp_top_mean
                )  # mean improv in %
                # check if mean imp. (in % !) is greater than tol
                if self.mean_improvements[-1] <= self._exp_tol:
                    self.exp_no_imp += 1
                else:
                    self.exp_no_imp = 0
                self.exp_top_mean = mean_finished_trials

                # ----- experiment logging:
                self.experiment_logger(result, trial_id, best, iterations)

        return False

    def stop_all(self, override: bool = False):
        """Return whether to stop and prevent trials from starting."""
        # NOTE: checking for "done" trials does not work, as scheduler
        # info (i.e. ASHA) does not happen before calling early_stopping.py
        if self.stopped:
            return True
        if override:
            with open(self._exp_logfile, "a+") as log:
                print("\n\n\n*** Stopping experiment . ***", file=log)
            self.stopped = True
            return True
        # if self._experiment_stopper:
        #     trials = len(self.done_trials)
        #     if trials >= self._exp_grace_period:  # check if enough trials finished.
        #         if self.exp_no_imp >= self._exp_num_results:
        #             with open(self._exp_logfile, "a+") as log:
        #                 print("\n\n\n*** Stopping experiment . ***", file=log)
        #             return True
        return False

    def experiment_logger(self, result, trial_id, best, iterations):
        with open(self._exp_logfile, "a+") as log:
            print("===== Trial with id %s done at epoch %d." % (trial_id, iterations), file=log)
            for key in result["config"]:
                print(key + ":", file=log)
                print(result["config"][key], file=log)
            print("\n\n", file=log)
            print("%d trials done." % len(self.done_trials), file=log)
            print(
                "Trials without improvements: %d/%d." % (self.exp_no_imp, self._exp_num_results),
                file=log,
            )
            print("This trial's best score: %.6f" % best, file=log)
            print("Last %d mean improvements (in percent):" % self._exp_top_models, file=log)
            print(
                "   ".join("%.6f" % i for i in self.mean_improvements[-self._exp_top_models :]),
                file=log,
            )
            print("Mean of top %d trials: %.6f." % (self._exp_top_models, self.exp_top_mean), file=log)
            print("\n\n", file=log)

    def trial_logger(self, result, trial_id, best, no_improvement, iterations):
        with open(self._trial_logfile, "a+") as log:
            print("===== Epoch %d at trial with id %s." % (iterations, trial_id), file=log)
            for key in result["config"]:
                print(key + ":", file=log)
                print(result["config"][key], file=log)
            print("\n\n", file=log)
            print("Last %d scores this trial." % self._num_results, file=log)
            print(
                "   ".join("%.6f" % i for i in self.scores[trial_id][-self._num_results :]),
                file=log,
            )
            print(
                "Best score %.6f; no improvement for %d iterations." % (best, no_improvement),
                file=log,
            )
            print("\n\n", file=log)


# %%
