# %%
import pandas as pd
import os
import platform
import datetime
from pathlib import Path
import cloudpickle
from shutil import copyfile


import ray
from ray import tune
from ray.tune import Tuner
from ray.air import session, Checkpoint
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.search import BasicVariantGenerator


import lightgbm as lgb

from predictable_mispricing.utils import timed_callout
from predictable_mispricing.model.data_loading import sample_selection

from _setup import first_training_year


# %%
# Train func
def train_loop_per_worker(config):
    # --- data
    train_X, train_y, weights = sample_selection(**config["data_params"], sample="training")
    val_X, val_y, weights = sample_selection(**config["data_params"], sample="validation")

    # --- model
    train_set = lgb.Dataset(train_X, train_y, weight=weights, free_raw_data=False, params={"verbose": -1}).construct()
    val_set = lgb.Dataset(val_X, val_y, weight=weights, free_raw_data=False, params={"verbose": -1}).construct()

    model = lgb.train(
        config["model_params"],
        train_set=train_set,
        valid_sets=[val_set],
    )

    # --- fit model
    score = (model.predict(val_X) - val_y["resid"]).pow(2).mean()
    train_score = (model.predict(train_X) - train_y["resid"]).pow(2).mean()
    r2 = 1 - score / val_y["resid"].pow(2).mean()
    train_r2 = 1 - train_score / train_y["resid"].pow(2).mean()

    # --- reporting
    print(f"\t MSE: {score:.3f}/{train_score:.3f}")
    print(f"\t R2:  {r2 * 100:.3f}/{train_r2 * 100:.3f}")

    session.report(
        {"score": score, "train_score": train_score},
        checkpoint=Checkpoint.from_dict(
            {
                "score": score,
                "train_score": train_score,
                "model": model,
            }
        ),
    )


def test_loop_per_worker(results_df):
    # --- data
    test_start_month = results_df["config/data_params"]["test_start_month"]
    test_X, test_y, weights = sample_selection(**results_df["config/data_params"], sample="testing")
    print(test_X.index.get_level_values("date").min(), test_X.index.get_level_values("date").max())

    output = 0.0
    for ensemble in range(results_df["config/hyper_params"]["num_ensemble"]):
        # --- load optimized model parameters
        checkpoint_to_load = os.path.join(
            os.path.split(os.path.split(results_df.logdir)[0])[0], f"best_{test_start_month}_{ensemble+1}"
        )
        with Path(checkpoint_to_load).expanduser().open("rb") as f:
            checkpoint = cloudpickle.load(f)

        model = checkpoint["model"]

        # --- run testing
        output += model.predict(test_X)

    output /= results_df["config/hyper_params"]["num_ensemble"]
    output = pd.DataFrame(output, index=test_X.index, columns=["pred"])
    output["target"] = test_y

    # saving
    saveLoc = os.path.join(logLoc, "results")
    os.makedirs(saveLoc, exist_ok=True)
    output.sort_index().to_parquet(os.path.join(saveLoc, f"output_{test_start_month}.pq"))


# %%
# Training

# --- Setup
MODEL_LOC = "../05_models"
MODEL_TYPE = "RF"
TARGET_FILE = "../04_dataset/alphas/oos.pq"

USE_GPU = True if platform.system() in ["Linux", "Windows"] else False

N_TEST_MONTHS = 12
TEST_START_MONTHS = pd.date_range(f"{first_training_year}-01-01", "2022-12-31", freq="M").to_period("M").to_timestamp()
TEST_START_MONTHS = [date.strftime("%Y-%m-%d") for date in TEST_START_MONTHS]
TEST_START_MONTHS = TEST_START_MONTHS[::12]
TEST_START_MONTHS = TEST_START_MONTHS[::-1]


NUM_WORKERS = 4
RESUME_TRIAL = ""

# --- init ray
ray.init(include_dashboard=False, num_cpus=NUM_WORKERS)
timed_callout("Starting new experiment.")
trial_number = datetime.datetime.now().strftime("%Y%m%d_%H%M")
logLoc = os.path.abspath(
    os.path.join(
        MODEL_LOC,
        MODEL_TYPE + "___" + os.path.split(TARGET_FILE)[-1].replace(".pq", "") + "___" + str(trial_number),
    )
)
os.makedirs(logLoc, exist_ok=True)
os.makedirs(os.path.join(logLoc, "logs"), exist_ok=True)


# --- config
config = {}
config["hyper_params"] = dict(
    num_workers=NUM_WORKERS,
    num_samples=25,  # take the best out of N models
    num_ensemble=5,  # final model = average of predictions of N best models
)
config["model_params"] = dict(
    objective="regression",
    metric=["l2"],
    num_threads=2,
    boosting="rf",  # "gbdt"
    num_iterations=500,
    num_leaves=tune.choice([30, 100]),
    max_depth=0,  # unrestricted
    min_data_in_leaf=tune.choice([10, 30]),
    bagging_freq=1,
    bagging_fraction=tune.uniform(0.3, 1),
    feature_fraction=tune.uniform(0.3, 1),
    lambda_l1=tune.loguniform(0.05, 1.0),
    lambda_l2=tune.loguniform(0.05, 1.0),
    learning_rate=tune.loguniform(0.05, 1.0),
    early_stopping_round=20,
)
config["data_params"] = dict(
    char_file=os.path.abspath("../04_dataset/jkp_1973_filtered.pq"),
    mcap_file=os.path.abspath("../03_raw_data/jkp.pq"),
    target_file=os.path.abspath(TARGET_FILE),
    weighted_loss=False,
    test_start_month=tune.grid_search(TEST_START_MONTHS),
    n_test_months=N_TEST_MONTHS,
    val_pct=0.3,
    shuffle_months=False,
    winsorize=True,
)
config["tune_params"] = dict(
    trial_stopper=False,
    trial_num_results=0,
    trial_grace_period=0,
    trial_tol=0.0,
    experiment_stopper=False,
    exp_top_models=10,
    exp_num_results=64,
    exp_grace_period=0,
    exp_tol=0.0,
)


# --- Tune setup
search_alg = BasicVariantGenerator()

# --- run Ray Tune
tuner = Tuner(
    trainable=train_loop_per_worker,
    run_config=RunConfig(
        name=MODEL_TYPE,
        local_dir=logLoc,
        verbose=3,
        checkpoint_config=CheckpointConfig(num_to_keep=config["tune_params"]["trial_num_results"] + 1),
    ),
    param_space=config,
    tune_config=tune.TuneConfig(
        search_alg=search_alg, mode="min", metric="score", num_samples=config["hyper_params"]["num_samples"]
    ),
)
tuned = tuner.fit()


# %%
# --- save best runs
def save_best_trials(
    trialLoc, logLoc, results_df, n_best_models_to_save: int = 10, grace_period: int = None, grid_variables: list = []
):
    # save best config dataframe:
    analysis = tune.ExperimentAnalysis(trialLoc)
    dfs = analysis.trial_dataframes
    cfs = analysis.get_all_configs()
    best_df = []

    # extract columns names:
    for trial in dfs.keys():
        if len(dfs[trial]) > 0:
            if "score" in dfs[trial].columns.tolist():
                column_names = dfs[trial].columns.tolist()
    for trial in dfs.keys():
        if len(dfs[trial]) > 0:
            tmp = dfs[trial]
            tmp.columns = column_names
            tmp["path"] = trial
            cf = pd.Series(cfs[trial]).to_frame().T
            cf.columns = ["config/" + str(c) for c in cf.columns]
            tmp = pd.concat((tmp, cf), axis=1)
            for col in tmp.columns:
                if "config/" in col:
                    tmp[col] = tmp[col].ffill()
            best_df.append(tmp)
    best_df = pd.concat(best_df).reset_index(drop=True)

    if grace_period is None:
        grace_period = best_df["config/tune_params"].apply(pd.Series)["trial_grace_period"]
        best_df = best_df[best_df["training_iteration"] > grace_period]
    else:
        best_df = best_df[best_df["training_iteration"] > grace_period]
    best_df = best_df.sort_values("score")

    for conf in best_df.columns:
        if "config/" in conf:
            to_merge = best_df[conf].apply(pd.Series)
            best_df = pd.concat((best_df, to_merge), axis=1)

    # get checkpoint paths
    best_df = best_df.merge(results_df[["trial_id", "logdir"]].set_index("trial_id"), on="trial_id")
    best_df.to_pickle(os.path.join(logLoc, "all_trials.pkl"))

    # ---- retrieve up to "n_models" best checkpoints save in logLoc.
    best_df = best_df.sort_values("score").groupby("trial_id").first()
    if grid_variables:
        best_df["rank"] = best_df.groupby(grid_variables).score.rank(ascending=True, method="dense")
    else:
        best_df["rank"] = best_df.score.rank(ascending=True, method="dense")
    best_df = best_df[best_df["rank"] <= n_best_models_to_save]
    best_df = best_df.sort_values("rank")

    for i, (idx, row) in enumerate(best_df.iterrows()):  # copy n models over
        print("Copying best trials (%d-best model, path=%s)" % (i, idx))
        best_iteration = row["training_iteration"] - 1
        # paths = get_checkpoints_paths(row["path"])
        print("Looking for best iteration %d." % best_iteration)
        path = os.path.join(
            row["logdir"], f"checkpoint_{str(row.training_iteration-1).zfill(6)}", "dict_checkpoint.pkl"
        )
        rank = row["rank"]
        specifier = "best_" + "_".join(row[var] for var in grid_variables)
        print(f"{specifier}")
        copyfile(path, os.path.join(logLoc, f"{specifier}_{int(rank)}"))

    best_df.dropna(how="all").to_pickle(os.path.join(logLoc, "configs.pkl"))


save_best_trials(
    trialLoc=os.path.join(logLoc, MODEL_TYPE),
    logLoc=logLoc,
    results_df=tuned.get_dataframe(),
    n_best_models_to_save=config["hyper_params"]["num_ensemble"],
    grace_period=None,
    grid_variables=["test_start_month"],
)


# %%
# Testing
best_df = pd.read_pickle(os.path.join(logLoc, "configs.pkl"))
best_df = best_df.sort_values("rank").groupby("test_start_month").first()
for _, cfg in best_df.iterrows():
    test_loop_per_worker(cfg)


# %%
# Shut down Ray
ray.shutdown()


# %%
