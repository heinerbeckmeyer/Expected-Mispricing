# %%
import pandas as pd
import os
import platform
import datetime
from pathlib import Path
import cloudpickle
from shutil import copyfile


import ray
from ray import train, tune
from ray.tune import Tuner
from ray.air import session, Checkpoint
from ray.train.torch import TorchTrainer, TorchConfig
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from ray.tune.search import BasicVariantGenerator

import torch
from torch.nn.modules.loss import _Loss

from predictable_mispricing.model.model import Network
from predictable_mispricing.utils import timed_callout
from predictable_mispricing.model.early_stopping import ExperimentAndTrialPlateauStopper
from predictable_mispricing.model.callbacks import MyCallback
from predictable_mispricing.model.loop import train_epoch, validate_epoch, testing_epoch
from predictable_mispricing.model.data_loading import MyDistributedSampler, ModelDataset

from _setup import first_training_year


# %%
# Loss function
class WeightedMSELoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight):
        loss = (((input - target) ** 2) * weight).mean()
        return loss


# %%
# Train func
def train_loop_per_worker(config):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    # --- data
    train_dataset = ModelDataset(**config["data_params"], sample="training")
    val_dataset = ModelDataset(**config["data_params"], sample="validation")

    # --- batch size
    batch_size = config["hyper_params"]["batch_size"]

    # --- sampling
    train_sampler = MyDistributedSampler(dataset=train_dataset)
    val_sampler = MyDistributedSampler(dataset=val_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, sampler=train_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size * 4, num_workers=0, pin_memory=True, sampler=val_sampler
    )
    # NOTE: necessary for all inputs to be on same device!
    train_loader = train.torch.prepare_data_loader(train_loader, add_dist_sampler=False)
    validation_loader = train.torch.prepare_data_loader(validation_loader, add_dist_sampler=False)

    # --- model
    sizes = train_loader._dataloader.dataset.retrieve_info()
    config["model_params"]["N_char"] = sizes["N_char"]
    model = Network(**config["model_params"])
    model = train.torch.prepare_model(model)

    # --- loss and optimization
    loss_fn = WeightedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), **config["optim_params"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        **config["scheduler_params"],
        epochs=config["hyper_params"]["epochs"],
        steps_per_epoch=(len(train_dataset) // batch_size) + 1,
    )

    # resume checkpoint?
    checkpoint = session.get_checkpoint() or {}
    if checkpoint:
        timed_callout("Restoring checkpoint.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        current_epoch = checkpoint["epoch"] + 1
    else:
        current_epoch = 0

    for epoch in range(current_epoch, config["hyper_params"]["epochs"]):
        train_score, train_r2 = train_epoch(
            dataloader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler
        )
        score, r2 = validate_epoch(dataloader=validation_loader, model=model, loss_fn=loss_fn)
        timed_callout(f"{epoch}/{config['hyper_params']['epochs']} ({get_lr(optimizer):.5f}).")
        print(f"\t MSE: {score:.3f}/{train_score:.3f}")
        print(f"\t R2:  {r2 * 100:.3f}/{train_r2 * 100:.3f}")

        session.report(
            {"score": score, "train_score": train_score},
            checkpoint=Checkpoint.from_dict(
                {
                    "epoch": epoch,
                    "_training_iteration": epoch,
                    "score": score,
                    "train_score": train_score,
                    "model": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
            ),
        )


def test_loop_per_worker(results_df):
    # --- setup
    config = results_df["config/train_loop_config"]

    # --- data
    test_start_month = config["data_params"]["test_start_month"]
    test_dataset = ModelDataset(**config["data_params"], sample="testing")

    # --- batch size
    batch_size = config["hyper_params"]["batch_size"]

    # --- sampling
    sampler = MyDistributedSampler(dataset=test_dataset)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size * 4,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
    )
    # NOTE: necessary for all inputs to be on same device!
    dataloader = train.torch.prepare_data_loader(dataloader, add_dist_sampler=False)

    # --- load optimized model parameters
    output = 0.0
    for ensemble in range(config["hyper_params"]["num_ensemble"]):
        checkpoint_to_load = os.path.join(
            os.path.split(os.path.split(results_df.logdir)[0])[0], f"best_{test_start_month}_{ensemble+1}"
        )
        with Path(checkpoint_to_load).expanduser().open("rb") as f:
            checkpoint = cloudpickle.load(f)

        model_state = checkpoint["model"]
        sizes = dataloader._dataloader.dataset.retrieve_info()
        config["model_params"]["N_char"] = sizes["N_char"]

        model = Network(**config["model_params"])
        for key in list(model_state.keys()):
            if key[:7] == "module.":
                model_state[key[7:]] = model_state.pop(key)
        model.load_state_dict(model_state)
        model.eval()  # evaluation mode, fixed batch norm and dropout layers.
        model = train.torch.prepare_model(model)

        # --- run testing
        output += testing_epoch(dataloader=dataloader, model=model)

    output /= config["hyper_params"]["num_ensemble"]

    # saving
    saveLoc = os.path.join(logLoc, "results")
    os.makedirs(saveLoc, exist_ok=True)
    output.sort_index().to_parquet(os.path.join(saveLoc, f"output_{test_start_month}.pq"))

    session.report({"score": 1})
    return


# %%
# Training

# --- Setup
MODEL_LOC = "../05_models"
MODEL_TYPE = "NN"
TARGET_FILE = "../04_dataset/alphas/oos.pq"

USE_GPU = True if platform.system() in ["Linux", "Windows"] else False

N_TEST_MONTHS = 12
TEST_START_MONTHS = pd.date_range(f"{first_training_year}-01-01", "2022-12-31", freq="M").to_period("M").to_timestamp()
TEST_START_MONTHS = [date.strftime("%Y-%m-%d") for date in TEST_START_MONTHS]
TEST_START_MONTHS = TEST_START_MONTHS[::N_TEST_MONTHS]
TEST_START_MONTHS = TEST_START_MONTHS[::-1]


NUM_WORKERS = 1
RESUME_TRIAL = ""


# --- init ray
ray.init(include_dashboard=False)
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
    epochs=50,
    batch_size=5000,
)
config["model_params"] = dict(
    dropout_p=0.1,
    hidden_layer_sizes=[32, 16, 8],
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
config["optim_params"] = dict(lr=0.001, amsgrad=True, weight_decay=0.001)
config["scheduler_params"] = dict(max_lr=0.005, pct_start=0.3, div_factor=1, final_div_factor=1)
config["tune_params"] = dict(
    trial_stopper=True,
    trial_num_results=10,
    trial_grace_period=2,
    trial_tol=0.0,
    experiment_stopper=False,
    exp_top_models=10,
    exp_num_results=64,
    exp_grace_period=0,
    exp_tol=0.0,
)


# --- Tune setup
search_alg = BasicVariantGenerator()

stopper = ExperimentAndTrialPlateauStopper(
    metric="score",
    mode="min",
    epochs=config["hyper_params"]["epochs"],
    trial_logfile=os.path.join(logLoc, "logs/trial_log.txt"),
    exp_logfile=os.path.join(logLoc, "logs/early_stopping_log.txt"),
    **config["tune_params"],
)

callback = MyCallback(
    mode="min",
    stopper=stopper,
    exp_logfile=os.path.join(logLoc, "logs/experiment_log.txt"),
    exp_top_models=config["tune_params"]["exp_top_models"],
    exp_num_results=config["tune_params"]["exp_num_results"],
    exp_grace_period=config["tune_params"]["exp_grace_period"],
    trial_grace_period=config["tune_params"]["trial_grace_period"],
    exp_tol=config["tune_params"]["exp_tol"],
)


# --- run Ray Tune
scaling_config = ScalingConfig(num_workers=config["hyper_params"]["num_workers"], use_gpu=USE_GPU)
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    torch_config=TorchConfig(backend="nccl" if platform.system() == "Linux" else "gloo"),
    scaling_config=scaling_config,
)
tuner = Tuner(
    trainable=trainer,
    run_config=RunConfig(
        name=MODEL_TYPE,
        local_dir=logLoc,
        verbose=3,
        stop=stopper,
        checkpoint_config=CheckpointConfig(num_to_keep=config["tune_params"]["trial_num_results"] + 1),
    ),
    param_space={"train_loop_config": config},
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
        grace_period = (
            best_df["config/train_loop_config"].apply(pd.Series)["tune_params"].apply(pd.Series)["trial_grace_period"]
        )
        best_df = best_df[best_df["training_iteration"] > grace_period]
    else:
        best_df = best_df[best_df["training_iteration"] > grace_period]
    best_df = best_df.sort_values("score")

    for conf in best_df["config/train_loop_config"].apply(pd.Series).columns:
        to_merge = best_df["config/train_loop_config"].apply(pd.Series)[conf].apply(pd.Series)
        best_df = pd.concat((best_df, to_merge), axis=1)

    # get checkpoint paths
    best_df = best_df.merge(results_df[["trial_id", "logdir"]].set_index("trial_id"), on="trial_id")
    best_df.to_pickle(os.path.join(logLoc, "all_trials.pkl"))

    # ---- retrieve up to "n_models" best checkpoints save in logLoc.
    best_df = best_df.sort_values("score").groupby("trial_id").first()
    if grid_variables:
        best_df["rank"] = best_df.groupby(grid_variables).score.rank(ascending=True)
    else:
        best_df["rank"] = best_df.score.rank(ascending=True)
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
scaling_config = ScalingConfig(
    num_workers=config["hyper_params"]["num_workers"],
    use_gpu=USE_GPU,
    _max_cpu_fraction_per_node=0.8,
)

best_df = pd.read_pickle(os.path.join(logLoc, "configs.pkl"))
best_df = best_df.sort_values("rank").groupby("test_start_month").first()
for _, cfg in best_df.iterrows():
    trainer = TorchTrainer(
        train_loop_per_worker=test_loop_per_worker,
        torch_config=TorchConfig(backend="nccl" if platform.system() == "Linux" else "gloo"),
        train_loop_config=cfg,
        scaling_config=scaling_config,
    )
    trainer.fit()


# %%
# Shut down Ray
ray.shutdown()


# %%
