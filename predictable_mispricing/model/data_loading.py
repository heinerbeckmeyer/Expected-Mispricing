# %%
# Packages
from typing import Iterator
import pandas as pd
import math
import numpy as np
from random import shuffle

# torch
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist


# %%
# Sample selection
def sample_selection(
    char_file: str,
    mcap_file: str,
    target_file: str,
    weighted_loss: bool,
    test_start_month: str,
    val_pct: float,
    n_test_months: int,
    shuffle_months: bool,
    sample: str,
    winsorize: bool,
    choose_char: list = [],
):
    print(f"Loading {sample} data.")

    # ---- read data
    char = pd.read_parquet(char_file).sort_index()
    char = char.drop(columns=["ret_exc_lead1m"])

    # set missing characteristics to zero
    char = char.fillna(0)

    # choose characteristics
    if choose_char:
        char = char[choose_char]

    # read target
    target = pd.read_parquet(target_file).sort_index()

    # read mcap
    weights = pd.read_parquet(mcap_file, columns=["id", "market_equity"]).sort_index()
    weights = weights.set_index("id", append=True)
    weights.columns = ["mcap_weight"]

    # merge to assure that we have same sorting
    char = target.merge(char, on=["date", "id"])
    char = char.merge(weights, on=["date", "id"])
    char = char.dropna(subset=["mcap_weight"])
    weights = char["mcap_weight"]
    char = char.drop(columns=["mcap_weight"])

    # extract mcap weights
    if weighted_loss:
        weights = np.sqrt(weights) / np.sqrt(weights).groupby("date").transform("mean")
    else:
        weights = weights * 0 + 1
    weights = weights.to_frame()

    # --- extract target column
    target = char["resid"].copy()
    if winsorize:
        qs = target.quantile([0.005, 0.995])
        target[target <= qs.iloc[0]] = qs.iloc[0]
        target[target >= qs.iloc[1]] = qs.iloc[1]
    target = target.to_frame()
    char = char.drop(columns=["resid"])

    # ---- create train/validation/testing split
    test_start_month = pd.to_datetime(test_start_month, format="%Y-%m-%d")
    test_end_month = test_start_month + pd.DateOffset(months=n_test_months - 1)
    all_months = char.index.get_level_values("date").sort_values().unique()
    train_val_dates = all_months[all_months < test_start_month]
    N_train_val_months = len(train_val_dates)
    n_val_months = int(N_train_val_months * val_pct)
    if shuffle_months:
        train_val_dates = pd.DatetimeIndex(pd.Series(train_val_dates).sample(frac=1, random_state=123))
        if sample == "training":
            char = char.loc[train_val_dates[:-n_val_months]].sort_index()
            target = target.loc[train_val_dates[:-n_val_months]].sort_index()
            weights = weights.loc[train_val_dates[:-n_val_months]].sort_index()
        elif sample == "validation":
            char = char.loc[train_val_dates[-n_val_months:]].sort_index()
            target = target.loc[train_val_dates[-n_val_months:]].sort_index()
            weights = weights.loc[train_val_dates[-n_val_months:]].sort_index()
        elif sample == "testing":
            char = char.loc[test_start_month:test_end_month]
            target = target.loc[test_start_month:test_end_month]
            weights = weights.loc[test_start_month:test_end_month]
    else:
        val_start_month = test_start_month - pd.DateOffset(months=n_val_months)
        if sample == "training":
            char = char.loc[: val_start_month - pd.DateOffset(months=1)]
            target = target.loc[: val_start_month - pd.DateOffset(months=1)]
            weights = weights.loc[: val_start_month - pd.DateOffset(months=1)]
        elif sample == "validation":
            char = char.loc[val_start_month : (test_start_month - pd.DateOffset(months=1))]
            target = target.loc[val_start_month : (test_start_month - pd.DateOffset(months=1))]
            weights = weights.loc[val_start_month : (test_start_month - pd.DateOffset(months=1))]
        elif sample == "testing":
            char = char.loc[test_start_month:test_end_month]
            target = target.loc[test_start_month:test_end_month]
            weights = weights.loc[test_start_month:test_end_month]
        elif sample == "test_all":
            pass
        else:
            raise ValueError("sample must be in ['training', 'validation', 'testing'].")

    return char, target, weights


# %%
# Functions
class MyDistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.dataset = dataset
        self.shuffle = shuffle
        self.total_size = len(dataset)  # dropping excess already done in dataset
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

        # then shuffle indices per rank >> WITHIN EACH RANK!
        if self.shuffle:
            shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Length per part of the DistributedSampler group (so for each rank!)"""
        return self.num_samples


class ModelDataset(Dataset):
    """PyTorch Dataset."""

    def __init__(
        self,
        char_file,
        mcap_file,
        target_file,
        weighted_loss: bool,
        test_start_month: str,
        val_pct: float,
        n_test_months: int,
        shuffle_months: bool,
        sample: str,
        winsorize: bool,
    ):
        """Initialize dataset."""

        char, target, weight = sample_selection(
            char_file=char_file,
            mcap_file=mcap_file,
            target_file=target_file,
            weighted_loss=weighted_loss,
            test_start_month=test_start_month,
            val_pct=val_pct,
            n_test_months=n_test_months,
            shuffle_months=shuffle_months,
            sample=sample,
            winsorize=winsorize,
        )

        # --- ranks
        self.sample = sample
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()

        # --- get dates and ids
        dates = char.index.get_level_values("date")
        iter_dates, self.dates = pd.factorize(dates)

        ids = char.index.get_level_values("id")
        iter_ids, self.ids = pd.factorize(ids)

        self.char_columns = char.columns

        # ---- get evenly divisible number of samples, which is required by PyTorch's DDP
        self.N = len(target)
        if self.N % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((self.N - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.N / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.N = self.total_size
        print(f"Number of samples per rank: {self.num_samples}")

        # apply self.N size to input data
        char = char.astype("float32").to_numpy()[-self.N :]
        target = target.astype("float32").to_numpy()[-self.N :]
        weight = weight.astype("float32").to_numpy()[-self.N :]
        iter_dates = iter_dates.astype("float32")[-self.N :]
        iter_ids = iter_ids.astype("float32")[-self.N :]

        # create equal input shape on each rank:
        char = char[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        target = target[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        weight = weight[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        iter_dates = iter_dates[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        iter_ids = iter_ids[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

        # ---- create tensors
        self.char = torch.as_tensor(char)
        self.target = torch.as_tensor(target)
        self.weight = torch.as_tensor(weight)
        self.iter_dates = torch.as_tensor(iter_dates)
        self.iter_permnos = torch.as_tensor(iter_ids)

    def retrieve_info(self):
        return {
            "N_char": self.char.shape[1],
            "dates": self.dates,
            "ids": self.ids,
            "chars": self.char_columns,
        }

    def __len__(self):
        """Total length of current data set."""
        return self.N

    def __getitem__(self, index):
        """Get next item (used for batching)."""
        index = index - self.rank * self.num_samples
        return (
            self.char[index],
            self.target[index],
            self.weight[index],
            self.iter_dates[index],
            self.iter_permnos[index],
        )


# %%
# future_ret = {}
# for m in [1, 3, 6, 12, 24, 60, 120]:
# 	print(m)
# 	a = out.merge(out["target"].unstack().iloc[::-1].rolling(m, m // 2).mean().iloc[::-1].stack().to_frame("lead"), on=["date", "id"])
# 	ret = a.groupby(["date", a.groupby("date", group_keys=False).apply(lambda x: pd.qcut(x.pred, 10, False))]).lead.mean().unstack()
# 	ret["hml"] = ret[9] - ret[0]
# 	future_ret[m] = ret.mean()
# pd.concat(future_ret, axis=1)


# %%
