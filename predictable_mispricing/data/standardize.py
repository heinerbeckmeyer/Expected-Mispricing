# %%
# Packages
import numpy as np


# %%
def standardize(group, standardization_type: str, n_groups: int = 0, spare_cols: list = []):
    if standardization_type == "percentiles":
        # enfore roughly equally-sized groups
        ranks = np.ceil(group.rank(pct=True, method="min") * n_groups)
    elif standardization_type == "ranks":
        # same ranks can be shared
        # NOTE: method="dense" produces equally-spaced intervals if not enough variation is provided,
        # method="min" assumes that spacing is proportional to number of observations with the same
        # rank.
        ranks = group.rank(method="min")
        ranks = ranks.sub(1).divide(ranks.max() - ranks.min(), axis=1) - 0.5
    elif standardization_type == "normalize":
        ranks = (group - group.mean()) / group.std()
    else:
        raise ValueError("Wrong standardization_type selected. Can be in ['percentiles', 'ranks'].")

    group.loc[:, [c for c in group.columns if c not in spare_cols]] = ranks.drop(columns=spare_cols)

    return group


# %%
