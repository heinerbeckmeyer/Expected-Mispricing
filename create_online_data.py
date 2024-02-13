"""
    The file exemplifies how we create the ensemble forecast and creates a file to be made publicly available.

    The data will be stored in the Apache Parquet file `EMP_online_data.pq` and contain the following information:
        - date:         (monthly) datetime index
        - permno:       CRSP permanent stock identifier
        - expected_mp:  next month (expected) mispricing (model prediction)
        - lead1m_mp:    next month realized mispricing (IPCA residual component)
"""
# %%
# Packages
import pandas as pd
import os


# %%
# Declaration of variables
modelNames = [
    "NN___oos___20230711_1841",  # feed-forward neural network
    "RF___oos___20230712_0755",  # random forest
    "GBT___oos___20230712_1316",  # gradient-boosted regression tree
]
modelLoc = "../05_models"
dataLoc = "../04_dataset"

# %%
# Get data (i.e., ensemble average)
data = 0.0
for m in modelNames:
    data += pd.read_parquet(os.path.join(modelLoc, m, "results"))
data /= len(modelNames)

# Rename for clarity
data = data[["pred"]].rename(columns={"pred": "expected_mp"})

# Add realized future mispricing (i.e., the target)
mp = pd.read_parquet(os.path.join(dataLoc, "alphas/oos.pq"))
mp = mp["resid"].to_frame("lead1m_mp")
data = data.merge(mp, on=["date", "id"])

# Rename id to permno
data.index = data.index.rename({"id": "permno"})

# Save
data.to_parquet("../09_public/EMP_data.pq")
data.to_csv("../09_public/EMP_data.csv")


# %%
