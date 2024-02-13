"""
    This file creates our input dataset.

    We use the monthly firm-characteristics provided by Jensen, Kelly and Pedersen (2023). See
    https://github.com/bkelly-lab/ReplicationCrisis for an overview, the corresponding manual and SAS code
    for the data download via WRDS.

"""

# %%
# Packages
import pandas as pd
from predictable_mispricing.data.standardize import standardize

# %%
# Load the output file obtainend from Jensen et al. (2023)
jkp = pd.read_csv("../03_raw_data/usa.csv", low_memory=False)

# %%
# ---------- Filters ----------
# Only CRSP
jkp = jkp[jkp.source_crsp == 1]

# Suggested Screens by Jensen et al. (2023)
jkp = jkp[jkp.obs_main == 1]
jkp = jkp[jkp.common == 1]
jkp = jkp[jkp.exch_main == 1]
jkp = jkp[jkp.primary_sec == 1]

# Ensure datetime formats
jkp.loc[:, "date"] = pd.to_datetime(jkp["date"], format="%Y%m%d")
jkp.loc[:, "eom"] = pd.to_datetime(jkp["eom"], format="%Y%m%d")

# Set dates to start of month. This is our convention.
# NOTE Entries are still valid until the END of the respective month
jkp.loc[:, "date"] = jkp["date"].dt.to_period("M").dt.to_timestamp()

# --- CROP to relevant features
# Use 153 features only. Jensen et al. (2023) provide this list in the github repository
jkp_char = pd.read_excel("./jkp_chars.xlsx")
jkp = jkp.set_index("date").sort_index()
jkp = jkp.loc[:, ["id", "ret_exc_lead1m"] + jkp_char["name_in_our"].tolist()]

# Save "raw" data to make it more accessible
jkp.to_parquet("../03_raw_data/jkp.pq")
# -------------------------------


# %%
# Returns must be available
jkp = jkp.dropna(subset=["ret_exc_lead1m"])
jkp = jkp.set_index("id", append=True)

# start our sample in 1973
jkp = jkp.loc["1973":]

# Standardize between -0.5 and 0.5
jkp = jkp.groupby("date", group_keys=False).apply(
    lambda x: standardize(x, standardization_type="ranks", spare_cols=["ret_exc_lead1m"])
)

# Save
jkp.to_parquet("../04_dataset/jkp_1973_filtered.pq")


# %%
