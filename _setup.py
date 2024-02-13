# %%
# Packages
import os

# %%
# Some logging
print("Make sure that ../03_raw_data/usa.csv exists (from JKP's SAS code).")


# %%
# Create folders
os.makedirs("../03_raw_data", exist_ok=True)
os.makedirs("../04_dataset", exist_ok=True)
os.makedirs("../05_models", exist_ok=True)
os.makedirs("../09_public", exist_ok=True)


# %%
# Setup
first_training_year = 1993


# %%
