# %%
# Packages
import time


# %%
# Functions
def timed_callout(callout: str):
    print(time.ctime() + " ::: " + str(callout), flush=True)


# %%
