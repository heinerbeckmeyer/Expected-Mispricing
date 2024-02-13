# Expected Mispricing
by Turan G. Bali, Heiner Beckmeyer and Timo Wiedemann (2023)


## Overview
This repository provides replication code for the paper [Expected Mispricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4638234) by Bali, Beckmeyer and Wiedemann (2023). Please cite this paper if you are using the code or data:

```
@techreport{bali2023expected,
  title={{Expected Mispricing}},
  author={Bali, G. Turan and Beckmeyer, Heiner and Wiedemann, Timo},
  type={{Working Paper}},
  institution={{Available at SSRN}},
  year={2023}
}
```

#### 1. Dataset creation:
The file `1_create_dataset.py` creates the inital dataset. We use the code provided by [Jensen, Kelly and Pedersen (2023)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13249) ([GitHub](https://github.com/bkelly-lab/ReplicationCrisis)) to get a a time-series for a set of 153 monthly firm-level characteristics and apply filters proposed by the authors.

#### 2. Obtain stock-specific *realized* mispricing via IPCA:
The file `2_run_ipca.py` estimates monthly stock-specific realized mispricing (MP) defined as the residual return component relative to a six-factor IPCA model. We carefully set up an estimation procedure that avoids the inclusion of forward-looking information by including only information available at time $t$ when calculating the realized mispricing for the next month, $t+1$.

#### 3. Calculate *expected* mispricing:
Our measure of firm $i$'s expected mispricing is expressed as a non-linear function $g$ of today's firm-characteristics $z_{i,t}$, $E_t[MP_{i,t+1}] = g(z_{i,t})$.
We approximate $g(\cdot)$ by three well-established machine learning estimators:  (i) a feed-forward neural network with tree hidden layers; (ii) a gradient-boosted regression tree; and (iii) a random forest. The folling files estimate the models, respectively:

 1. `3_run_nn.py` estimates the feed-forward neural network.
 2. `3_run_gbt.py` estimates the gradient-boosted regression tree.
 3. `3_run_rf.py` estimates the random forest.

We obtain our final measure of expected mispricing based on an equal-weighted ensemble of these three forecasts: $E_t[MP_{i,t+1}] = (g_j(z_{i,t})^{NN} + g_j(z_{i,t})^{GBT} + g_j(z_{i,t})^{RF}) / 3$.

#### 4. Generate mispricing dataset:
The file `create_online_data.py` exemplifies how we create the ensemble forecasts and creates a file to be made publicly available. The Apache Parquet file `EMP_data.pq` contains the following information:

| Variable | Description |
| -------- | ----------- |
| date | Datetime index $t$ (monthly)|
| permno | CRSP permanent stock identifier| 
| expected_mp | Next month $t+1$ (expected) mispricing (model prediction) |
| lead1m_mp | Next month $t+1$ realized mispricing (i.e., IPCA residual component) |

We also provide a `.csv` file (`EMP_data.csv`) with the same information.

For convenience, both files can be downloaded directly from [Dropbox](https://www.dropbox.com/scl/fo/ylmjcth0wh0x1le7vxie7/h?rlkey=isbnjmtbpzpw5ref4yx2x3zns&dl=0), with firm-level mispricing data covering January 1993 through December 2022.