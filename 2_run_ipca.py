# %%
# Packages
import pandas as pd
from predictable_mispricing.ipca.asset_class_ipca import MultiIPCA
from _setup import first_training_year
from joblib import dump, Parallel, delayed, cpu_count


# %%
# IPCA setup
n_factors = 6
n_iter = 500
tol = 0.005

intercept = False


# %%
# Full-sample IPCA model
# NOTE: we fit the full-sample model first to get good starting values for the OOS fits for faster convergence.
# NOTE: the full-sample models are not used in the paper.
model = MultiIPCA(n_factors=n_factors, n_iter=n_iter, tol=tol, n_jobs=1)
X, y = model.load_data()

model.fit(X=X, y=y, n_verbose=1)

print("no intercept", model.score(X, y, False))
print("intercept", model.score(X, y, True))


# %%
# IPCA models per out-of-sample year
# OUT-OF-SAMPLE
months = y.index.get_level_values("date").unique()
min_months_oos = (months.year < first_training_year).sum()


def oos_fit(date):
    fs, gs = model.cluster_oos_fit(X, y, date, intercept=intercept)
    return date, fs, gs


print(f"Working on a total of {len(X.index.get_level_values('date').unique()[min_months_oos:])} dates.")
out = Parallel(n_jobs=cpu_count() - 2, verbose=100)(
    delayed(oos_fit)(date) for date in X.index.get_level_values("date").unique()[min_months_oos:]
)
dates = [o[0] for o in out]
factors = [o[1] for o in out]
gammas = [o[2] for o in out]

factors = {d: f for d, f in zip(dates, factors)}
gammas = {d: g for d, g in zip(dates, gammas)}


if intercept:
    model.factors_AB_OOS = factors.copy()
    model.gamma_AB_OOS = gammas.copy()
else:
    model.factors_B_OOS = factors.copy()
    model.gamma_B_OOS = gammas.copy()


# save model
dump(model, "../04_dataset/ipca_models/oos.pkl")


# save residuals
if intercept:
    resid_oos = []
    for i, date in enumerate(model.factors_AB_OOS):
        if i == 0:
            resid_oos.append(
                y.loc[model.factors_AB_OOS[date].T.index]
                - X.loc[model.factors_AB_OOS[date].T.index]
                .dot(model.gamma_AB_OOS[date])
                .multiply(model.factors_AB_OOS[date].T)
                .drop(columns=["intercept"])
                .sum(axis=1)
            )
        else:
            resid_oos.append(
                (
                    y.loc[model.factors_AB_OOS[date].T.index]
                    - X.loc[model.factors_AB_OOS[date].T.index]
                    .dot(model.gamma_AB_OOS[date])
                    .multiply(model.factors_AB_OOS[date].T)
                    .drop(columns=["intercept"])
                    .sum(axis=1)
                ).loc[date.strftime("%Y-%m")]
            )

else:
    resid_oos = []
    for i, date in enumerate(model.factors_B_OOS):
        if i == 0:
            resid_oos.append(
                y.loc[model.factors_B_OOS[date].T.index]
                - X.loc[model.factors_B_OOS[date].T.index]
                .dot(model.gamma_B_OOS[date])
                .multiply(model.factors_B_OOS[date].T)
                .sum(axis=1)
            )
        else:
            resid_oos.append(
                (
                    y.loc[model.factors_B_OOS[date].T.index]
                    - X.loc[model.factors_B_OOS[date].T.index]
                    .dot(model.gamma_B_OOS[date])
                    .multiply(model.factors_B_OOS[date].T)
                    .sum(axis=1)
                ).loc[date.strftime("%Y-%m")]
            )

resid_oos = pd.concat(resid_oos)
resid_oos = resid_oos.sort_index()
resid_oos.to_frame("resid").to_parquet("../04_dataset/alphas/oos.pq")


# %%
