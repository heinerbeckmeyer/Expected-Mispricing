# %%
# Packages
import pandas as pd
import numpy as np
from numba import njit

from copy import deepcopy

from time import gmtime, strftime, time


def flush_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


# %%
class MultiIPCA:
    def __init__(self, n_factors, n_iter, tol, n_jobs=1):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.fit_func = self._fit_cmp

        self.gamma_B_OOS = {}
        self.factors_B_OOS = {}

    def fit(self, X, y, sqrt_weights: pd.DataFrame = None, n_verbose: int = 1):
        # --- fit model without intercept
        gamma_new, f_new = self.fit_func(
            X=X,
            y=y,
            intercept=False,
            sqrt_weights=sqrt_weights,
            initial_gamma=None,
            initial_factors=None,
            n_verbose=n_verbose,
        )
        self.factors_B = f_new.copy()
        self.gamma_B = pd.DataFrame(gamma_new, columns=self.factors_B.index, index=X.columns)

        # --- fit model with intercept
        initial_gamma = self.gamma_B.copy()
        initial_gamma["intercept"] = 0

        gamma_new, f_new = self.fit_func(
            X=X,
            y=y,
            intercept=True,
            sqrt_weights=sqrt_weights,
            initial_gamma=initial_gamma,
            initial_factors=self.factors_B.copy(),
            n_verbose=n_verbose,
        )
        self.factors_AB = f_new.copy()
        self.gamma_AB = pd.DataFrame(gamma_new, columns=self.factors_AB.index, index=X.columns)

    def _fit_cmp(
        self,
        X,
        y,
        intercept: bool,
        initial_gamma=None,
        initial_factors=None,
        n_verbose: int = 1,
        **kwargs,
    ):
        start_time = time()

        # --- obtain CMPs for stacked characteristics

        # obtain CMPs (Q/W):
        Q = (
            X.multiply(y, axis=0).groupby("date").mean()
        )  # NOTE: missing entries should be nan, otherwise mean will be skewed
        W = X.groupby("date").apply(lambda x: x.fillna(0).T.dot(x.fillna(0)) / (~x.isnull()).sum())
        # ---

        Ns = X.groupby("date").size()

        # --- initial guess for Gammas
        if initial_gamma is None:
            if n_verbose == 1:
                print("Initial guess.")
            gamma_old, s, v = self.numba_svd(Q.T.to_numpy())
            gamma_old = gamma_old[:, : self.n_factors]
            if intercept:
                gamma_old = np.concatenate((gamma_old, np.zeros((len(gamma_old), 1))), 1)
            s = s[: self.n_factors]
            v = v[: self.n_factors, :]
            f_old = np.diag(s).dot(v)
        else:
            gamma_old = initial_gamma.to_numpy()

        # --- obtain all dates in sample
        # dates = Xs_stacked.index.get_level_values("date").unique()
        dates = y.index.get_level_values("date").unique()

        if n_verbose > 0:
            flush_print(f"Joint IPCA model with n_factors={self.n_factors}; intercept={intercept}")
            flush_print(f"Dimensions of Q: {Q.shape} / W: {W.shape}")

        # --- if intercept should be fitted, use as PSF
        if intercept:
            PSF = pd.DataFrame(np.ones((len(dates))), index=dates, columns=["intercept"])
        else:
            PSF = None

        # --- Fit Factors and Gammas
        for iter in range(self.n_iter):
            current_time = time()

            # --- Fit Factors
            # NOTE: same factors for all asset classes
            Fs = []
            for date in dates:
                m1 = gamma_old[:, : self.n_factors].T.dot(W.loc[date]).dot(gamma_old[:, : self.n_factors])
                m2 = gamma_old[:, : self.n_factors].T.dot(Q.loc[date])
                # if PSF, subtract their influence on returns
                if PSF is not None:
                    m2 -= (
                        gamma_old[:, : self.n_factors]
                        .T.dot(W.loc[date])
                        .dot(gamma_old[:, self.n_factors :])
                        .dot(PSF.loc[date])
                    )
                Fs.append(self.numba_solve(m1, m2.reshape(-1, 1)))
            Fs = np.concatenate(Fs, axis=1)
            Fs = pd.DataFrame(data=Fs, columns=dates)
            f_new = Fs

            if iter == 0:
                if initial_factors is not None:
                    f_old = initial_factors.copy()
                else:
                    f_old = f_new.copy()

            # --- Fit Gamma
            numer = 0.0
            denom = 0.0
            for date in dates:
                if PSF is not None:
                    stacked = np.vstack(
                        (f_new.loc[:, date].to_numpy().reshape((-1, 1)), PSF.loc[date].to_numpy().reshape((-1, 1)))
                    )
                    numer += np.kron(Q.loc[date].to_numpy().reshape((-1, 1)), stacked) * Ns.loc[date]
                    denom += np.kron(W.loc[date], stacked.dot(stacked.T)) * Ns.loc[date]

                else:
                    numer += (
                        np.kron(
                            Q.loc[date].to_numpy().reshape((-1, 1)),
                            f_new.loc[:, date].to_numpy().reshape((-1, 1)),
                            # ) * len(Xs_stacked.loc[date])
                        )
                        * Ns.loc[date]
                    )
                    denom += (
                        np.kron(
                            W.loc[date],
                            f_new.loc[:, date]
                            .to_numpy()
                            .reshape((-1, 1))
                            .dot(f_new.loc[:, date].to_numpy().reshape((1, -1))),
                            # ) * len(Xs_stacked.loc[date])
                        )
                        * Ns.loc[date]
                    )
            gamma_new = self.numba_solve(denom, numer).reshape((Q.shape[1], -1))

            # --- Enforce Orthogonality
            if self.n_factors > 0:
                # Enforce Orthogonality in Gamma_alpha and Gamma_beta
                R1 = self.numba_chol(gamma_new[:, : self.n_factors].T.dot(gamma_new[:, : self.n_factors])).T
                R2, _, _ = self.numba_svd(R1.dot(f_new).dot(f_new.T).dot(R1.T))

                # first, orthogonalize factors to ensure the use of the same set of factors per Gamma
                _f_new = self.numba_solve(R2, R1.dot(f_new))
                f_new = pd.DataFrame(_f_new, index=f_new.index, columns=f_new.columns)

                # then, orthonormalize gamma
                gamma_new[:, : self.n_factors] = self.numba_lstsq(gamma_new[:, : self.n_factors].T, R1.T)[0].dot(R2)

                # Enforce sign convention for Gamma_Beta and F_New
                sg = np.sign(np.mean(f_new.to_numpy(), axis=1)).reshape((-1, 1))
                sg[sg == 0] = 1
                gamma_new[:, : self.n_factors] = np.multiply(gamma_new[:, : self.n_factors], sg.T)
                f_new = f_new * sg

                if PSF is not None:
                    gamma_new[:, self.n_factors :] = (
                        np.identity(gamma_new.shape[0])
                        - gamma_new[:, : self.n_factors].dot(gamma_new[:, : self.n_factors].T)
                    ).dot(gamma_new[:, self.n_factors :])
                    f_new += gamma_new[:, : self.n_factors].T.dot(gamma_new[:, self.n_factors :]).dot(PSF.T)

                    # Enforce sign convention for Gamma_Beta and F_New
                    sg = np.sign(np.mean(f_new.to_numpy(), axis=1)).reshape((-1, 1))
                    sg[sg == 0] = 1
                    gamma_new[:, : self.n_factors] = np.multiply(gamma_new[:, : self.n_factors], sg.T)
                    f_new = f_new * sg

            # tolerance
            tol_current_G = np.max(np.abs(gamma_new - gamma_old))  # relative Gamma!
            tol_current_F = np.max((np.abs(f_new - f_old)).to_numpy())
            tol_current = max(np.max(tol_current_G), tol_current_F)
            if n_verbose > 0:
                if n_verbose > 1:
                    if (iter % n_verbose == 0) and (iter > 0):
                        flush_print(
                            f"{strftime('%Y-%m-%d %H:%M:%S', gmtime())}; {iter}: "
                            + f"Current tol: {tol_current:.5f}/{tol_current_G:.5f}/{tol_current_F:.5f} ({time() - current_time:.0f}s elapsed)"
                        )
                else:
                    flush_print(
                        f"{strftime('%Y-%m-%d %H:%M:%S', gmtime())}; {iter}: "
                        + f"Current tol: {tol_current:.5f}/{tol_current_G:.5f}/{tol_current_F:.5f} ({time() - current_time:.0f}s elapsed)"
                    )

            gamma_old = gamma_new.copy()
            f_old = f_new.copy()

            if iter >= self.n_iter:
                if n_verbose > 0:
                    flush_print("Exceeded max number of iterations.\n\t== Done ==")
                break

            if tol_current <= self.tol:
                if n_verbose > 0:
                    flush_print("Minimum tolerance reached.\n\t== Done ==")
                break

        if n_verbose > 0:
            flush_print(f"Total time elapsed: {time() - start_time:.0f}s; {iter} iterations required.")

        if PSF is not None:
            f_new = pd.concat((f_new, PSF.T))

        return gamma_new, f_new

    def cluster_oos_fit(self, X, y, date, intercept: bool, n_verbose: int = 0):
        tmp_X = X.loc[: date - pd.DateOffset(months=1)].copy()
        tmp_y = y.loc[: date - pd.DateOffset(months=1)].copy()

        # --- fit model intercept
        if intercept:
            gamma_new, f_new = self.fit_func(
                X=tmp_X,
                y=tmp_y,
                intercept=True,
                initial_gamma=self.gamma_AB if hasattr(self, "gamma_AB") else None,
                initial_factors=None,
                n_verbose=n_verbose,
            )
        else:
            gamma_new, f_new = self.fit_func(
                X=tmp_X,
                y=tmp_y,
                intercept=False,
                initial_gamma=self.gamma_B if hasattr(self, "gamma_B") else None,
                initial_factors=None,
                n_verbose=n_verbose,
            )
        gamma_new = pd.DataFrame(gamma_new, columns=self.factors_B.index, index=X.columns)

        # --- add factor realization for t+1 through cross-sectional regression
        exog = X.dot(gamma_new)

        oos_factors = self.numba_lstsq(
            exog.loc[date].iloc[:, : self.n_factors].to_numpy(),
            y.loc[date].to_numpy()[:, None],
        )[0].squeeze()
        if intercept:
            oos_factors = np.append(oos_factors, [1])

        # save t+1 factor realizations
        f_new[date] = oos_factors.copy()

        return f_new, gamma_new

    def score(self, X, y, intercept: bool):
        def r2(yp, y):
            return 1 - (yp - y).pow(2).sum() / y.pow(2).sum()

        ypred, _ = self.pred(X, y, intercept=intercept)

        r2s = r2(ypred, y.loc[ypred.index])

        return {"R2": r2s}

    def pred(self, X, y, intercept: bool):
        if intercept:
            ypred = X.dot(self.gamma_AB).multiply(self.factors_AB.T).sum(axis=1)
            resid = y - ypred
        else:
            ypred = X.dot(self.gamma_B).multiply(self.factors_B.T).sum(axis=1)
            resid = y - ypred
        return ypred, resid

    @staticmethod
    @njit(nogil=True)
    def numba_svd(Q):
        gamma_old, s, v = np.linalg.svd(Q)
        return gamma_old, s, v

    @staticmethod
    @njit(nogil=True)
    def numba_lstsq(X, y):
        return np.linalg.lstsq(X, y)

    @staticmethod
    @njit(nogil=True)
    def numba_chol(X):
        return np.linalg.cholesky(X)

    @staticmethod
    @njit(nogil=True)
    def numba_solve(a, b):
        return np.linalg.solve(a, b)

    @staticmethod
    @njit(nogil=True)
    def numba_kron(a, b):
        return np.kron(np.ascontiguousarray(a), np.ascontiguousarray(b))

    @staticmethod
    def load_data():
        pd.options.mode.chained_assignment = None

        data = pd.read_parquet("../04_dataset/jkp_1973_filtered.pq")
        data = data.dropna(subset=["ret_exc_lead1m"])
        data["const"] = 1

        data = data.sort_index()

        X = data.drop(columns=["ret_exc_lead1m"])

        pct_available = (~X.isnull()).groupby("date").mean().mean()

        X = X.fillna(0)  # missing char = 0
        y = data["ret_exc_lead1m"]

        print(f"Total shape of characteristics: {X.shape}")

        # loop through features and eliminate those that are > 95% correlated
        # when two features are found, retain the one more often available
        dropped_cols = []
        corr = X.corr().abs()
        np.fill_diagonal(corr.values, np.nan)
        corr = pd.DataFrame(np.triu(corr), corr.index, corr.columns)
        corr[corr == 0] = np.nan
        corr = corr.stack()
        corr = corr.sort_values(ascending=False)

        do_loop = True
        while do_loop:
            for (char1, char2), val in corr.head(1).items():
                if val > 0.95:
                    f_row = pct_available.loc[char1]
                    f_col = pct_available.loc[char2]
                    if f_row > f_col:
                        print(f"Correlation of {val:.2f} between {char1} and {char2}. Dropping {char2}.")
                        dropped_cols.append(char2)
                        if char2 in X.columns:
                            X = X.drop(columns=char2)
                    else:
                        print(f"Correlation of {val:.2f} between {char1} and {char2}. Dropping {char1}.")
                        if char1 in X.columns:
                            X = X.drop(columns=char1)
                        dropped_cols.append(char1)
            corr = corr.iloc[1:]
            do_loop = (corr > 0.95).any()
        dropped_cols = list(set(dropped_cols))
        print(f"Dropped columns (n={len(dropped_cols)}):\n{dropped_cols}")
        print(f"Total shape of characteristics after dropping correlated characteristics: {X.shape}")

        return X, y


# %%
