from itertools import product

import numpy as np
import pandas as pd
from numba import njit
from scipy import linalg
from sklearn.linear_model import LinearRegression


@njit
def count_non_zero_coeffs(theta_vec):
    s = 0
    for i in theta_vec:
        if np.abs(i) > 1e-04:
            s += 1
    return s


def soft_threshold(rho, lamda, w):
    """Soft threshold function used for normalized data and lasso regression"""
    if rho < -lamda * w:
        return rho + lamda * w
    elif rho > lamda * w:
        return rho - lamda * w
    else:
        return 0


@njit
def soft_threshold_numba(rho, lamda, w):
    if rho < -lamda * w:
        return rho + lamda * w
    elif rho > lamda * w:
        return rho - lamda * w
    else:
        return 0.0


@njit
def get_lamda_path_numba(X, y):
    epsilon = 0.0001
    K = 100
    m, p = X.shape

    y = y.reshape((m, 1))
    sx = X
    sy = y

    lambda_max = np.max(np.abs(np.sum(sx * sy, axis=0))) / m
    lamda_path = np.exp(
        np.linspace(np.log(lambda_max), np.log(lambda_max * epsilon), np.int64(K))
    )

    return lamda_path


def get_lamda_path(X, y, epsilon=0.0001, K=100):
    m, p = X.shape

    y = y.reshape((m, 1))
    sx = X
    sy = y

    if 0.5 * m <= p:
        epsilon = 0.01

    lambda_max = np.max(np.abs(np.sum(sx * sy, axis=0))) / m
    lamda_path = np.exp(
        np.linspace(
            start=np.log(lambda_max), stop=np.log(lambda_max * epsilon), num=np.int64(K)
        )
    )

    return lamda_path


def update_coeffs(
    X, y, theta, active_set, penalty_factors, intercept, lamda, thresh, active_thresh
):
    active_set_converged_check = np.full((len(active_set),), False)
    active_set_update = np.full((len(active_set),), True)

    for subindex, j in enumerate(active_set):
        w_j = penalty_factors[j]
        X_j = X[:, j].reshape(-1, 1)

        y_pred = X @ theta
        rho = X_j.T @ (y - y_pred + theta[j] * X_j)
        z = np.sum(np.square(X_j))

        if intercept:
            if j == 0:
                tmp = rho / z
                if np.abs(tmp) < active_thresh:
                    active_set_update[subindex] = False
                if np.abs(theta[j] - tmp) < thresh:
                    active_set_converged_check[subindex] = True
                theta[j] = tmp
            else:
                tmp = (1 / z) * soft_threshold(rho, lamda, w_j)
                if np.abs(tmp) < active_thresh:
                    active_set_update[subindex] = False
                if np.abs(theta[j] - tmp) < thresh:
                    active_set_converged_check[subindex] = True
                theta[j] = tmp

        else:
            tmp = (1 / z) * soft_threshold(rho, lamda, w_j)
            if np.abs(tmp) < active_thresh:
                active_set_update[subindex] = False
            if np.abs(theta[j] - tmp) < thresh:
                active_set_converged_check[subindex] = True
            theta[j] = tmp

    active_set_converged = np.all(active_set_converged_check)
    active_set = active_set[active_set_update]

    return [theta, active_set, active_set_converged]


def naive_lasso(
    X,
    y,
    penalty_factors=None,
    theta=None,
    lamda_path=None,
    num_iters=100,
    intercept=True,
):

    m, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path is None:
        path = m * get_lamda_path_numba(X=X, y=y)
    else:
        path = m * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    m, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / m

        for _i in range(num_iters):
            for j in range(p):
                w_j = penalty_factors[j]
                X_j = X[:, j].reshape(-1, 1)

                y_pred = X @ theta
                rho = X_j.T @ (y - y_pred + theta[j] * X_j)
                z = np.sum(np.square(X_j))

                if intercept:
                    if j == 0:
                        theta[j] = rho / z
                    else:
                        theta[j] = (1 / z) * soft_threshold(rho, lamda, w_j)

                else:
                    theta[j] = (1 / z) * soft_threshold(rho, lamda, w_j)

        if not intercept:
            theta_nat = theta.flatten() / x_std * y_std
        if intercept:
            theta_0 = (
                theta.flatten()[0] - np.sum((x_mean / x_std) * theta.flatten()[1:])
            ) * y_std + y_mean
            theta_betas = theta.flatten()[1:] / x_std * y_std
            theta_nat = np.insert(arr=theta_betas, obj=0, values=theta_0)

        output["theta_std"] = theta.flatten()
        output["theta_nat"] = theta_nat
        result.append(output)
    return result


def eps_thresh_lasso(
    X,
    y,
    penalty_factors=None,
    theta=None,
    lamda_path=None,
    num_iters=100,
    intercept=True,
    thresh=1e-7,
):

    m, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path is None:
        path = m * get_lamda_path_numba(X=X, y=y)
    else:
        path = m * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    m, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / m
        tol_vals = np.full((p,), False)

        for _i in range(num_iters):
            if not np.all(tol_vals):
                for j in range(p):
                    w_j = penalty_factors[j]
                    X_j = X[:, j].reshape(-1, 1)

                    y_pred = X @ theta
                    rho = X_j.T @ (y - y_pred + theta[j] * X_j)
                    z = np.sum(np.square(X_j))

                    if intercept:
                        if j == 0:
                            if np.abs(theta[j] - rho / z) < thresh:
                                tol_vals[j] = True
                            theta[j] = rho / z
                        else:
                            if (
                                np.abs(
                                    theta[j] - (1 / z) * soft_threshold(rho, lamda, w_j)
                                )
                                < thresh
                            ):
                                tol_vals[j] = True
                            theta[j] = (1 / z) * soft_threshold(rho, lamda, w_j)

                    else:
                        if (
                            np.abs(theta[j] - (1 / z) * soft_threshold(rho, lamda, w_j))
                            < thresh
                        ):
                            tol_vals[j] = True
                        theta[j] = (1 / z) * soft_threshold(rho, lamda, w_j)
            else:
                break

        if not intercept:
            theta_nat = theta.flatten() / x_std * y_std
        if intercept:
            theta_0 = (
                theta.flatten()[0] - np.sum((x_mean / x_std) * theta.flatten()[1:])
            ) * y_std + y_mean
            theta_betas = theta.flatten()[1:] / x_std * y_std
            theta_nat = np.insert(arr=theta_betas, obj=0, values=theta_0)

        output["theta_std"] = theta.flatten()
        output["theta_nat"] = theta_nat
        result.append(output)

    return result


def eps_thresh_lasso_warm_start(
    X,
    y,
    penalty_factors=None,
    theta=None,
    lamda_path=None,
    num_iters=100,
    intercept=True,
    thresh=1e-7,
    warm_start=True,
):
    """Coordinate gradient descent for lasso regression - for standardized data """

    m, p = X.shape
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path is None:
        path = m * get_lamda_path_numba(X=X, y=y)
    else:
        path = m * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    m, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        if not warm_start:
            theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / m
        tol_vals = np.full((p,), False)

        for _i in range(num_iters):
            if not np.all(tol_vals):
                for j in range(p):
                    w_j = penalty_factors[j]
                    X_j = X[:, j].reshape(-1, 1)

                    y_pred = X @ theta
                    rho = X_j.T @ (y - y_pred + theta[j] * X_j)
                    z = np.sum(np.square(X_j))

                    if intercept:
                        if j == 0:
                            if np.abs(theta[j] - rho / z) < thresh:
                                tol_vals[j] = True
                            theta[j] = rho / z
                        else:
                            if (
                                np.abs(
                                    theta[j] - (1 / z) * soft_threshold(rho, lamda, w_j)
                                )
                                < thresh
                            ):
                                tol_vals[j] = True
                            theta[j] = (1 / z) * soft_threshold(rho, lamda, w_j)

                    else:
                        if (
                            np.abs(theta[j] - (1 / z) * soft_threshold(rho, lamda, w_j))
                            < thresh
                        ):
                            tol_vals[j] = True
                        theta[j] = (1 / z) * soft_threshold(rho, lamda, w_j)
            else:
                break

        if not intercept:
            theta_nat = theta.flatten() / x_std * y_std
        if intercept:
            theta_0 = (
                theta.flatten()[0] - np.sum((x_mean / x_std) * theta.flatten()[1:])
            ) * y_std + y_mean
            theta_betas = theta.flatten()[1:] / x_std * y_std
            theta_nat = np.insert(arr=theta_betas, obj=0, values=theta_0)

        output["theta_std"] = theta.flatten()
        output["theta_nat"] = theta_nat
        result.append(output)

    return result


def active_set_lasso(
    X,
    y,
    penalty_factors=None,
    theta=None,
    lamda_path=None,
    num_iters=100,
    intercept=True,
    thresh=1e-7,
    active_thresh=1e-7,
    warm_start=True,
):

    m, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path is None:
        path = m * get_lamda_path_numba(X=X, y=y)
    else:
        path = m * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    m, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        if not warm_start:
            theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / m
        sec_check_all_converged = False
        active_set = np.arange(p)
        active_set_converged = False

        for _i in range(num_iters):
            if (active_set.size != 0) and (not active_set_converged):
                theta, active_set, active_set_converged = update_coeffs(
                    X=X,
                    y=y,
                    theta=theta,
                    active_set=active_set,
                    penalty_factors=penalty_factors,
                    intercept=intercept,
                    lamda=lamda,
                    thresh=thresh,
                    active_thresh=active_thresh,
                )
            elif not sec_check_all_converged:
                active_set = np.arange(p)
                theta, active_set, active_set_converged = update_coeffs(
                    X=X,
                    y=y,
                    theta=theta,
                    active_set=active_set,
                    penalty_factors=penalty_factors,
                    intercept=intercept,
                    lamda=lamda,
                    thresh=thresh,
                    active_thresh=active_thresh,
                )

                if active_set_converged:
                    sec_check_all_converged = True
                    break
            else:
                break

        if not intercept:
            theta_nat = theta.flatten() / x_std * y_std
        if intercept:
            theta_0 = (
                theta.flatten()[0] - np.sum((x_mean / x_std) * theta.flatten()[1:])
            ) * y_std + y_mean
            theta_betas = theta.flatten()[1:] / x_std * y_std
            theta_nat = np.insert(arr=theta_betas, obj=0, values=theta_0)

        output["theta_std"] = theta.flatten()
        output["theta_nat"] = theta_nat
        result.append(output)

    return result


@njit
def lasso_numba(
    X,
    y,
    lamda_path=None,
    penalty_factors=None,
    theta=None,
    num_iters=100,
    intercept=True,
    thresh=1e-7,
    active_thresh=1e-7,
    warm_start=True,
):

    m, p = X.shape

    x_mean = np.zeros((p,), dtype=np.float64)

    for i in range(p):
        x_mean[i] = X[:, i].mean()

    x_std = np.zeros((p,), dtype=np.float64)

    for i in range(p):
        x_std[i] = X[:, i].std()

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_standardized = (X - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    if intercept:
        X_tmp = np.ones((m, p + 1))
        X_tmp[:, 1:] = X
        X = X_tmp

    if lamda_path is None:
        path = m * get_lamda_path_numba(X=X_standardized, y=y_standardized)
    else:
        path = m * lamda_path

    if intercept:
        X_tmp = np.ones((m, p + 1))
        X_tmp[:, 1:] = X_standardized
        X_standardized = X_tmp

    m, p = X_standardized.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    lamdas = []
    thetas = []
    thetas_nat = []
    BIC = []

    for lamda in path:
        if not warm_start:
            theta = np.zeros((p, 1))
        sec_check_all_converged = False
        active_set = np.arange(p)
        active_set_converged = False

        for _i in range(num_iters):
            if (active_set.size != 0) and (not active_set_converged):
                active_set_converged_check = np.full((len(active_set),), False)
                active_set_update = np.full((len(active_set),), True)

                for subindex, j in enumerate(active_set):
                    w_j = penalty_factors[j].item()

                    y_pred = X_standardized @ theta

                    rho = 0.0
                    z = 0.0

                    for obs in range(m):
                        rho += X_standardized[obs, j].item() * (
                            y_standardized[obs].item()
                            - y_pred[obs].item()
                            + theta[j].item() * X_standardized[obs, j].item()
                        )
                        z += np.square(X_standardized[obs, j].item())

                    if intercept:
                        if j == 0:
                            tmp = rho / z
                            if np.abs(tmp) < active_thresh:
                                active_set_update[subindex] = False
                            if np.abs(theta[j] - tmp) < thresh:
                                active_set_converged_check[subindex] = True
                            theta[j] = tmp
                        else:
                            tmp = (1 / z) * soft_threshold_numba(rho, lamda, w_j)
                            if np.abs(tmp) < active_thresh:
                                active_set_update[subindex] = False
                            if np.abs(theta[j] - tmp) < thresh:
                                active_set_converged_check[subindex] = True
                            theta[j] = tmp

                    else:
                        tmp = (1 / z) * soft_threshold_numba(rho, lamda, w_j)
                        if np.abs(tmp) < active_thresh:
                            active_set_update[subindex] = False
                        if np.abs(theta[j] - tmp) < thresh:
                            active_set_converged_check[subindex] = True
                        theta[j] = tmp

                active_set_converged = np.all(active_set_converged_check)
                active_set = active_set[active_set_update]

            elif not sec_check_all_converged:
                active_set = np.arange(p)

                active_set_converged_check = np.full((len(active_set),), False)
                active_set_update = np.full((len(active_set),), True)

                m, p = X_standardized.shape

                for subindex, j in enumerate(active_set):
                    w_j = penalty_factors[j].item()

                    y_pred = X_standardized @ theta
                    rho = 0.0
                    z = 0.0

                    for obs in range(m):
                        rho += X_standardized[obs, j].item() * (
                            y_standardized[obs].item()
                            - y_pred[obs].item()
                            + theta[j].item() * X_standardized[obs, j].item()
                        )
                        z += np.square(X_standardized[obs, j].item())

                    if intercept:
                        if j == 0:
                            tmp = rho / z
                            if np.abs(tmp) < active_thresh:
                                active_set_update[subindex] = False
                            if np.abs(theta[j] - tmp) < thresh:
                                active_set_converged_check[subindex] = True
                            theta[j] = tmp
                        else:
                            tmp = (1 / z) * soft_threshold_numba(rho, lamda, w_j)
                            if np.abs(tmp) < active_thresh:
                                active_set_update[subindex] = False
                            if np.abs(theta[j] - tmp) < thresh:
                                active_set_converged_check[subindex] = True
                            theta[j] = tmp

                    else:
                        tmp = (1 / z) * soft_threshold_numba(rho, lamda, w_j)
                        if np.abs(tmp) < active_thresh:
                            active_set_update[subindex] = False
                        if np.abs(theta[j] - tmp) < thresh:
                            active_set_converged_check[subindex] = True
                        theta[j] = tmp

                active_set_converged = np.all(active_set_converged_check)
                active_set = active_set[active_set_update]

                if active_set_converged:
                    sec_check_all_converged = True
                    break
            else:
                break

        if not intercept:
            theta_tmp = theta.flatten() / x_std * y_std
        if intercept:
            theta_0 = (
                theta.flatten()[0] - np.sum((x_mean / x_std) * theta.flatten()[1:])
            ) * y_std + y_mean
            theta_betas = theta.flatten()[1:] / x_std * y_std
            theta_tmp = np.ones((p,))
            theta_tmp[1:] = theta_betas
            theta_tmp[0] = theta_0

        m, p = X.shape
        theta_bic = np.ones((p, 1))
        theta_bic[:, 0] = theta_tmp
        residuals_hat = np.sum(np.square(y - X @ theta_bic))
        df_lamda = count_non_zero_coeffs(theta_vec=theta_bic.flatten())
        BIC_lasso = residuals_hat / (m * y_std ** 2) + np.log(m) / m * df_lamda

        lamdas.append(lamda / m)
        thetas.append(np.copy(theta).flatten())
        thetas_nat.append(theta_tmp)
        BIC.append(BIC_lasso)

    return lamdas, thetas, thetas_nat, BIC


def adaptive_lasso(
    X,
    y,
    intercept=True,
    lamda_path=None,
    gamma_path=None,
    first_stage="OLS",
    num_iters=100,
    out_as_df=True,
):

    m, p = X.shape

    if gamma_path is None:
        path_gamma = np.array([0.1, 0.5, 1, 2, 3, 4, 6, 8])
    else:
        path_gamma = gamma_path

    if first_stage == "OLS":
        reg = LinearRegression(fit_intercept=intercept).fit(X, y)
        coeffs = reg.coef_.T
    elif first_stage == "Lasso":
        res = lasso_numba(X=X, y=y)

        index_lamda_opt = np.where(res[3] == np.amin(res[3]))[0][0]
        coeffs = np.delete(res[1][index_lamda_opt], 0).reshape((p, 1))

    else:
        raise AssertionError(
            "This feature has so far only been implemented for OLS and Lasso as its first-stage estimators."
        )

    coeffs[np.abs(coeffs) < 1.00e-15] = 1.00e-15

    results = []
    weight_path = []
    for gamma in path_gamma:

        if intercept:
            weights = np.ones((p + 1, 1))
            weights[1:, :] = 1.0 / np.abs(coeffs) ** gamma
        else:
            weights = 1.0 / np.abs(coeffs) ** gamma

        res = lasso_numba(
            X,
            y,
            lamda_path=lamda_path,
            penalty_factors=weights,
            theta=None,
            num_iters=num_iters,
            intercept=intercept,
            thresh=1e-7,
            active_thresh=1e-7,
            warm_start=True,
        )

        weight_path.append(weights)
        results.append(res)

    if out_as_df:
        lamda_p = results[0][0]
        df = pd.DataFrame(
            list(product(path_gamma, lamda_p)), columns=["gamma", "lamda"]
        )
        df["theta_std"] = np.nan
        df["theta_nat"] = np.nan
        df["gamma_weights"] = np.nan
        df = df.astype(object)
        df = df.set_index(["gamma", "lamda"])

        for id_gamma, gamma in enumerate(path_gamma):
            for idx, lamda in enumerate(results[id_gamma][0]):
                index = (gamma, lamda)
                df.at[index, "theta_std"] = results[id_gamma][1][idx]
                df.at[index, "theta_nat"] = results[id_gamma][2][idx]
                df.at[index, "gamma_weights"] = weight_path[id_gamma]
        return df

    else:
        return path_gamma, results, weight_path


def get_conf_intervals(
    lamda, weights, theta_std, theta_nat, X, X_std, intercept, y, y_std
):
    n, p = X.shape
    if intercept:
        X_with_intercept = np.insert(X, 0, 1, axis=1)
        X_std_with_intercept = np.insert(X_std, 0, 1, axis=1)

        sigma_hat_nat = (
            1
            / n
            * np.sum(
                np.square(y - X_with_intercept @ theta_nat.reshape((len(theta_nat), 1)))
            )
        )
        sigma_hat_std = (
            1
            / n
            * np.sum(
                np.square(
                    y_std
                    - X_std_with_intercept @ theta_std.reshape((len(theta_std), 1))
                )
            )
        )
    else:
        sigma_hat_nat = (
            1 / n * np.sum(np.square(y - X @ theta_nat.reshape((len(theta_nat), 1))))
        )
        sigma_hat_std = (
            1
            / n
            * np.sum(np.square(y_std - X_std @ theta_std.reshape((len(theta_std), 1))))
        )

    if intercept:
        theta_std = np.delete(arr=theta_std, obj=0)
        theta_nat = np.delete(arr=theta_nat, obj=0)
        weights = np.delete(arr=weights, obj=0, axis=0)

    weights = weights.flatten()

    active_set = np.invert(np.isclose(np.zeros(p), theta_nat, atol=1e-06))

    X_active = X[:, active_set]
    X_std_active = X_std[:, active_set]
    theta_nat_active = theta_nat[active_set]
    theta_std_active = theta_std[active_set]
    weights_active = weights[active_set]

    diag_std = weights_active / theta_std_active
    diag_nat = weights_active / theta_nat_active

    sigma_beta_std = np.diag(v=diag_std, k=0)
    sigma_beta_nat = np.diag(v=diag_nat, k=0)

    main_mat_nat = X_active.T @ X_active + lamda * sigma_beta_nat
    main_mat_std = X_std_active.T @ X_std_active + lamda * sigma_beta_std

    main_mat_nat_inverse = linalg.inv(main_mat_nat)
    main_mat_std_inverse = linalg.inv(main_mat_std)

    beta_hat_nat_cov_mat = sigma_hat_nat * (
        main_mat_nat_inverse @ X_active.T @ X_active @ main_mat_nat_inverse
    )
    beta_hat_std_cov_mat = sigma_hat_std * (
        main_mat_std_inverse @ X_std_active.T @ X_std_active @ main_mat_std_inverse
    )

    conf_intervals_nat_upper_bound = theta_nat_active + 1.96 * np.sqrt(
        np.diag(beta_hat_nat_cov_mat)
    )
    conf_intervals_nat_lower_bound = theta_nat_active - 1.96 * np.sqrt(
        np.diag(beta_hat_nat_cov_mat)
    )

    conf_intervals_nat = np.column_stack(
        (conf_intervals_nat_lower_bound, conf_intervals_nat_upper_bound)
    )

    conf_intervals_std_upper_bound = theta_std_active + 1.96 * np.sqrt(
        np.diag(beta_hat_std_cov_mat)
    )
    conf_intervals_std_lower_bound = theta_std_active - 1.96 * np.sqrt(
        np.diag(beta_hat_std_cov_mat)
    )

    conf_intervals_std = np.column_stack(
        (conf_intervals_std_lower_bound, conf_intervals_std_upper_bound)
    )

    return {
        "beta_hat_nat_cov_mat": beta_hat_nat_cov_mat,
        "beta_hat_std_cov_mat": beta_hat_std_cov_mat,
        "conf_intervals_nat": conf_intervals_nat,
        "conf_intervals_std": conf_intervals_std,
        "active_set": active_set,
    }


def make_prediction(X, y, theta_nat, intercept=True):
    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    y_hat = X @ theta_nat.reshape((X.shape[1], 1))
    mse = np.sum(np.square(y - y_hat))
    return y_hat, mse


def cv_adaptive_lasso(X, y, intercept=True, first_stage="OLS"):
    n, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_standardized = (X - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    indices = np.random.permutation(n)
    fold_1_idx, fold_2_idx = indices[: int(n / 2)], indices[int(n / 2) :]
    X_fold_1, X_fold_2 = X[fold_1_idx, :], X[fold_2_idx, :]
    y_fold_1, y_fold_2 = y[fold_1_idx, :], y[fold_2_idx, :]

    gamma_path = np.array([0.5, 1, 2, 3, 4, 6, 8, 10])
    lamda_path = get_lamda_path(X=X_standardized, y=y_standardized)

    trained_on_fold_1 = adaptive_lasso(
        X=X_fold_1,
        y=y_fold_1,
        intercept=True,
        lamda_path=lamda_path,
        gamma_path=gamma_path,
        first_stage=first_stage,
        num_iters=100,
        out_as_df=True,
    )

    trained_on_fold_2 = adaptive_lasso(
        X=X_fold_2,
        y=y_fold_2,
        intercept=True,
        lamda_path=lamda_path,
        gamma_path=gamma_path,
        first_stage=first_stage,
        num_iters=100,
        out_as_df=True,
    )

    trained_on_fold_1["mse_1"] = np.nan
    trained_on_fold_2["mse_2"] = np.nan

    prod = product(
        trained_on_fold_1.index.get_level_values("gamma").unique(),
        trained_on_fold_1.index.get_level_values("lamda").unique(),
    )
    for gamma, lamda in prod:
        index = (gamma, lamda)
        y_hat_1, mse_1 = make_prediction(
            X=X_fold_2,
            y=y_fold_2,
            theta_nat=trained_on_fold_1.at[index, "theta_nat"],
            intercept=intercept,
        )

        y_hat_2, mse_2 = make_prediction(
            X=X_fold_1,
            y=y_fold_1,
            theta_nat=trained_on_fold_2.at[index, "theta_nat"],
            intercept=intercept,
        )

        trained_on_fold_1.at[index, "mse_1"] = mse_1
        trained_on_fold_2.at[index, "mse_2"] = mse_2

    cv_overview = trained_on_fold_1.merge(
        trained_on_fold_2, how="left", on=["gamma", "lamda"]
    )[["mse_1", "mse_2"]]
    cv_overview["mean_mse"] = cv_overview.mean(axis=1)

    params_opt = cv_overview.iloc[
        cv_overview["mean_mse"].argmin(),
    ].name

    return cv_overview, params_opt


def adaptive_lasso_tuned(X, y, first_stage="OLS"):
    n, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_standardized = (X - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    cv_results, params_opt = cv_adaptive_lasso(
        X=X, y=y, intercept=True, first_stage=first_stage
    )
    gamma_opt = params_opt[0]
    lamda_opt = params_opt[1]

    train_opt_ada_lasso = adaptive_lasso(
        X=X,
        y=y,
        intercept=True,
        lamda_path=np.array([lamda_opt]),
        gamma_path=np.array([gamma_opt]),
        first_stage=first_stage,
        num_iters=100,
        out_as_df=True,
    )

    ada_lasso_opt_res = get_conf_intervals(
        lamda=lamda_opt,
        weights=train_opt_ada_lasso.iloc[0]["gamma_weights"],
        theta_std=train_opt_ada_lasso.iloc[0]["theta_std"],
        theta_nat=train_opt_ada_lasso.iloc[0]["theta_nat"],
        X=X,
        X_std=X_standardized,
        intercept=True,
        y=y,
        y_std=y_standardized,
    )

    selected_support = ada_lasso_opt_res["active_set"]
    conf_intervals_nat = ada_lasso_opt_res["conf_intervals_nat"]
    conf_intervals_std = ada_lasso_opt_res["conf_intervals_std"]

    return {
        "selected_support": selected_support,
        "conf_intervals_nat": conf_intervals_nat,
        "conf_intervals_std": conf_intervals_std,
    }
