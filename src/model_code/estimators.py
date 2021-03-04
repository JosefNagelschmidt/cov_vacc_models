import numpy as np
from numba import njit
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
        thetas.append(np.copy(theta))
        thetas_nat.append(theta_tmp)
        BIC.append(BIC_lasso)

    return lamdas, thetas, thetas_nat, BIC


def adaptive_lasso(
    X,
    y,
    intercept=True,
    lamda_path=None,
    gamma_path=None,
    first_stage="Lasso",
    num_iters=100,
):

    m, p = X.shape

    if gamma_path is None:
        path_gamma = np.array([0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 6, 8])
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
            "This feature has so far only been implemented for OLS as its first-stage estimator."
        )

    coeffs[np.abs(coeffs) < 1.00e-15] = 1.00e-15

    results = []
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

        results.append(res)

    return path_gamma, results
