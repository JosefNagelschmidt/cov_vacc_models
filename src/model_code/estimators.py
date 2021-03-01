import numpy as np
from numba import njit


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
    # Calculate lambda path
    # get lambda_max
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
    lamda_path="auto",
    num_iters=100,
    intercept=True,
):

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path == "auto":
        path = get_lamda_path(X=X, y=y, epsilon=0.0001, K=100)
    else:
        path = lamda_path

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
        output["lamda"] = lamda

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
    lamda_path="auto",
    num_iters=100,
    intercept=True,
    thresh=1e-7,
):
    """Coordinate gradient descent for lasso regression - for standardized data """
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path == "auto":
        path = get_lamda_path(X=X, y=y, epsilon=0.0001, K=100)
    else:
        path = lamda_path

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
        output["lamda"] = lamda
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
    lamda_path="auto",
    num_iters=100,
    intercept=True,
    thresh=1e-7,
    warm_start=True,
):
    """Coordinate gradient descent for lasso regression - for standardized data """
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path == "auto":
        path = get_lamda_path(X=X, y=y, epsilon=0.0001, K=100)
    else:
        path = lamda_path

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
        output["lamda"] = lamda
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
    lamda_path="auto",
    num_iters=100,
    intercept=True,
    thresh=1e-7,
    active_thresh=1e-7,
    warm_start=True,
):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    if lamda_path == "auto":
        path = get_lamda_path(X=X, y=y, epsilon=0.0001, K=100)
    else:
        path = lamda_path

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
        output["lamda"] = lamda
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

    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std

    # if lamda_path == "auto":
    #    path = get_lamda_path_numba(X=X, y=y)
    # else:
    #    path = np.asarray(lamda_path_custom, dtype=float)

    path = lamda_path or get_lamda_path_numba(X=X, y=y)

    if intercept:
        X_tmp = np.ones((m, p + 1))
        X_tmp[:, 1:] = X
        X = X_tmp

    m, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    lamdas = []
    thetas = []
    thetas_nat = []

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

                    y_pred = X @ theta

                    rho = 0.0
                    z = 0.0

                    for obs in range(m):
                        rho += X[obs, j].item() * (
                            y[obs].item()
                            - y_pred[obs].item()
                            + theta[j].item() * X[obs, j].item()
                        )
                        z += np.square(X[obs, j].item())

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

                m, p = X.shape

                for subindex, j in enumerate(active_set):
                    w_j = penalty_factors[j].item()

                    y_pred = X @ theta
                    rho = 0.0
                    z = 0.0

                    for obs in range(m):
                        rho += X[obs, j].item() * (
                            y[obs].item()
                            - y_pred[obs].item()
                            + theta[j].item() * X[obs, j].item()
                        )
                        z += np.square(X[obs, j].item())

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

        lamdas.append(lamda)
        thetas.append(theta)
        thetas_nat.append(theta_tmp)

    return lamdas, thetas, thetas_nat
