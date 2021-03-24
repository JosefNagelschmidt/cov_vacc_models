from itertools import product

import numpy as np
import pandas as pd
from numba import njit
from scipy import linalg
from sklearn.linear_model import LinearRegression


def standardize_input(X, y):
    """Standardizes the regressor matrix and the dependent vector.
    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1) or (n, )

    Returns:
        tuple: tuple containing:

            **X_standardized** (*np.ndarray*): standardized regressor matrix of shape (n, p) \n
            **y_standardized** (*np.ndarray*): standardized vector of the dependent variable *y*, of shape (n, 1) or (n, ) \n
            **x_mean** (*np.ndarray*): column means of the regressor matrix of shape (p, ) \n
            **x_std** (*np.ndarray*): column standard deviations of the regressor matrix of shape (p, ) \n
            **y_mean** (*np.float64*): mean of the vector of the dependent variable *y* \n
            **y_std** (*np.float64*): standard deviation of the vector of the dependent variable *y*

    """

    if y.ndim == 1:
        y = y.reshape((len(y), 1))

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_standardized = (X - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    return X_standardized, y_standardized, x_mean, x_std, y_mean, y_std


@njit
def count_non_zero_coeffs(theta_vec):

    """Determines the cardinality of the non-zero elements of a given input vector *theta_vec*.

    Args:
        theta_vec (np.ndarray): 1d array (p, ) representation of a coefficient vector

    Returns:
        **s** (*int*): cardinality of the non-zero elements of *theta_vec*

    """

    s = 0
    for i in theta_vec:
        if np.abs(i) > 1e-04:
            s += 1
    return s


def soft_threshold(rho, lamda, w):

    """Soft threshold function used for standardized data within the lasso regression.

    Args:
        rho (float): defined as :math:`\\rho := X_j^T (y - y_{pred} + \\theta_j \\cdot X_j)`,
            where :math:`X_j` is the *j*-th column of the regressor matrix, *y* is the dependent variable
            vector, :math:`y_{pred} := X \\cdot \\theta` (projection), and :math:`\\theta` the coefficient vector.
        lamda (float): non-negative regularization parameter in the l1-penalty term of the lasso
        w (float): weight for a given coefficient within the l1-penalty term of the adaptive lasso

    Returns:
        (*float*): proximal mapping of the l1-norm; solution of coordinate descent for the update step in lasso

    """

    if rho < -lamda * w:
        return rho + lamda * w
    elif rho > lamda * w:
        return rho - lamda * w
    else:
        return 0.0


@njit
def soft_threshold_numba(rho, lamda, w):

    """Just-in-time compiled version of the *soft_threshold* function used within the lasso regression"""

    if rho < -lamda * w:
        return rho + lamda * w
    elif rho > lamda * w:
        return rho - lamda * w
    else:
        return 0.0


@njit
def get_lamda_path_numba(X_std, y_std):

    """Just-in-time compiled version of the *get_lamda_path* function used within the lasso regression"""

    epsilon = 0.0001
    K = 100

    n, p = X_std.shape

    y_std = y_std.reshape((n, 1))

    if 0.5 * n <= p:
        epsilon = 0.01

    lambda_max = np.max(np.abs(np.sum(X_std * y_std, axis=0))) / n
    lamda_path = np.exp(
        np.linspace(np.log(lambda_max), np.log(lambda_max * epsilon), np.int64(K))
    )

    return lamda_path


def get_lamda_path(X_std, y_std, epsilon=0.0001, K=100):

    """Calculates a data-dependent sequence of lambdas, for which we want to solve the lasso optimization problem.
    This approach follows the one used in *glmnet* package in R.

    Args:
        X_std (np.ndarray): standardized regressor matrix of shape (n, p)
        y_std (np.ndarray): standardized vector of the dependent variable *y*, of shape (n, 1) or (n, )
        epsilon (float): parameter determining the lower bound of the lambda sequence
        K (int): parameter determining the number of elements within the lambda sequence

    Returns:
        **lamda_path** (*np.ndarray*): data-dependent sequence of lambdas (optimization path for lasso)

    """

    n, p = X_std.shape

    y_std = y_std.reshape((n, 1))

    # to ensure that matrix inversion is stable in case that p > n
    if 0.5 * n <= p:
        epsilon = 0.01

    # data-dependent part from glmnet
    lambda_max = np.max(np.abs(np.sum(X_std * y_std, axis=0))) / n

    # transformation such that we get many lambda elements close to zero, and a few large ones
    lamda_path = np.exp(
        np.linspace(
            start=np.log(lambda_max), stop=np.log(lambda_max * epsilon), num=np.int64(K)
        )
    )

    return lamda_path


def update_coeffs(
    X_std,
    y_std,
    theta,
    active_set,
    penalty_factors,
    intercept,
    lamda,
    thresh,
    active_thresh,
):
    """Calculates the update of the coefficients within each loop of *active_set_lasso*.

    Args:
        X_std (np.ndarray): standardized regressor matrix of shape (n, p)
        y_std (np.ndarray): standardized vector of the dependent variable *y*, of shape (n, 1)
        theta (np.ndarray): vector of coefficients of shape (p, 1)
        active_set (np.ndarray): indeces of coefficients to consider in the update, i.e. these
            coefficients are not zero and still active
        penalty_factors (np.ndarray): vector of penalties that function as weights for the coefficients
            within the l1-penalty term of the adaptive lasso; later used in the *soft_threshold* function.
            The shape must be (p, 1)
        intercept (bool): logical value whether an intercept should be used when fitting (adaptive) lasso
        lamda (float): non-negative regularization parameter in the l1-penalty term of the lasso
        thresh (float): threshold for determining whether the update was small enough to classify the coefficient
            as converged
        active_thresh (float): threshold for determining whether the coefficient is still different enough from zero
            to be considered active

    Returns:
        tuple: tuple containing:

            **theta** (*np.ndarray*): updated vector of coefficients of shape (p, 1) \n
            **active_set** (*np.ndarray*): indeces of coefficients to consider in the next cycle, i.e. updated version of *active_set* \n
            **active_set_converged** (*bool*): Logical value whether all ex ante active coefficients have converged within this cycle

    """

    # set up two logical vectors for the active_set classication later
    active_set_converged_check = np.full((len(active_set),), False)
    active_set_update = np.full((len(active_set),), True)

    # main update step
    for subindex, j in enumerate(active_set):
        w_j = penalty_factors[j]
        X_j = X_std[:, j].reshape(-1, 1)

        y_pred = X_std @ theta
        rho = X_j.T @ (y_std - y_pred + theta[j] * X_j)
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

    # test whether all ex ante active coefficients have converged within this cycle
    active_set_converged = np.all(active_set_converged_check)

    # remove coefficients from the active set that were too close to zero
    active_set = active_set[active_set_update]

    return theta, active_set, active_set_converged


def naive_lasso(
    X,
    y,
    penalty_factors=None,
    theta=None,
    lamda_path=None,
    num_iters=100,
    intercept=True,
):
    """Naive coordinate descent implementation of the basic lasso optimization problem without stopping criterion
    except the maximum number of iterations.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        penalty_factors (np.ndarray): vector of penalties that function as weights for the coefficients
            within the l1-penalty term of the adaptive lasso; later used in the *soft_threshold* function.
            The shape must be (p, 1). If none are provided, the function defaults to providing a vector of ones,
            which is the standard lasso version.
        theta (np.ndarray): initial starting values for the vector of coefficients of shape (p, 1).
            If none are provided, the function defaults to setting each coefficient to zero initially.
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for.
            If none are provided, the function provides a data-dependent sequence by default.
        num_iters (int): Maximum number of cycles to update the coefficients; usually convergence is
            reached in under 100 iterations. Defaults to 100.
        intercept (bool): logical value whether an intercept should be used when fitting (adaptive) lasso

    Returns:
        list: list of dicts, each containing:

            **lamda** (*float*): non-negative regularization parameter in the l1-penalty term of the lasso and element of lamda_path \n
            **theta_std** (*np.ndarray*): optimal lasso coefficients on the standardized scale for the given lambda in the dictionary, shape (p, ) \n
            **theta_nat** (*np.ndarray*): optimal lasso coefficients on the original scale for the given lambda in the dictionary, shape (p, )
    """

    n, p = X.shape

    X, y, x_mean, x_std, y_mean, y_std = standardize_input(X=X, y=y)

    if lamda_path is None:
        path = n * get_lamda_path_numba(X_std=X, y_std=y)
    else:
        path = n * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    n, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / n

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
    """Improved coordinate descent implementation of the basic lasso optimization problem with
    threshold as stopping criterion. Cycling is stopped if the absolute difference between each updated
    theta and its former value is below the threshold *thresh*.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        penalty_factors (np.ndarray): vector of penalties that function as weights for the coefficients
            within the l1-penalty term of the adaptive lasso; later used in the *soft_threshold* function.
            The shape must be (p, 1). If none are provided, the function defaults to providing a vector of ones,
            which is the standard lasso version.
        theta (np.ndarray): initial starting values for the vector of coefficients of shape (p, 1). If none
            are provided, the function defaults to setting each coefficient to zero initially.
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for. If none are
            provided, the function provides a data-dependent sequence by default.
        num_iters (int): Maximum number of cycles to update the coefficients; usually convergence is reached
            in under 100 iterations. Defaults to 100.
        intercept (bool): logical value whether an intercept should be used when fitting (adaptive) lasso
        thresh (float): determines the relevant threshold for the  absolute difference between each updated
            theta and its former value


    Returns:
        list: list of dicts, each containing:

            **lamda** (*float*): non-negative regularization parameter in the l1-penalty term of the lasso and element of lamda_path \n
            **theta_std** (*np.ndarray*): optimal lasso coefficients on the standardized scale for the given lambda in the dictionary, shape (p, ) \n
            **theta_nat** (*np.ndarray*): optimal lasso coefficients on the original scale for the given lambda in the dictionary, shape (p, )
    """

    n, p = X.shape
    X, y, x_mean, x_std, y_mean, y_std = standardize_input(X=X, y=y)

    if lamda_path is None:
        path = n * get_lamda_path_numba(X_std=X, y_std=y)
    else:
        path = n * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    n, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / n
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
        output["theta_nat"] = theta_nat.flatten()
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
    """Further improved coordinate descent implementation of the basic lasso optimization problem with threshold as
    stopping criterion and the usage of *warm_starts*. Cycling is stopped if the absolute difference between each updated
    theta and its former value is below the threshold *thresh*, and warm starts reuse the previously learned optimal
    coefficients as starting values for theta in the optimization for the next lambda element in the *lamda_path*.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        penalty_factors (np.ndarray): vector of penalties that function as weights for the coefficients within
            the l1-penalty term of the adaptive lasso; later used in the *soft_threshold* function. The shape must
            be (p, 1). If none are provided, the function defaults to providing a vector of ones, which is
            the standard lasso version.
        theta (np.ndarray): initial starting values for the vector of coefficients of shape (p, 1).
            If none are provided, the function defaults to setting each coefficient to zero initially.
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for.
            If none are provided, the function provides a data-dependent sequence by default.
        num_iters (int): Maximum number of cycles to update the coefficients; usually convergence is reached in
            under 100 iterations. Defaults to 100.
        intercept (bool): logical value whether an intercept should be used when fitting (adaptive) lasso
        thresh (float): threshold for determining whether the update was small enough to classify
            the coefficient as converged
        warm_start (bool): Logical value determining whether the *warm_starts* feature should be used.


    Returns:
        list: list of dicts, each containing:

            **lamda** (*float*): non-negative regularization parameter in the l1-penalty term of the lasso and element of lamda_path \n
            **theta_std** (*np.ndarray*): optimal lasso coefficients on the standardized scale for the given lambda in the dictionary, shape (p, ) \n
            **theta_nat** (*np.ndarray*): optimal lasso coefficients on the original scale for the given lambda in the dictionary, shape (p, )
    """

    n, p = X.shape
    X, y, x_mean, x_std, y_mean, y_std = standardize_input(X=X, y=y)

    if lamda_path is None:
        path = n * get_lamda_path_numba(X_std=X, y_std=y)
    else:
        path = n * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    n, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        if not warm_start:
            theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / n
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
        output["theta_nat"] = theta_nat.flatten()
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
    """Even more improved coordinate descent implementation of the lasso optimization problem with threshold as
    stopping criterion, the usage of *warm_starts*, and the usage of *active sets*. Cycling is stopped if the absolute
    difference between each updated theta and its former value is below the threshold *thresh*, and warm starts
    reuses the previously learned optimal coefficients as starting values for theta in the optimization for the next
    lambda element in the *lamda_path*. After an initial cycle through all *p* variables, the *active_set* feature
    restricts further iterations to the *active set* till convergence; and finally does one more cycle through all *p*
    variables to check if the active set has changed. This helps especially when *p* is large. Uses the base function
    *update_coeffs* for readability.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        penalty_factors (np.ndarray): vector of penalties that function as weights for the coefficients within
            the l1-penalty term of the adaptive lasso; later used in the *soft_threshold* function.
            The shape must be (p, 1). If none are provided, the function defaults to providing a vector of ones,
            which is the standard lasso version
        theta (np.ndarray): initial starting values for the vector of coefficients of shape (p, 1). If none are provided,
            the function defaults to setting each coefficient to zero initially
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for. If none are provided, the
            function provides a data-dependent sequence by default
        num_iters (int): Maximum number of cycles to update the coefficients; usually convergence is reached in
            under 100 iterations. Defaults to 100
        intercept (bool): logical value whether an intercept should be used when fitting (adaptive) lasso
        thresh (float): threshold for determining whether the update was small enough to classify
            the coefficient as converged
        active_thresh (float): threshold for determining whether the coefficient is still different
            enough from zero to be considered active
        warm_start (bool): Logical value determining whether the *warm_starts* feature should be used


    Returns:
        list: list of dicts, each containing:

            **lamda** (*float*): non-negative regularization parameter in the l1-penalty term of the lasso and element of lamda_path \n
            **theta_std** (*np.ndarray*): optimal lasso coefficients on the standardized scale for the given lambda in the dictionary, shape (p, ) \n
            **theta_nat** (*np.ndarray*): optimal lasso coefficients on the original scale for the given lambda in the dictionary, shape (p, )
    """

    n, p = X.shape
    X, y, x_mean, x_std, y_mean, y_std = standardize_input(X=X, y=y)

    if lamda_path is None:
        path = n * get_lamda_path_numba(X_std=X, y_std=y)
    else:
        path = n * lamda_path

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    n, p = X.shape

    if theta is None:
        theta = np.zeros((p, 1))

    if penalty_factors is None:
        penalty_factors = np.ones((p, 1))

    result = []
    for lamda in path:
        if not warm_start:
            theta = np.zeros((p, 1))
        output = {}
        output["lamda"] = lamda / n
        sec_check_all_converged = False
        active_set = np.arange(p)
        active_set_converged = False

        for _i in range(num_iters):
            if (active_set.size != 0) and (not active_set_converged):
                theta, active_set, active_set_converged = update_coeffs(
                    X_std=X,
                    y_std=y,
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
                    X_std=X,
                    y_std=y,
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
        output["theta_nat"] = theta_nat.flatten()
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
    """Just-in-time compiled version of the *active_set_lasso* function.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        penalty_factors (np.ndarray): vector of penalties that function as weights for the coefficients within the
            l1-penalty term of the adaptive lasso; later used in the *soft_threshold* function. The shape must
            be (p, 1). If none are provided, the function defaults to providing a vector of ones,
            which is the standard lasso version
        theta (np.ndarray): initial starting values for the vector of coefficients of shape (p, 1).
            If none are provided, the function defaults to setting each coefficient to zero initially
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for. If none are provided,
            the function provides a data-dependent sequence by default
        num_iters (int): Maximum number of cycles to update the coefficients; usually convergence is reached
            in under 100 iterations. Defaults to 100
        intercept (bool): logical value whether an intercept should be used when fitting (adaptive) lasso
        thresh (float): threshold for determining whether the update was small enough to classify
            the coefficient as converged
        active_thresh (float): threshold for determining whether the coefficient is still different enough
            from zero to be considered active
        warm_start (bool): Logical value determining whether the *warm_starts* feature should be used

    Returns:
        tuple: tuple containing:

            **lamdas** (*list*): list of lamdas in lamda_path for which the lasso problem has been solved \n
            **thetas** (*list*): list of lasso coefficient vectors on the standardized scale, for each lambda in *lamdas* one optimal coefficient vector \n
            **thetas_nat** (*list*): list of lasso coefficient vectors on the original scale, for each lambda in *lamdas* one optimal coefficient vector \n
            **BIC** (*list*): list of BIC values, one for each lambda in *lamdas* (BIC calculated from the respective model trained given a lambda value). For details see **Zou, H., Hastie, T., & Tibshirani, R. (2007)**
    """

    n, p = X.shape

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
        X_tmp = np.ones((n, p + 1))
        X_tmp[:, 1:] = X
        X = X_tmp

    if lamda_path is None:
        path = n * get_lamda_path_numba(X_std=X_standardized, y_std=y_standardized)
    else:
        path = n * lamda_path

    if intercept:
        X_tmp = np.ones((n, p + 1))
        X_tmp[:, 1:] = X_standardized
        X_standardized = X_tmp

    n, p = X_standardized.shape

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

                    for obs in range(n):
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

                n, p = X_standardized.shape

                for subindex, j in enumerate(active_set):
                    w_j = penalty_factors[j].item()

                    y_pred = X_standardized @ theta
                    rho = 0.0
                    z = 0.0

                    for obs in range(n):
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

        n, p = X.shape
        theta_bic = np.ones((p, 1))
        theta_bic[:, 0] = theta_tmp
        residuals_hat = np.sum(np.square(y - X @ theta_bic))
        df_lamda = count_non_zero_coeffs(theta_vec=theta_bic.flatten())
        BIC_lasso = residuals_hat / (n * y_std ** 2) + np.log(n) / n * df_lamda

        lamdas.append(lamda / n)
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

    """Basic implementation of the adaptive lasso from **Zou, H. (2006)**.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        intercept (bool): logical value whether an intercept should be used when fitting lasso or OLS
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for. If none are provided,
            the function provides a data-dependent sequence by default
        gamma_path (np.ndarray): sequence of gamma values to solve the lasso problem for (see paper above for more details on gamma).
            If none are provided, the function provides a simple yet broad enough sequence by default
        first_stage (str): Options are "OLS" and "Lasso" currently. Determines which method should be used for
            getting initial first-stage estimates for the coefficient vector. Defaults to "OLS". If "Lasso"
            is chosen, the full *lamda_path* is calculated and the lamda that minimzes BIC is taken as a final
            estimate (for selection consistency), following **Zou, H., Hastie, T., & Tibshirani, R. (2007)**
        num_iters (int): Maximum number of cycles to update the coefficients; usually convergence is reached in
            under 100 iterations. Defaults to 100
        out_as_df (bool): Logical value determining whether the output should be in pd.DataFrame format instead
            of lists. This is necessary for later use with the *cv_adaptive_lasso* function

    Returns:
        tuple: if out_as_df = False, tuple containing:

            **path_gamma** (*np.ndarray*): sequence of gamma values for which the lasso problem has actually been solved \n
            **results** (*list*): list of tuples returned by calls to "lasso_numba", one tuple for each gamma in *path_gamma* \n
            **weight_path** (*list*): list of np.ndarrays, each consisting of the coefficient weights used for solving lasso (for each gamma one)

        if out_as_df = True, pd.DataFrame:
            **df** (*pd.DataFrame*): dataframe of results, consisting of the *path_gamma* and *path_lamda* vectors as its multiindex; and the standardized coefficient vectors, original scaled coefficient vectors, as well as gamma_weight vectors as its cell entries.
    """

    n, p = X.shape

    if gamma_path is None:
        path_gamma = np.array([0.1, 0.5, 1, 2, 3, 4, 6, 8])
    else:
        path_gamma = gamma_path

    if first_stage == "OLS":
        reg = LinearRegression(fit_intercept=intercept).fit(X, y)
        coeffs = reg.coef_.T
    elif first_stage == "Lasso":
        res = lasso_numba(X=X, y=y)

        # taking the lasso fit that minimizes the BIC estimate
        index_lamda_opt = np.where(res[3] == np.amin(res[3]))[0][0]
        # remove the intercept, since it should not be penalized
        coeffs = np.delete(res[1][index_lamda_opt], 0).reshape((p, 1))

    else:
        raise AssertionError(
            "This feature has so far only been implemented for OLS and Lasso as its first-stage estimators."
        )

    # avoiding numerical issues, division by zero etc.
    coeffs[np.abs(coeffs) < 1.00e-15] = 1.00e-15

    results = []
    weight_path = []

    # this is the second stage, making use of the first-stage estimates from before saved in "coeffs"
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

    """Calculates the adaptive lasso confidence intervals of active coefficients, i.e. coefficients in *theta_std*
    that are distinct from zero. The calculations are based on the Standard Error Formula in chapter 3.6. in Zou, H. (2006).

    Args:
        lamda (float): non-negative regularization parameter in the l1-penalty term of the lasso
        weights (np.ndarray): vector of penalties/ weights for the coefficients within the l1-penalty term
            of the adaptive lasso. The shape must be (p, 1)
        theta_std (np.ndarray): vector of previously fitted adaptive lasso coefficients on standardized scale of shape (p, )
        theta_nat (np.ndarray): vector of previously fitted adaptive lasso coefficients on original scale of shape (p, )
        X (np.ndarray): regressor matrix of shape (n, p)
        X_std (np.ndarray): standardized regressor matrix of shape (n, p)
        intercept (bool): logical value whether an intercept was used while fitting the adaptive lasso
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        y_std (np.ndarray): standardized vector of the dependent variable *y*, of shape (n, 1)

    Returns:
        dict: dict containing:

            **beta_hat_nat_cov_mat** (*np.ndarray*): estimated asymptotic covariance matrix (on original scale) for the active set of estimated coefficients from adaptive lasso \n
            **beta_hat_std_cov_mat** (*np.ndarray*): estimated asymptotic covariance matrix (on standardized scale) for the active set of estimated coefficients from adaptive lasso \n
            **conf_intervals_nat** (*np.ndarray*): elementwise confidence intervals at the 95% confidence level for coefficients in the active set, on original scale \n
            **conf_intervals_std** (*np.ndarray*): elementwise confidence intervals at the 95% confidence level for coefficients in the active set, on standardized scale \n
            **active_set** (*np.ndarray*): the active (non-zero) set of coefficients in *theta_nat* for which confidence intervals were calculated

    """

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

    # selection of the relevant ("active") columns in the regressor matrix X and in the coefficient vectors
    active_set = np.invert(np.isclose(np.zeros(p), theta_nat, atol=1e-06))

    X_active = X[:, active_set]
    X_std_active = X_std[:, active_set]
    theta_nat_active = theta_nat[active_set]
    theta_std_active = theta_std[active_set]
    weights_active = weights[active_set]

    # the steps below follow the Standard Error Formula in chapter 3.6. in Zou, H. (2006)
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

    """Helper function that takes fitted coefficients from a lasso problem and outputs fitted values on some
    sample data matrix X (possibly not the same as the one used for fitting). It also calculates the
    mean-squared-error between fitted and actual responses *y_hat* and *y*.

    Args:
        X (np.ndarray): regressor (data) matrix of shape (n, p), not necessarily the same matrix that was used for fitting *theta_nat*
        y (np.ndarray): corresponding vector of the dependent variable *y*, of shape (n, 1)
        theta_nat (np.ndarray): vector of previously fitted adaptive lasso coefficients on original scale of shape (p, )
        intercept (bool): logical value whether an intercept was used while fitting the adaptive lasso for *theta_nat*

    Returns:
        tuple: tuple containing:

            **y_hat** (*np.ndarray*): fitted responses for the given sample in *X* \n
            **mse** (*float*): mean-squared-error between fitted and actual responses *y_hat* and *y*

    """

    if intercept:
        X = np.insert(X, 0, 1, axis=1)

    y_hat = X @ theta_nat.reshape((X.shape[1], 1))
    mse = np.sum(np.square(y - y_hat))
    return y_hat, mse


def cv_adaptive_lasso(X, y, intercept=True, first_stage="OLS", cross_valid_split=True):

    """Helper function that cross-validates the adaptive lasso in the dimensions (gamma, lambda). Two folds
    are used in the current implementation.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        intercept (bool): logical value whether an intercept shall be used while fitting the adaptive lasso for *theta_nat*
        first_stage (str): Options are "OLS" and "Lasso" currently. Determines which method should be used for getting
            initial first-stage estimates for the coefficient vector. Defaults to "OLS". If "Lasso" is chosen, the full *lamda_path* is calculated and the lamda that minimzes BIC is taken as a final estimate (for selection consistency), following **Zou, H., Hastie, T., & Tibshirani, R. (2007)**
        cross_valid_split (bool): Option to turn-off the exchange of the two cross-validation folds, thus evaluation
            is only done once on the second fold, while training is done on the first fold. In small samples this is
            currently necessary, due to issues with float multiindexing.

    Returns:
        tuple: tuple containing:

            **cv_overview** (*pd.DataFrame*): Summary of all the possible combinations of gammas and lambdas with corresponding model performances in the differnt folds \n
            **params_opt** (*tuple*): Optimal tuple of (gamma_opt, lambda_opt), measured in mean-squared error loss. Element of *cv_overview*.

    """

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
    lamda_path = get_lamda_path(X_std=X_standardized, y_std=y_standardized)

    trained_on_fold_1 = adaptive_lasso(
        X=X_fold_1,
        y=y_fold_1,
        intercept=intercept,
        lamda_path=lamda_path,
        gamma_path=gamma_path,
        first_stage=first_stage,
        num_iters=100,
        out_as_df=True,
    )

    if cross_valid_split:
        trained_on_fold_2 = adaptive_lasso(
            X=X_fold_2,
            y=y_fold_2,
            intercept=intercept,
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

    else:
        trained_on_fold_1["mse_1"] = np.nan
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

            trained_on_fold_1.at[index, "mse_1"] = mse_1

        cv_overview = trained_on_fold_1[["mse_1"]]

        params_opt = cv_overview.iloc[
            cv_overview["mse_1"].argmin(),
        ].name

    return cv_overview, params_opt


def adaptive_lasso_tuned(
    X, y, first_stage="OLS", intercept=True, cross_valid_split=True
):

    """Tuned (i.e. cross-validated) version of the adaptive lasso.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        first_stage (str): Options are "OLS" and "Lasso" currently. Determines which method should be used for getting initial
            first-stage estimates for the coefficient vector. Defaults to "OLS". If "Lasso" is chosen, the full *lamda_path*
            is calculated and the lamda that minimzes BIC is taken as a final estimate (for selection consistency),
            following **Zou, H., Hastie, T., & Tibshirani, R. (2007)**
        intercept (bool): logical value whether an intercept shall be used while fitting the adaptive lasso for *theta_nat*
        cross_valid_split (bool): Option to turn-off the exchange of the two cross-validation folds, thus evaluation
            is only done once on the second fold, while training is done on the first fold. In small samples
            this is currrently necessary, due to issues with float multiindexing.

    Returns:
        dict: dict containing:

            **selected_support** (*np.ndarray*): logical vector of the coefficients that are active in the optimal adaptive lasso fit \n
            **theta_opt_nat** (*np.ndarray*): optimal coefficient vector from fitting and cross-validating the adaptive lasso, on original scale \n
            **theta_opt_std** (*np.ndarray*): optimal coefficient vector from fitting and cross-validating the adaptive lasso, on standardized scale \n
            **conf_intervals_nat** (*np.ndarray*): elementwise confidence intervals at the 95% confidence level for optimal coefficients in the active set, on original scale \n
            **conf_intervals_std** (*np.ndarray*): elementwise confidence intervals at the 95% confidence level for optimal coefficients in the active set, on standardized scale

    """

    n, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_standardized = (X - x_mean) / x_std
    y_standardized = (y - y_mean) / y_std

    cv_results, params_opt = cv_adaptive_lasso(
        X=X,
        y=y,
        intercept=intercept,
        first_stage=first_stage,
        cross_valid_split=cross_valid_split,
    )
    gamma_opt = params_opt[0]
    lamda_opt = params_opt[1]

    train_opt_ada_lasso = adaptive_lasso(
        X=X,
        y=y,
        intercept=intercept,
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
        intercept=intercept,
        y=y,
        y_std=y_standardized,
    )

    selected_support = ada_lasso_opt_res["active_set"]
    conf_intervals_nat = ada_lasso_opt_res["conf_intervals_nat"]
    conf_intervals_std = ada_lasso_opt_res["conf_intervals_std"]

    return {
        "selected_support": selected_support,
        "theta_opt_nat": train_opt_ada_lasso.iloc[0]["theta_nat"],
        "theta_opt_std": train_opt_ada_lasso.iloc[0]["theta_std"],
        "conf_intervals_nat": conf_intervals_nat,
        "conf_intervals_std": conf_intervals_std,
    }


def interpretable_confidence_intervals(adaptive_lasso_tuned_obj, intercept):

    """Rearranges the output from the *adaptive_lasso_tuned* method for later analysis.

    Args:
        adaptive_lasso_tuned_obj (dict): return object from a call to *adaptive_lasso_tuned*
        intercept (bool): logical value whether an intercept was used while fitting the adaptive lasso

    Returns:
        tuple: tuple containing:

            (*pd.DataFrame*): contains a column of the selected_support, a column of estimated coefficients, but without the intercept, and two columns of lower- and upper confidence bounds \n
            (*float*): value of the estimated intercept, if *intercept* = True
    """

    if intercept:
        theta_opt_nat = np.delete(arr=adaptive_lasso_tuned_obj["theta_opt_nat"], obj=0)
    else:
        theta_opt_nat = adaptive_lasso_tuned_obj["theta_opt_nat"]

    lower_conf_bound = np.full(
        [len(adaptive_lasso_tuned_obj["selected_support"])], np.nan
    )
    upper_conf_bound = np.full(
        [len(adaptive_lasso_tuned_obj["selected_support"])], np.nan
    )

    np.put(
        a=lower_conf_bound,
        ind=np.nonzero(adaptive_lasso_tuned_obj["selected_support"])[0],
        v=adaptive_lasso_tuned_obj["conf_intervals_nat"][:, 0],
    )
    np.put(
        a=upper_conf_bound,
        ind=np.nonzero(adaptive_lasso_tuned_obj["selected_support"])[0],
        v=adaptive_lasso_tuned_obj["conf_intervals_nat"][:, 1],
    )

    d = {
        "selected_support": adaptive_lasso_tuned_obj["selected_support"],
        "theta_nat": theta_opt_nat,
        "lower_conf_bound": lower_conf_bound,
        "upper_conf_bound": upper_conf_bound,
    }

    if intercept:
        return pd.DataFrame(data=d), adaptive_lasso_tuned_obj["theta_opt_nat"][0]
    else:
        return pd.DataFrame(data=d), np.nan
