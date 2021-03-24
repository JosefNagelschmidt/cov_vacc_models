import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import toeplitz


def toeplitz_cov(p, rho):
    """Generates a toeplitz-type covariance matrix, given dimensionality *p*, and *rho*.

    Args:
        p (int): dimensionality of the covariance matrix; number of regressors
        rho (float): the covariance matrix has power-decay entries :math:`\\Sigma_{ij} = \\rho^{|i-j|}, 0 < \\rho < 1`

    Returns:
        (*np.ndarray*): toeplitz-type covariance matrix

    """
    first_col_toeplitz = np.repeat(rho, p)
    exponents = np.arange(p)
    first_col_toeplitz = first_col_toeplitz ** exponents

    return toeplitz(c=first_col_toeplitz)


# both of these choices of covariance matrices satisfy the strong irrepresentability condition
def get_cov_mat(p, identity):
    """Generates either an identity, or a toeplitz-type covariance matrix with dimensionality *p*.

    Args:
        p (int): dimensionality of the covariance matrix; number of regressors
        identity (bool): whether the identity matrix shall be used as a covariance matrix. If False,
            a toeplitz-type matrix is provided with power decay rho = 0.8

    Returns:
        (*np.ndarray*): identity or toeplitz-type covariance matrix

    """
    if identity:
        return np.identity(p)
    else:
        return toeplitz_cov(p=p, rho=0.8)


def get_X_mat(n, p, identity_cov):

    """Generates a random regressor matrix of dimensionality *p* and sample size *n*. The regressors
    follow a multivariate normal distributions with either an identity, or a toeplitz-type covariance matrix.

    Args:
        n (int): sample size, or equivalently, the number of rows of the generated matrix
        p (int): number of regressors, or equivalently, the number of columns of the generated matrix
        identity_cov (bool): whether the identity matrix shall be used as a covariance matrix.
            If False, a toeplitz-type matrix is provided with power decay rho = 0.8

    Returns:
        (*np.ndarray*): regressor matrix of dimensionality (n, p), following a multivariate normal distribution

    """

    assert p > 10

    # np.random.seed(seed=seed)

    if identity_cov:
        return np.random.multivariate_normal(
            mean=np.repeat(0, repeats=p),
            cov=get_cov_mat(p=p, identity=identity_cov),
            size=n,
        )
    else:
        return np.random.multivariate_normal(
            mean=np.repeat(0, repeats=p),
            cov=get_cov_mat(p=p, identity=identity_cov),
            size=n,
        )


def get_true_beta_vec(p):

    """Generates a true coefficient vector for the artifical DGPs considered in the simulation study.
    The number of relevant (active) coefficients is fixed to 10, the rest of the vector is filled with zeros.

    Args:
        p (int): number of regressors (or coefficients) in the corresponding dataset

    Returns:
        (np.ndarray): sparse true coefficient vector

    """

    assert p > 10
    return np.insert(
        arr=np.zeros((p - 10,)),
        obj=0,
        values=np.array([1.0, 0.5, 1.0, -1, -0.1, -0.5, 0.01, -0.05, -1.5, 1.5]),
    ).reshape((p, 1))


def linear_link(X, beta):
    """Generates a linear main effect for the artificial DGP. The main effect is in this case
    just a linear combination of regressors.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        beta (np.ndarray): vector of coefficients of shape (p, 1)

    Returns:
        (*np.ndarray*): vector of the main effects for the artificial DGP, which later become dependent variables after adding randomness; shape (n, )

    """
    return X @ beta


def polynomial_link(X, beta):
    """Generates a polynomial main effect for the artificial DGP, with a partially linear
    structure (in the first factor). For simplicity, the polynomial link only goes up to the second order.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        beta (np.ndarray): vector of coefficients of shape (p, 1)

    Returns:
        (*np.ndarray*): vector of the main effects for the artificial DGP, which later become dependent variables after adding randomness; shape (n, )

    """
    n, p = X.shape

    # part of the main effect for which inference is later intended
    linear_effect = X[:, 0].reshape((n, 1)) @ beta[0, :].reshape((1, 1))

    # non-linear (confounding) part of the main efffect (put into the polynomial link function)
    residual = X[:, 1:] @ beta[1:, :].reshape((p - 1, 1))

    return linear_effect + (residual) ** 2 + (residual) + 1


def sine_link(X, beta):
    """Generates a sine main effect for the artificial DGP, with a partially linear structure (in the first factor).

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        beta (np.ndarray): vector of coefficients of shape (p, 1)

    Returns:
        (*np.ndarray*): vector of the main effects for the artificial DGP, which later become dependent variables after adding randomness; shape (n, )

    """

    n, p = X.shape

    # part of the main effect for which inference is later intended
    linear_effect = X[:, 0].reshape((n, 1)) @ beta[0, :].reshape((1, 1))

    # non-linear (confounding) part of the main efffect (put into the sine link function)
    residual = X[:, 1:] @ beta[1:, :].reshape((p - 1, 1))

    return linear_effect + np.sin(2 * residual) + np.sin(residual) + 1


def get_artificial_dgp(n, p, link_function, identity_cov=True):

    """Generates an artificial DGP :math:`y = m(X) + \\epsilon` with a specified link function,
    partly linear main effect, and specified covariance structure for normally distributed regressors *X*.

    Args:
        n (int): sample size, or equivalently, the number of rows of the generated regressor matrix
        p (int): number of regressors, or equivalently, the number of columns of the generated regressor matrix
        link_function (str): the link_funtion for the main effect; currently available are
            "sine_link", "polynomial_link", and "linear_link".
        identity_cov (bool): whether the identity matrix shall be used as a covariance matrix.
            If False, a toeplitz-type matrix is provided with power decay rho = 0.8

    Returns:
        dict: dict containing:

            **X** (*np.ndarray*): generated regressor matrix of shape (n, p) \n
            **y** (*np.ndarray*): generated vector of dependent variables of shape (n, 1) \n
            **beta** (*np.ndarray*): vector of true coefficients of shape (p, 1)

    """

    X = get_X_mat(n=n, p=p, identity_cov=True)
    beta_vec = get_true_beta_vec(p=p)

    y = link_function(X, beta_vec).flatten() + np.random.normal(
        loc=0.0, scale=0.5, size=n
    )
    y = y.reshape((len(y), 1))

    return {"X": X, "y": y, "beta": beta_vec}


def get_real_data_dgp(rel_path, january=True, sd=1.0):

    """Generates a DGP :math:`y = g(X) + \\epsilon` based on real-world (LISS) data. This more realistic
    data generating process is constructed in the following way:

    1. Get a random sample of the data, e.g. of size :math:`\\lfloor \\frac{n}{2} \\rfloor` (*n* is sample size).
    2. Make a linear projection of *y* on the regressor matrix *X*, linearizing the model.
    3. Set medium-to-small regression parameters to zero, inducing some noise and adding sparsity characteristic.
    4. Get estimates from the projected model, using the updated parameters from (3) and adding some additional noise :math:`\\eta_i ∼ \\mathcal{N}(0,\\sigma^2)`.

    Following this procedure, one gets a linear characterization of the true conditional expectation function
    in order to assess coverage of estimated confidence intervals in a controlled environment, where the data
    generating process is known, but one does not risk to overfit to the data too much later on. This is
    particularly useful to gain information upon the adequacy of various methods in our real data sample,
    which includes many binary covariates, as well as a regressor matrix with a non-trivial covariance structure.

    Args:
        rel_path (str): path to the .csv file of pre-processed data
        january (bool): whether data from january shall be used. If False, data from july is used instead.
        sd (float): standard deviation for the error term that is added

    Returns:
        dict: dict containing:

            **X** (*np.ndarray*): true regressor matrix of shape (n, p) \n
            **y_true** (*np.ndarray*): vector of true dependent variablev values of shape (n, ) \n
            **y_artificial** (*np.ndarray*): generated vector of dependent variables of shape (n, ) \n
            **beta** (*np.ndarray*): vector of generated (but assumed true) coefficients of shape (p, 1) \n
            **support** (*np.ndarray*): logical vector of the coefficients that are active (non-zero) \n

    """

    data = pd.read_csv(rel_path)
    data = data.drop(["personal_id"], axis=1)

    if january:
        y = data[["vaccine_intention_jan"]]
        X = data.drop(["vaccine_intention_jan", "vaccine_intention_jul"], axis=1)
    else:
        y = data[["vaccine_intention_jul"]]
        X = data.drop(["vaccine_intention_jan", "vaccine_intention_jul"], axis=1)

    y_true_numpy = y.to_numpy()
    X_numpy = X.to_numpy()

    n, p = X_numpy.shape

    indices = np.random.permutation(n)
    fold_1_idx = indices[: int(n / 2)]
    X_fold_1 = X_numpy[fold_1_idx, :]
    y_fold_1 = y_true_numpy[fold_1_idx, :]

    mod = sm.OLS(endog=y_fold_1, exog=X_fold_1)
    lin_reg = mod.fit()
    beta = lin_reg.params.reshape((len(lin_reg.params), 1))

    beta[np.abs(beta) < 1.00e-1] = 0

    y_artificial = (X_numpy @ beta).flatten() + np.random.normal(
        loc=0.0, scale=sd, size=n
    )
    support = np.invert(np.isclose(np.zeros(p), beta.flatten(), atol=1e-06))

    return {
        "X": X_numpy,
        "y_true": y_true_numpy.flatten(),
        "y_artificial": y_artificial,
        "beta": beta,
        "support": support,
    }
