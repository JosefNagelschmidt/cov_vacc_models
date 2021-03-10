import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from sklearn.preprocessing import StandardScaler

# from statsmodels.stats.moment_helpers import corr2cov


def toeplitz_cov(p, rho):
    first_col_toeplitz = np.repeat(rho, p)
    exponents = np.arange(p)
    first_col_toeplitz = first_col_toeplitz ** exponents

    return toeplitz(c=first_col_toeplitz)


# both of these choices of covariance matrices satisfy the strong irrepresentability condition
def get_cov_mat(p, identity):
    if identity:
        return np.identity(p)
    else:
        return toeplitz_cov(p=p, rho=0.8)


def get_X_mat(n, p, identity_cov, seed=1):
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
    assert p > 10
    return np.insert(
        arr=np.zeros((p - 10,)),
        obj=0,
        values=np.array([1.0, 0.5, 1.0, -1, -0.1, -0.5, 0.01, -0.05, -1.5, 1.5]),
    ).reshape((p, 1))


def linear_link(X, beta):
    return X @ beta


def polynomial_link(X, beta):
    n, p = X.shape

    linear_effect = X[:, 0].reshape((n, 1)) @ beta[0, :].reshape((1, 1))
    residual = X[:, 1:] @ beta[1:, :].reshape((p - 1, 1))

    return linear_effect + (residual) ** 2 + (residual) + 1


def sine_link(X, beta):
    n, p = X.shape

    linear_effect = X[:, 0].reshape((n, 1)) @ beta[0, :].reshape((1, 1))
    residual = X[:, 1:] @ beta[1:, :].reshape((p - 1, 1))

    return linear_effect + np.sin(2 * residual) + np.sin(residual) + 1


def get_artificial_dgp(n, p, link_function, identity_cov=True):

    X = get_X_mat(n=n, p=p, identity_cov=True, seed=1)
    beta_vec = get_true_beta_vec(p=p)

    y = link_function(X, beta_vec).flatten() + np.random.normal(
        loc=0.0, scale=0.5, size=n
    )
    y = y.reshape((len(y), 1))
    return {"X": X, "y": y, "beta": beta_vec}


def get_real_data_dgp(rel_path, january=True):
    data = pd.read_csv(rel_path)
    data = data.drop(["personal_id"], axis=1)

    if january:
        y = data[["vaccine_intention_jan"]]
        X = data.drop(["vaccine_intention_jan", "vaccine_intention_jul"], axis=1)
    else:
        y = data[["vaccine_intention_jul"]]
        X = data.drop(["vaccine_intention_jan", "vaccine_intention_jul"], axis=1)

    y_numpy = y.to_numpy()
    X_numpy = X.to_numpy()

    scaler = StandardScaler()
    # y_std = scaler.fit_transform(y_numpy)
    X_std = scaler.fit_transform(X_numpy)

    n, p = X_std.shape

    # create random beta_vec
    active_set = np.concatenate(
        (np.repeat(0.8, 2), np.repeat(0.35, 2), np.repeat(0.15, 2), np.repeat(0.06, 9))
    )
    inactive_set = np.zeros(p - len(active_set))

    mixed_set = np.concatenate((active_set, inactive_set))
    np.random.shuffle(mixed_set)
    beta_vec_std = mixed_set.reshape((p, 1))

    return {
        "X": X_numpy,
        "y": y_numpy,
        "beta_std": beta_vec_std,
        "active_set": active_set,
    }
