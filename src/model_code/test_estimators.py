import numpy as np
import pytest
from sklearn.linear_model import Lasso

from .estimators import active_set_lasso
from .estimators import eps_thresh_lasso
from .estimators import eps_thresh_lasso_warm_start
from .estimators import get_lamda_path_numba
from .estimators import lasso_numba
from .estimators import naive_lasso


def sk_learn_lasso(X, y, intercept=True, lamda_path=None):

    m, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_std = (X - x_mean) / x_std
    y_std = (y - y_mean) / y_std

    if lamda_path is None:
        path = get_lamda_path_numba(X=X_std, y=y_std)
    else:
        path = lamda_path

    y_std = y_std.flatten()

    lamdas = []
    coeffs = []

    for lamda in path:
        reg = Lasso(alpha=lamda, fit_intercept=intercept)
        reg.fit(X_std, y_std)

        if intercept:
            coef = np.insert(arr=reg.coef_, obj=0, values=reg.intercept_)
        else:
            coef = reg.coef_

        lamdas.append(lamda)
        coeffs.append(np.copy(coef))

    return lamdas, coeffs


@pytest.fixture
def setup_p_less_n_data():

    np.random.seed(seed=1)
    X = np.random.rand(100, 50)
    y = np.array(
        1.5 * X[:, 0] - 14.5 * X[:, 1] + 0.01 * X[:, 45] + 5, dtype=np.float64
    ).reshape(-1, 1)

    return {"X": X, "y": y}


@pytest.fixture
def setup_p_larger_n_data():

    np.random.seed(seed=1)
    X = np.random.rand(50, 100)
    y = np.array(
        1.5 * X[:, 0] - 14.5 * X[:, 1] + 0.01 * X[:, 45] + 5, dtype=np.float64
    ).reshape(-1, 1)

    return {"X": X, "y": y}


def test_naive_lasso_p_less_n(setup_p_less_n_data):
    X = setup_p_less_n_data["X"]
    y = setup_p_less_n_data["y"]

    result_intercept = naive_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = naive_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_naive_lasso_p_larger_n(setup_p_larger_n_data):
    X = setup_p_larger_n_data["X"]
    y = setup_p_larger_n_data["y"]

    result_intercept = naive_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = naive_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_eps_thresh_lasso_p_less_n(setup_p_less_n_data):
    X = setup_p_less_n_data["X"]
    y = setup_p_less_n_data["y"]

    result_intercept = eps_thresh_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = eps_thresh_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_eps_thresh_lasso_p_larger_n(setup_p_larger_n_data):
    X = setup_p_larger_n_data["X"]
    y = setup_p_larger_n_data["y"]

    result_intercept = eps_thresh_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = eps_thresh_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_eps_thresh_lasso_warm_start_p_less_n(setup_p_less_n_data):
    X = setup_p_less_n_data["X"]
    y = setup_p_less_n_data["y"]

    result_intercept = eps_thresh_lasso_warm_start(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = eps_thresh_lasso_warm_start(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_eps_thresh_lasso_warm_start_p_larger_n(setup_p_larger_n_data):
    X = setup_p_larger_n_data["X"]
    y = setup_p_larger_n_data["y"]

    result_intercept = eps_thresh_lasso_warm_start(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = eps_thresh_lasso_warm_start(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_active_set_lasso_p_less_n(setup_p_less_n_data):
    X = setup_p_less_n_data["X"]
    y = setup_p_less_n_data["y"]

    result_intercept = active_set_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = active_set_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_active_set_lasso_p_larger_n(setup_p_larger_n_data):
    X = setup_p_larger_n_data["X"]
    y = setup_p_larger_n_data["y"]

    result_intercept = active_set_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = active_set_lasso(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[0]["theta_std"], expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[0]["theta_std"],
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_lasso_numba_p_less_n(setup_p_less_n_data):
    X = setup_p_less_n_data["X"]
    y = setup_p_less_n_data["y"]

    result_intercept = lasso_numba(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = lasso_numba(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[1][0].flatten(), expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[1][0].flatten(),
        expected_result_no_intercept[1][0],
        decimal=6,
    )


def test_lasso_numba_lasso_p_larger_n(setup_p_larger_n_data):
    X = setup_p_larger_n_data["X"]
    y = setup_p_larger_n_data["y"]

    result_intercept = lasso_numba(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=True
    )

    expected_result_intercept = sk_learn_lasso(
        X=X, y=y, intercept=True, lamda_path=np.array([0.01])
    )

    result_no_intercept = lasso_numba(
        X=X, y=y, lamda_path=np.array([0.01]), intercept=False
    )

    expected_result_no_intercept = sk_learn_lasso(
        X=X, y=y, intercept=False, lamda_path=np.array([0.01])
    )

    np.testing.assert_array_almost_equal(
        result_intercept[1][0].flatten(), expected_result_intercept[1][0], decimal=6
    )
    np.testing.assert_array_almost_equal(
        result_no_intercept[1][0].flatten(),
        expected_result_no_intercept[1][0],
        decimal=6,
    )
