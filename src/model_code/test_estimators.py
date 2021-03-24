import numpy as np
import pytest

from .estimators import active_set_lasso
from .estimators import eps_thresh_lasso
from .estimators import eps_thresh_lasso_warm_start
from .estimators import lasso_numba
from .estimators import naive_lasso
from .external_estimators import sk_learn_lasso


@pytest.fixture
def setup_p_less_n_data():

    """Creating a DGP where the regressor matrix X has less regressors (columns) than observations (rows).

    Args:

    Returns:
        dict: dict containing:
            **X** (*np.ndarray*): generated regressor matrix *X* (sample data) for testing \n
            **y** (*np.ndarray*): generated vector of dependent variable values *y*

    """

    np.random.seed(seed=1)
    X = np.random.rand(100, 50)
    y = np.array(
        1.5 * X[:, 0] - 14.5 * X[:, 1] + 0.01 * X[:, 45] + 5, dtype=np.float64
    ).reshape(-1, 1)

    return {"X": X, "y": y}


@pytest.fixture
def setup_p_larger_n_data():

    """Creating a DGP where the regressor matrix X has more regressors (columns) than observations (rows).

    Args:

    Returns:
        dict: dict containing:
            **X** (*np.ndarray*): generated regressor matrix *X* (sample data) for testing \n
            **y** (*np.ndarray*): generated vector of dependent variable values *y*

    """

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
