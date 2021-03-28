import numpy as np
import statsmodels.api as sm
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .estimators import get_lamda_path_numba


def boruta_selector(X_fold, y_fold):
    """Selector based on the boruta algorithm, paired with random forests as base learners.
    See https://github.com/scikit-learn-contrib/boruta_py for more details. No cross-validation
    of max_depth of the random forests is done, since this was not feasible in time, however, random
    forests are supposed to work reasonably well out-of-the box. This is the first-stage
    of a naive post-model selection inference procedure, where OLS confidence bands are later estimated
    on the remaining active set of regressors.

    Args:
        X_fold (np.ndarray): subsample of the regressor matrix of shape (n, p) used for the model selection step
        y_fold (np.ndarray): corresponding subsample of values of the dependent variable *y*, of shape (n, 1) or (n, )

    Returns:
        (*np.ndarray*): logical vector of shape (p, ), indicating which regressors are relevant (True)

    """
    rf = RandomForestRegressor(n_jobs=-1, n_estimators=500, max_depth=6, random_state=0)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=1)
    feat_selector.fit(X_fold, y_fold.flatten())
    return feat_selector.support_


def univariate_feature_selection(X_fold, y_fold):
    """Selector based on simple univariate test statistics,
    see https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    for more details. Grid search cross-validation over the optimal number of relevant regressors (here: *k*) is
    conducted via the scikit-learn GridSearchCV method. This is the first-stage of a naive post-model selection
    inference procedure, where OLS confidence bands are later estimated on the remaining active set of regressors.

    Args:
        X_fold (np.ndarray): subsample of the regressor matrix of shape (m, p) used for the model selection step
        y_fold (np.ndarray): corresponding subsample of values of the dependent variable *y*, of shape (n, 1) or (n, )

    Returns:
        (*np.ndarray*): logical vector of shape (p, ), indicating which regressors are relevant (True)

    """
    n, p = X_fold.shape
    if 1.5 * n <= p:
        upper = int(p / 3)
    else:
        upper = p

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_regression, k=5)),
            ("final_regression", LinearRegression()),
        ]
    )
    search_space = [{"selector__k": np.arange(start=1, stop=upper)}]
    grid_res = GridSearchCV(pipe, search_space, cv=2, verbose=0)
    grid_res = grid_res.fit(X_fold, y_fold.flatten())
    k_opt = grid_res.best_params_["selector__k"]

    support = (
        SelectKBest(f_regression, k=k_opt).fit(X_fold, y_fold.flatten()).get_support()
    )
    return support


def lasso_feature_selection(X_fold, y_fold, folds=2, intercept=True):
    """Simple cross-validated (two folds) lasso, setting coefficients of non-relevant regressors to zero.
    This is the first-stage of a naive post-model selection inference procedure, where OLS confidence bands
    are later estimated on the remaining active set of regressors.

    Args:
        X_fold (np.ndarray): subsample of the regressor matrix of shape (n, p) used for the model selection step
        y_fold (np.ndarray): corresponding subsample of values of the dependent variable *y*, of shape (n, 1) or (n, )

    Returns:
        (*np.ndarray*): logical vector of shape (p, ), indicating which regressors are relevant (True)

    """
    n, p = X_fold.shape
    reg = LassoCV(cv=folds, random_state=0, fit_intercept=intercept).fit(
        X_fold, y_fold.flatten()
    )
    coeffs = reg.coef_
    support = np.invert(np.isclose(np.zeros(p), coeffs, atol=1e-06))
    return support


def OLS_confidence_intervals(X_validation, y_validation, support, intercept=True):

    """This method produces an OLS fit of *y_validation* on a subset of regressors in the data matrix *X_validation*
    (only relevant variables are considered), and then generates confidence intervals for each of the coefficients in
    the active set. This is the second-stage of a naive post-model selection inference procedure, where a pre-selected
    set of relevant regressors is passed to the OLS stage.

    Args:
        X_validation (np.ndarray): subsample of the regressor matrix of shape (n, p) used for the model selection step
        y_validation (np.ndarray): corresponding subsample of values of the dependent variable *y*, of shape (n, 1) or (n, )
        support (np.ndarray): logical vector of shape (p, ), indicating which regressors in the columns
            of *X_validation* are relevant (True). Passed from the first-stage selector
        intercept (bool): logical value whether an intercept shall be used when fitting OLS

    Returns:
        (*np.ndarray*): confidence intervals for relevant regressors as indicated in *support*. Lower bounds are in the first column, upper bounds in the second column.

    """

    X_supp = X_validation[:, support]

    if intercept:
        X_supp = sm.add_constant(X_supp)

    mod = sm.OLS(endog=y_validation.flatten(), exog=X_supp)
    res = mod.fit()
    if intercept:
        conf_int = np.delete(res.conf_int(), 0, 0)
    else:
        conf_int = res.conf_int()
    return conf_int


def sk_learn_lasso(X, y, intercept=True, lamda_path=None):

    """Wrapper for the coordinate descent implementation of the lasso optimization problem by the scikit-learn library.

    Args:
        X (np.ndarray): regressor matrix of shape (n, p)
        y (np.ndarray): vector of the dependent variable *y*, of shape (n, 1)
        intercept (bool): logical value whether an intercept should be used when fitting lasso
        lamda_path (np.ndarray): sequence of lambda values to solve the lasso problem for.
            If none are provided, the function provides a data-dependent sequence by default.

    Returns:
        tuple: tuple containing:

            **lamdas** (*list*): sequence of lambda values as specified in *lamda_path*, otherwise generated data-dependent sequence of lambda values for which lasso was solved
            **coeffs** (*list*): list of optimal lasso coefficient vectors on the standardized scale, one for each lambda in *lamdas*

    """

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_std = (X - x_mean) / x_std
    y_std = (y - y_mean) / y_std

    if lamda_path is None:
        path = get_lamda_path_numba(X_std=X_std, y_std=y_std)
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
