import numpy as np
import statsmodels.api as sm
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def boruta_selector(X_fold, y_fold):
    rf = RandomForestRegressor(n_jobs=-1, n_estimators=500, max_depth=6, random_state=0)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=1)
    feat_selector.fit(X_fold, y_fold.flatten())
    return feat_selector.support_


def univariate_feature_selection(X_fold, y_fold):
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
    n, p = X_fold.shape
    reg = LassoCV(cv=folds, random_state=0, fit_intercept=intercept).fit(
        X_fold, y_fold.flatten()
    )
    coeffs = reg.coef_
    support = np.invert(np.isclose(np.zeros(p), coeffs, atol=1e-06))
    return support


def OLS_confidence_intervals(X_validation, y_validation, support, intercept=True):

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
