from itertools import product

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.model_code.dgp import get_artificial_dgp
from src.model_code.dgp import linear_link
from src.model_code.dgp import polynomial_link
from src.model_code.dgp import sine_link
from src.model_code.estimators import adaptive_lasso_tuned
from src.model_code.external_estimators import boruta_selector
from src.model_code.external_estimators import lasso_feature_selection
from src.model_code.external_estimators import OLS_confidence_intervals
from src.model_code.external_estimators import univariate_feature_selection
from src.model_code.helpers import selection_power
from src.model_code.helpers import true_params_in_conf_interval

# from pathlib import Path
# from src.config import SRC


@pytask.mark.skip
@pytask.mark.produces(BLD / "analysis" / "simulation_study_artificial_results.csv")
def task_simulation_artificial_dgps(produces):
    dgp_functions = {
        "linear_link": linear_link,
        "polynomial_link": polynomial_link,
        "sine_link": sine_link,
    }

    external_selectors = {
        "boruta_selector": boruta_selector,
        "univariate_feature_selection": univariate_feature_selection,
        "lasso_feature_selection": lasso_feature_selection,
        "adaptive_lasso_tuned": adaptive_lasso_tuned,
    }

    n = [100, 2000]
    number_simulations = 40
    simulation_id = np.arange(number_simulations)
    identity = [True, False]
    p = [20, 80]
    link_functions = list(dgp_functions.keys())
    selectors = list(external_selectors.keys())
    index = product(simulation_id, n, identity, p, link_functions, selectors)

    index = pd.MultiIndex.from_tuples(
        index,
        names=(
            "simulation_id",
            "n_obs",
            "identity_cov_matrix",
            "p_features",
            "link_function",
            "selector",
        ),
    )

    df = pd.DataFrame(
        columns=[
            "share_of_truth_uncovered",
            "ratio_total_select_coeffs_true_coeffs",
            "false_pos_share_true_support",
            "false_pos_share_right_selection",
            "linear_effect_coverage",
            "conf_int_width",
        ],
        index=index,
    )

    for sim in df.index.get_level_values("simulation_id").unique():
        for n in df.index.get_level_values("n_obs").unique():
            for ident in df.index.get_level_values("identity_cov_matrix").unique():
                for p in df.index.get_level_values("p_features").unique():
                    for link in df.index.get_level_values("link_function").unique():
                        dgp = get_artificial_dgp(
                            n=n,
                            p=p,
                            link_function=dgp_functions[link],
                            identity_cov=ident,
                        )
                        X = dgp["X"]
                        y = dgp["y"]
                        beta = dgp["beta"]
                        true_support = np.invert(
                            np.isclose(np.zeros(p), beta.flatten(), atol=1e-06)
                        )
                        indices = np.random.permutation(n)
                        fold_1_idx, fold_2_idx = (
                            indices[: int(n / 2)],
                            indices[int(n / 2) :],
                        )
                        X_fold_1, X_fold_2 = X[fold_1_idx, :], X[fold_2_idx, :]
                        y_fold_1, y_fold_2 = y[fold_1_idx, :], y[fold_2_idx, :]

                        for select in df.index.get_level_values("selector").unique():
                            if select != "adaptive_lasso_tuned":
                                selected_support = external_selectors[select](
                                    X_fold=X_fold_1, y_fold=y_fold_1
                                )
                                conf_int = OLS_confidence_intervals(
                                    X_validation=X_fold_2,
                                    y_validation=y_fold_2,
                                    support=selected_support,
                                    intercept=True,
                                )

                            else:
                                res_dict = external_selectors[select](X=X, y=y)
                                selected_support = res_dict["selected_support"]
                                conf_int = res_dict["conf_intervals_nat"]

                            selection_stats = selection_power(
                                true_support=true_support,
                                selected_support=selected_support,
                            )
                            index_df = (sim, n, ident, p, link, select)

                            if selected_support[0]:
                                coverage = true_params_in_conf_interval(
                                    true_theta_vec=beta[selected_support, :],
                                    conf_int_matrix=conf_int,
                                )
                                df.at[index_df, "linear_effect_coverage"] = coverage[0]
                                df.at[index_df, "conf_int_width"] = conf_int[
                                    0, :
                                ].flatten()
                            else:
                                df.at[index_df, "linear_effect_coverage"] = np.nan
                                df.at[index_df, "conf_int_width"] = np.nan

                            df.at[
                                index_df, "share_of_truth_uncovered"
                            ] = selection_stats["share_of_truth_uncovered"]
                            df.at[
                                index_df, "ratio_total_select_coeffs_true_coeffs"
                            ] = selection_stats["ratio_total_select_coeffs_true_coeffs"]
                            df.at[
                                index_df, "false_pos_share_true_support"
                            ] = selection_stats["false_pos_share_true_support"]
                            df.at[
                                index_df, "false_pos_share_right_selection"
                            ] = selection_stats["false_pos_share_right_selection"]

    df.to_csv(
        produces,
        index_label=[
            "simulation_id",
            "n_obs",
            "identity_cov_matrix",
            "p_features",
            "link_function",
            "selector",
        ],
    )
