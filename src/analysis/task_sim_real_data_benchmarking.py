from itertools import product

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.model_code.dgp import get_real_data_dgp
from src.model_code.estimators import adaptive_lasso_tuned
from src.model_code.external_estimators import boruta_selector
from src.model_code.external_estimators import lasso_feature_selection
from src.model_code.external_estimators import OLS_confidence_intervals
from src.model_code.external_estimators import univariate_feature_selection
from src.model_code.helpers import selection_power
from src.model_code.helpers import true_params_in_conf_interval
# from pathlib import Path
# from src.config import SRC

add_profession = ["yes", "no"]
add_political = ["yes", "no"]


@pytask.mark.skip
@pytask.mark.depends_on({"data": BLD / "data" / "sparse_modelling_df_add_profession_yes_add_political_yes.csv"})
@pytask.mark.produces(BLD / "analysis" / "simulation_study_real_dgp_add_profession_yes_add_political_yes.csv")
def task_simulation_real_dgps(depends_on, produces):

    external_selectors = {'boruta_selector': boruta_selector,
                          "univariate_feature_selection": univariate_feature_selection,
                          "lasso_feature_selection": lasso_feature_selection,
                          "adaptive_lasso_tuned": adaptive_lasso_tuned}

    number_simulations = 250
    simulation_id = np.arange(number_simulations)
    selectors = list(external_selectors.keys())

    index = product(simulation_id, selectors)

    index = pd.MultiIndex.from_tuples(
        index,
        names=("simulation_id", "selector"),
    )

    df = pd.DataFrame(columns=["share_of_truth_uncovered",
                               "ratio_total_select_coeffs_true_coeffs",
                               "false_pos_share_true_support",
                               "false_pos_share_right_selection",
                               "linear_effect_coverage"],
                      index=index)

    for sim in df.index.get_level_values("simulation_id").unique():
        data = get_real_data_dgp(rel_path=depends_on["data"], january=True, sd=1.0)
        X = data["X"]

        n, p = X.shape

        y = data["y_artificial"].reshape((len(data["y_artificial"]), 1))
        beta = data["beta"]
        true_support = data["support"]

        indices = np.random.permutation(n)
        fold_1_idx, fold_2_idx = indices[: int(n / 2)], indices[int(n / 2):]
        X_fold_1, X_fold_2 = X[fold_1_idx, :], X[fold_2_idx, :]
        y_fold_1, y_fold_2 = y[fold_1_idx, :], y[fold_2_idx, :]

        for select in df.index.get_level_values("selector").unique():
            if (select != "adaptive_lasso_tuned") and (select != "lasso_feature_selection"):
                selected_support = external_selectors[select](X_fold=X_fold_1, y_fold=y_fold_1)
                conf_int = OLS_confidence_intervals(X_validation=X_fold_2, y_validation=y_fold_2, support=selected_support, intercept=False)

            elif select == "lasso_feature_selection":
                selected_support = external_selectors[select](X_fold=X_fold_1, y_fold=y_fold_1, intercept=False)
                conf_int = OLS_confidence_intervals(X_validation=X_fold_2, y_validation=y_fold_2, support=selected_support, intercept=False)

            else:
                res_dict = external_selectors[select](X=X, y=y, intercept=False, cross_valid_split=False)
                selected_support = res_dict["selected_support"]
                conf_int = res_dict["conf_intervals_nat"]

            selection_stats = selection_power(true_support=true_support, selected_support=selected_support)
            index_df = (sim, select)

            if conf_int.shape[0] == np.sum(selected_support):
                coverage = np.sum(true_params_in_conf_interval(true_theta_vec=beta[selected_support, :], conf_int_matrix=conf_int)) / np.sum(selected_support)
                df.at[index_df, "linear_effect_coverage"] = coverage

            df.at[index_df, "share_of_truth_uncovered"] = selection_stats["share_of_truth_uncovered"]
            df.at[index_df, "ratio_total_select_coeffs_true_coeffs"] = selection_stats["ratio_total_select_coeffs_true_coeffs"]
            df.at[index_df, "false_pos_share_true_support"] = selection_stats["false_pos_share_true_support"]
            df.at[index_df, "false_pos_share_right_selection"] = selection_stats["false_pos_share_right_selection"]

    df.to_csv(produces, index_label=["simulation_id", "selector"])
