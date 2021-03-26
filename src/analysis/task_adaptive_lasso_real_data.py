import itertools

import pandas as pd
import pytask

from src.config import BLD
from src.model_code.estimators import adaptive_lasso_tuned
from src.model_code.estimators import interpretable_confidence_intervals

# from pathlib import Path
# import numpy as np
# from src.config import SRC


add_profession = ["yes", "no"]
add_political = ["yes", "no"]
january = ["yes", "no"]


@pytask.mark.parametrize(
    "case, depends_on, produces",
    [
        (
            {
                "january": f"{jan_data}",
            },
            BLD
            / "data"
            / f"sparse_modelling_df_add_profession_{profession}_add_political_{political_data}.csv",
            BLD
            / "analysis"
            / f"sparse_modelling_adaptive_lasso_add_profession_{profession}_add_political_{political_data}_january_{jan_data}.csv",
        )
        for profession, political_data, jan_data in itertools.product(
            add_profession, add_political, january
        )
    ],
)
def task_adaptive_lasso_real_data(case, depends_on, produces):

    data = pd.read_csv(depends_on)
    data = data.drop(["personal_id"], axis=1)

    if case["january"] == "yes":
        y = data[["vaccine_intention_jan"]]
        X = data.drop(["vaccine_intention_jan", "vaccine_intention_jul"], axis=1)
    else:
        y = data[["vaccine_intention_jul"]]
        X = data.drop(["vaccine_intention_jan", "vaccine_intention_jul"], axis=1)

    y_numpy = y.to_numpy()
    X_numpy = X.to_numpy()

    n, p = X_numpy.shape

    res = adaptive_lasso_tuned(
        X=X_numpy,
        y=y_numpy,
        first_stage="OLS",
        intercept=False,
        cross_valid_split=False,
    )
    df_clean, intercept_val = interpretable_confidence_intervals(
        adaptive_lasso_tuned_obj=res, intercept=False
    )
    df_clean["variable_name"] = list(X)
    df_clean["intercept_value"] = intercept_val

    df_clean.to_csv(produces)
