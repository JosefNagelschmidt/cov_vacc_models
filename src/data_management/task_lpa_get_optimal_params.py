import json

import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    {
        "set_1": BLD / "analysis" / "lpa_var_set_1_performance.csv",
        "set_2": BLD / "analysis" / "lpa_var_set_2_performance.csv",
        "set_3": BLD / "analysis" / "lpa_var_set_3_performance.csv",
        "set_4": BLD / "analysis" / "lpa_var_set_4_performance.csv",
    }
)
@pytask.mark.produces(
    {
        "set_1": SRC / "model_specs" / "lpa_optimal_params_set_1.json",
        "set_2": SRC / "model_specs" / "lpa_optimal_params_set_2.json",
        "set_3": SRC / "model_specs" / "lpa_optimal_params_set_3.json",
        "set_4": SRC / "model_specs" / "lpa_optimal_params_set_4.json",
    }
)
def task_lpa_get_optimal_params(depends_on, produces):
    set_1 = pd.read_csv(depends_on["set_1"])
    set_2 = pd.read_csv(depends_on["set_2"])
    set_3 = pd.read_csv(depends_on["set_3"])
    set_4 = pd.read_csv(depends_on["set_4"])

    set_1_opt_params = (
        (set_1.drop(set_1.columns[0], axis=1).round(2).sort_values(by=["BIC"]))[
            ["Model", "Classes"]
        ]
        .iloc[0]
        .to_dict()
    )
    set_2_opt_params = (
        (set_2.drop(set_2.columns[0], axis=1).round(2).sort_values(by=["BIC"]))[
            ["Model", "Classes"]
        ]
        .iloc[0]
        .to_dict()
    )
    set_3_opt_params = (
        (set_3.drop(set_3.columns[0], axis=1).round(2).sort_values(by=["BIC"]))[
            ["Model", "Classes"]
        ]
        .iloc[0]
        .to_dict()
    )
    set_4_opt_params = (
        (set_4.drop(set_4.columns[0], axis=1).round(2).sort_values(by=["BIC"]))[
            ["Model", "Classes"]
        ]
        .iloc[0]
        .to_dict()
    )

    with open(produces["set_1"], "w") as f1:
        json.dump(set_1_opt_params, f1)
    with open(produces["set_2"], "w") as f2:
        json.dump(set_2_opt_params, f2)
    with open(produces["set_3"], "w") as f3:
        json.dump(set_3_opt_params, f3)
    with open(produces["set_4"], "w") as f4:
        json.dump(set_4_opt_params, f4)
