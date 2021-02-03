from pathlib import Path

import pytask

from src.config import BLD
from src.config import SRC


model_names = ["1", "2", "3", "4"]
specs = [
    {
        "r": Path("lpa_optimal_params_plot.r"),
        "deps": [
            SRC / "model_specs" / f"lpa_optimal_params_set_{model_name}.json",
            BLD / "data" / f"lpa_df_var_subset_{model_name}.csv",
            BLD / "data" / "lpa_df_aux_vars.csv",
            SRC / "model_specs" / "lpa_aux_set.json",
        ],
        "result": BLD / "figures" / f"lpa_var_set_{model_name}_profile_plot.pdf",
        "aux_barplot": BLD / "figures" / f"lpa_aux_var_set_{model_name}_barplot.pdf",
    }
    for model_name in model_names
]


@pytask.mark.parametrize(
    "r, depends_on, produces",
    [
        (
            [str(x) for x in [*s["deps"], s["result"], s["aux_barplot"]]],
            [s["r"], *s["deps"]],
            [s["result"], s["aux_barplot"]],
        )
        for s in specs
    ],
)
def task_estimate():
    pass
