from pathlib import Path

import pytask

from src.config import BLD
from src.config import SRC


model_names = ["1", "2", "3", "4"]

specs = [
    {
        "r": Path("real_data_lpa_estimator.r"),
        "deps": [
            SRC / "model_specs" / "lpa_estimator_specs.json",
            BLD / "data" / f"lpa_df_var_subset_{model_name}.csv",
        ],
        "result": BLD / "analysis" / f"lpa_var_set_{model_name}_performance.csv",
    }
    for model_name in model_names
]


@pytask.mark.parametrize(
    "r, depends_on, produces",
    [
        (
            [str(x) for x in [*s["deps"], s["result"]]],
            [s["r"], *s["deps"]],
            [s["result"]],
        )
        for s in specs
    ],
)
def task_estimate():
    pass
