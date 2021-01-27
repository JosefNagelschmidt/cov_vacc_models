import json
from functools import reduce

import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def scaler(x, lower, upper, df, column_name):
    if pd.isnull(x):
        return x
    else:
        val = lower + (upper - lower) / (
            df[column_name].max() - df[column_name].min()
        ) * (x - df[column_name].min())
        return val


def clean_data(covid_data_2020_12, covid_data_2020_03, political_data):

    # change indeces (drop month):
    covid_data_2020_12.index = covid_data_2020_12.index.droplevel(1)
    covid_data_2020_03.index = covid_data_2020_03.index.droplevel(1)

    covid_data_2020_12_select = covid_data_2020_12[
        [
            "covid_vaccine_safe",
            "flu_vaccine_safe",
            "covid_vaccine_effective",
            "flu_vaccine_effective",
            "covid_health_concern",
            "flu_health_concern",
            "p_2m_infected",
        ]
    ]

    covid_data_2020_03_select = covid_data_2020_03[
        [
            "trust_gov",
            "effect_close_schools",
            "effect_close_sports",
            "effect_close_food_service",
            "effect_close_most_stores",
            "effect_forbid_hospital_visits",
            "effect_curfew_high_risk",
            "effect_curfew_non_crucial",
            "effect_mask",
            "effect_wash_hands",
            "effect_pray",
        ]
    ]

    political_data_2018 = political_data.query("year == 2018")
    political_data_2018.index = political_data_2018.index.droplevel(1)
    political_data_select = political_data_2018[
        ["confidence_science", "confidence_media"]
    ]

    political_data_select = political_data_select.dropna()

    # cleaning:
    covid_data_2020_12_select = covid_data_2020_12_select.dropna()

    covid_data_2020_12_cleaned = covid_data_2020_12_select.replace(
        {
            "totally disagree": 1,
            "disagree": 2,
            "neither/nore": 3,
            "agree": 4,
            "totally agree": 5,
            "never": 1,
            "rarely": 2,
            "sometimes": 3,
            "often": 4,
            "mostly": 5,
            "constantly": 6,
            "too strict": 1,
            "rather too strict": 2,
            "just enough": 3,
            "rather flexible": 4,
            "too flexible": 5,
            "Helemaal mee oneens": 1,
            "Oneens": 2,
            "Niet oneens en niet eens": 3,
            "Eens": 4,
            "Helemaal mee eens": 5,
        }
    )

    covid_data_2020_03_select = covid_data_2020_03_select.dropna()

    covid_data_2020_03_cleaned = covid_data_2020_03_select.replace(
        {
            "1 no confidence at all": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5 a lot of confidence": 5,
            "not effective at all": 1,
            "hardly effective": 2,
            "somewhat effective": 3,
            "effective": 4,
            "very effective": 5,
            "none at all": 1,
            "hardly any": 2,
            "some": 3,
            "a lot": 4,
            "a whole lot": 5,
            "never": 1,
            "once a week": 2,
            "several times a week": 3,
            "daily": 4,
        }
    )

    covid_data_2020_03_cleaned["subj_effect_measures"] = covid_data_2020_03_cleaned[
        [
            "effect_close_schools",
            "effect_close_sports",
            "effect_close_food_service",
            "effect_close_most_stores",
            "effect_forbid_hospital_visits",
            "effect_curfew_high_risk",
            "effect_curfew_non_crucial",
        ]
    ].sum(axis=1)

    covid_data_2020_03_cleaned = covid_data_2020_03_cleaned.drop(
        [
            "effect_close_schools",
            "effect_close_sports",
            "effect_close_food_service",
            "effect_close_most_stores",
            "effect_forbid_hospital_visits",
            "effect_curfew_high_risk",
            "effect_curfew_non_crucial",
        ],
        axis=1,
    )

    covid_data_2020_12_cleaned["p_2m_infected"] = [
        scaler(x, 1, 5, covid_data_2020_12_cleaned, "p_2m_infected")
        for x in covid_data_2020_12_cleaned["p_2m_infected"]
    ]
    covid_data_2020_03_cleaned["subj_effect_measures"] = [
        scaler(x, 1, 5, covid_data_2020_03_cleaned, "subj_effect_measures")
        for x in covid_data_2020_03_cleaned["subj_effect_measures"]
    ]
    political_data_select["confidence_science"] = [
        scaler(x, 1, 5, political_data_select, "confidence_science")
        for x in political_data_select["confidence_science"]
    ]
    political_data_select["confidence_media"] = [
        scaler(x, 1, 5, political_data_select, "confidence_media")
        for x in political_data_select["confidence_media"]
    ]

    return [
        covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned,
        political_data_select,
    ]


def merge_subsets(
    covid_data_2020_12_cleaned,
    covid_data_2020_03_cleaned,
    political_data_select,
    var_set,
):
    covid_data_2020_12_subset = covid_data_2020_12_cleaned[
        var_set["covid_data_2020_12"]
    ]
    covid_data_2020_03_subset = covid_data_2020_03_cleaned[
        var_set["covid_data_2020_03"]
    ]
    political_data_subset = political_data_select[var_set["politics_values"]]
    dfs = [covid_data_2020_12_subset, covid_data_2020_03_subset, political_data_subset]
    df_final = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="left"
        ),
        dfs,
    )
    df_final_cleaned = df_final.dropna()
    return df_final_cleaned


model_names = ["1", "2", "3", "4"]


@pytask.mark.parametrize(
    "depends_on, produces",
    [
        (
            {
                "subset_names": SRC / "model_specs" / f"lpa_var_set_{model_name}.json",
                "covid_data_2020_12": SRC
                / "original_data"
                / "covid_data_2020_12.pickle",
                "covid_data_2020_03": SRC
                / "original_data"
                / "covid_data_2020_03.pickle",
                "politics_values": SRC / "original_data" / "politics_values.pickle",
            },
            BLD / "data" / f"lpa_df_var_subset_{model_name}.csv",
        )
        for model_name in model_names
    ],
)
def task_lpa_create_partial_dfs(depends_on, produces):
    covid_data_2020_12 = pd.read_pickle(depends_on["covid_data_2020_12"])
    covid_data_2020_03 = pd.read_pickle(depends_on["covid_data_2020_03"])
    political_data = pd.read_pickle(depends_on["politics_values"])

    (
        covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned,
        political_data_select,
    ) = clean_data(covid_data_2020_12, covid_data_2020_03, political_data)
    df_final_cleaned = merge_subsets(
        covid_data_2020_12_cleaned=covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned=covid_data_2020_03_cleaned,
        political_data_select=political_data_select,
        var_set=json.loads(depends_on["subset_names"].read_text(encoding="utf-8")),
    )

    df_final_cleaned.to_csv(produces)
