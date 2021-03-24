from functools import reduce

import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def clean_aux_data(
    covid_data_2020_12,
    covid_data_2020_03,
    covid_data_2020_04,
    covid_data_2020_05,
    background_data,
    political_data,
):
    """Function to be called by `task_lpa_process_aux_var_data`. Cleans and then returns the input data files.

    Args:
        covid_data_2020_12 (pd.DataFrame): the dataframe from the pickle file
            named *covid_data_2020_12.pickle*  (preprocessed data from a LISS questionnaire)
        covid_data_2020_03 (pd.DataFrame): the dataframe from the pickle file
            named *covid_data_2020_03.pickle*  (preprocessed data from a LISS questionnaire)
        covid_data_2020_04 (pd.DataFrame): the dataframe from the pickle file
            named *covid_data_2020_04.pickle*  (preprocessed data from a LISS questionnaire)
        covid_data_2020_05 (pd.DataFrame): the dataframe from the pickle file
            named *covid_data_2020_05.pickle*  (preprocessed data from a LISS questionnaire)
        background_data (pd.DataFrame): the dataframe from the pickle file
            named *background_data_merged.pickle*  (preprocessed data from a LISS questionnaire)
        political_data (pd.DataFrame): the dataframe from the pickle file
            named *politics_values.pickle* (preprocessed data from a LISS questionnaire)

    Returns:
        list: list containing:

            **covid_data_2020_12_cleaned** (*pd.DataFrame*): cleaned version of *covid_data_2020_12* \n
            **covid_data_2020_03_cleaned** (*pd.DataFrame*): cleaned version of *covid_data_2020_03*  \n
            **covid_data_2020_04_cleaned** (*pd.DataFrame*): cleaned version of *covid_data_2020_04* \n
            **covid_data_2020_05_cleaned** (*pd.DataFrame*): cleaned version of *covid_data_2020_05*  \n
            **political_data_cleaned** (*pd.DataFrame*): cleaned version of *political_data* \n
            **background_data_cleaned** (*pd.DataFrame*): cleaned version of *background_data*

    """

    # change indeces (drop month):
    covid_data_2020_12.index = covid_data_2020_12.index.droplevel(1)
    covid_data_2020_03.index = covid_data_2020_03.index.droplevel(1)
    covid_data_2020_04.index = covid_data_2020_04.index.droplevel(1)
    covid_data_2020_05.index = covid_data_2020_05.index.droplevel(1)

    covid_data_2020_12_select = covid_data_2020_12[
        [
            "vaccine_intention_jan",
            "vaccine_intention_jul",
            "covid_test_comply",
            "happy_month",
            "nervous_month",
            "depressed_month",
            "calm_month",
            "gloomy_month",
            "support_childcare_open",
            "support_cafe_open",
            "support_transport_rules",
            "support_no_visitors",
            "support_curfew_high_risk",
            "support_curfew_non_crucial",
            "support_curfew",
            "support_close_cc_some_schools",
            "support_close_cc_all_schools",
        ]
    ]

    covid_data_2020_05_select = covid_data_2020_05[
        [
            "concern_4w_bored",
            "concern_4w_serious_ill",
            "concern_4w_infect_others",
            "concern_4w_loved_ill",
            "concern_4w_food",
            "concern_4w_health_care",
            "concern_4w_fav_shop_bancrupt",
        ]
    ]

    covid_data_2020_04_select = covid_data_2020_04[
        [
            "feeling_emptiness",
            "enough_people_approach",
            "enough_people_trust",
            "enough_people_bond",
            "miss_people_around",
            "feel_let_down_often",
        ]
    ]

    covid_data_2020_03_select = covid_data_2020_03[
        [
            "avoid_busy_places",
            "avoid_public_places",
            "maintain_distance",
            "adjust_school_work",
            "quarantine_symptoms",
            "quarantine_no_symptoms",
        ]
    ]

    background_data_select = background_data[
        [
            "age",
            "hh_members",
            "hh_children",
            "location_urban",
            "edu_4",
            "gender",
            "net_income",
        ]
    ]

    background_data_select = background_data_select.dropna()
    background_data_select_expand = background_data_select.merge(
        pd.get_dummies(background_data_select.gender, prefix="gender", drop_first=True),
        how="left",
        on="personal_id",
    )
    background_data_select_expand = background_data_select_expand.drop(
        ["gender"], axis=1
    )

    background_data_cleaned = background_data_select_expand.replace(
        {
            "poor": 1,
            "moderate": 2,
            "good": 3,
            "very good": 4,
            "excellent": 5,
            "Not urban": 1,
            "Slightly urban": 2,
            "Moderately urban": 3,
            "Very urban": 4,
            "Extremely urban": 5,
            "primary": 1,
            "lower_secondary": 2,
            "upper_secondary": 3,
            "tertiary": 4,
        }
    )

    political_data_2018 = political_data.query("year == 2018")
    political_data_2018.index = political_data_2018.index.droplevel(1)

    political_data_select = political_data_2018[
        [
            "news_interest",
            "political_interest",
            "parties_not_care",
            "ppl_no_influence",
            "politically_able",
            "understand_pol_issues",
            "how_rightwing",
        ]
    ]

    political_data_select = political_data_select.dropna()

    political_data_cleaned = political_data_select.replace(
        {"not interested": 1, "fairly interested": 2, "very interested": 3}
    )

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

    covid_data_2020_12_cleaned["support_current_measures"] = covid_data_2020_12_cleaned[
        ["support_childcare_open", "support_cafe_open", "support_transport_rules"]
    ].sum(axis=1)
    covid_data_2020_12_cleaned[
        "support_possible_measures"
    ] = covid_data_2020_12_cleaned[
        [
            "support_no_visitors",
            "support_curfew_high_risk",
            "support_curfew_non_crucial",
            "support_curfew",
            "support_close_cc_some_schools",
            "support_close_cc_all_schools",
        ]
    ].sum(
        axis=1
    )

    covid_data_2020_12_cleaned = covid_data_2020_12_cleaned.drop(
        [
            "support_childcare_open",
            "support_cafe_open",
            "support_transport_rules",
            "support_no_visitors",
            "support_curfew_high_risk",
            "support_curfew_non_crucial",
            "support_curfew",
            "support_close_cc_some_schools",
            "support_close_cc_all_schools",
        ],
        axis=1,
    )
    covid_data_2020_12_cleaned = covid_data_2020_12_cleaned.dropna()

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

    covid_data_2020_03_cleaned["compliance_measures"] = covid_data_2020_03_cleaned[
        [
            "avoid_busy_places",
            "avoid_public_places",
            "maintain_distance",
            "adjust_school_work",
            "quarantine_symptoms",
            "quarantine_no_symptoms",
        ]
    ].sum(axis=1)

    covid_data_2020_03_cleaned = covid_data_2020_03_cleaned.drop(
        [
            "avoid_busy_places",
            "avoid_public_places",
            "maintain_distance",
            "adjust_school_work",
            "quarantine_symptoms",
            "quarantine_no_symptoms",
        ],
        axis=1,
    )

    covid_data_2020_04_cleaned = covid_data_2020_04_select.replace(
        {
            "no": 1,
            "more or less": 2,
            "yes": 3,
            "much less": 1,
            "less": 2,
            "roughly equal": 3,
            "more": 4,
            "much more": 5,
        }
    )

    covid_data_2020_04_cleaned["loneliness"] = (
        covid_data_2020_04_cleaned["feeling_emptiness"]
        - covid_data_2020_04_cleaned["enough_people_approach"]
        - covid_data_2020_04_cleaned["enough_people_trust"]
        - covid_data_2020_04_cleaned["enough_people_bond"]
        + covid_data_2020_04_cleaned["miss_people_around"]
        + covid_data_2020_04_cleaned["feel_let_down_often"]
    )

    covid_data_2020_04_cleaned = covid_data_2020_04_cleaned.drop(
        [
            "feeling_emptiness",
            "enough_people_approach",
            "enough_people_trust",
            "enough_people_bond",
            "miss_people_around",
            "feel_let_down_often",
        ],
        axis=1,
    )

    covid_data_2020_05_select = covid_data_2020_05_select.dropna()

    covid_data_2020_05_cleaned = covid_data_2020_05_select.replace(
        {
            "too few": 1,
            "rather too few": 2,
            "right amount": 3,
            "rather too many": 4,
            "too many": 5,
            "1 not worried at all": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5 very worried": 5,
            "much worse": 1,
            "a little worse": 2,
            "same": 3,
            "a little better": 4,
            "much better": 5,
        }
    )

    covid_data_2020_05_cleaned["concerns"] = covid_data_2020_05_cleaned[
        [
            "concern_4w_bored",
            "concern_4w_serious_ill",
            "concern_4w_infect_others",
            "concern_4w_loved_ill",
            "concern_4w_food",
            "concern_4w_health_care",
            "concern_4w_fav_shop_bancrupt",
        ]
    ].sum(axis=1)

    covid_data_2020_05_cleaned = covid_data_2020_05_cleaned.drop(
        [
            "concern_4w_bored",
            "concern_4w_serious_ill",
            "concern_4w_infect_others",
            "concern_4w_loved_ill",
            "concern_4w_food",
            "concern_4w_health_care",
            "concern_4w_fav_shop_bancrupt",
        ],
        axis=1,
    )

    return [
        covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned,
        covid_data_2020_04_cleaned,
        covid_data_2020_05_cleaned,
        political_data_cleaned,
        background_data_cleaned,
    ]


def merge_aux_subsets(
    covid_data_2020_12_cleaned,
    covid_data_2020_03_cleaned,
    covid_data_2020_04_cleaned,
    covid_data_2020_05_cleaned,
    background_data_cleaned,
    political_data_cleaned,
):

    """Function to be called by `task_lpa_process_aux_var_data`. Reads in the cleaned dataframes passed
    by *clean_aux_data* and merges them.

    Args:
        covid_data_2020_12_cleaned (pd.DataFrame): cleaned version of *covid_data_2020_12*,
            passed by function *clean_aux_data*
        covid_data_2020_03_cleaned (pd.DataFrame): cleaned version of *covid_data_2020_03*,
            passed by function *clean_aux_data*
        covid_data_2020_04_cleaned (pd.DataFrame): cleaned version of *covid_data_2020_04*,
            passed by function *clean_aux_data*
        covid_data_2020_05_cleaned (pd.DataFrame): cleaned version of *covid_data_2020_05*,
            passed by function *clean_aux_data*
        background_data_cleaned (pd.DataFrame): cleaned version of *background_data*,
            passed by function *clean_aux_data*
        political_data_cleaned (pd.DataFrame): cleaned version of *political_data*,
            passed by function *clean_aux_data*


    Returns:
        df_final_cleaned (pd.DataFrame): Merged df of the cleaned input dataframes

    """

    dfs = [
        covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned,
        covid_data_2020_04_cleaned,
        covid_data_2020_05_cleaned,
        background_data_cleaned,
        political_data_cleaned,
    ]
    df_final = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="left"
        ),
        dfs,
    )
    df_final_cleaned = df_final.dropna()
    return df_final_cleaned


@pytask.mark.depends_on(
    {
        "covid_data_2020_04": SRC / "original_data" / "covid_data_2020_04.pickle",
        "covid_data_2020_05": SRC / "original_data" / "covid_data_2020_05.pickle",
        "covid_data_2020_12": SRC / "original_data" / "covid_data_2020_12.pickle",
        "covid_data_2020_03": SRC / "original_data" / "covid_data_2020_03.pickle",
        "background_data": SRC / "original_data" / "background_data_merged.pickle",
        "politics_values": SRC / "original_data" / "politics_values.pickle",
    }
)
@pytask.mark.produces(BLD / "data" / "lpa_df_aux_vars.csv")
def task_lpa_create_partial_dfs(depends_on, produces):
    covid_data_2020_12 = pd.read_pickle(depends_on["covid_data_2020_12"])
    covid_data_2020_03 = pd.read_pickle(depends_on["covid_data_2020_03"])
    covid_data_2020_04 = pd.read_pickle(depends_on["covid_data_2020_04"])
    covid_data_2020_05 = pd.read_pickle(depends_on["covid_data_2020_05"])
    political_data = pd.read_pickle(depends_on["politics_values"])
    background_data = pd.read_pickle(depends_on["background_data"])
    (
        covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned,
        covid_data_2020_04_cleaned,
        covid_data_2020_05_cleaned,
        political_data_cleaned,
        background_data_cleaned,
    ) = clean_aux_data(
        covid_data_2020_12=covid_data_2020_12,
        covid_data_2020_03=covid_data_2020_03,
        covid_data_2020_04=covid_data_2020_04,
        covid_data_2020_05=covid_data_2020_05,
        background_data=background_data,
        political_data=political_data,
    )
    df_final_cleaned = merge_aux_subsets(
        covid_data_2020_12_cleaned=covid_data_2020_12_cleaned,
        covid_data_2020_03_cleaned=covid_data_2020_03_cleaned,
        covid_data_2020_04_cleaned=covid_data_2020_04_cleaned,
        covid_data_2020_05_cleaned=covid_data_2020_05_cleaned,
        background_data_cleaned=background_data_cleaned,
        political_data_cleaned=political_data_cleaned,
    )
    df_final_cleaned.to_csv(produces)
