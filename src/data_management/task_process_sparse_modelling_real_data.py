import itertools
from functools import reduce

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def clean_data(
    covid_data_2020_12,
    covid_data_2020_03,
    covid_data_2020_04,
    covid_data_2020_05,
    covid_data_2020_06,
    covid_data_2020_09,
    background_data,
    political_data,
    add_profession,
    add_political,
):

    # change indeces (drop month):
    covid_data_2020_12.index = covid_data_2020_12.index.droplevel(1)
    covid_data_2020_03.index = covid_data_2020_03.index.droplevel(1)
    covid_data_2020_04.index = covid_data_2020_04.index.droplevel(1)
    covid_data_2020_05.index = covid_data_2020_05.index.droplevel(1)
    covid_data_2020_06.index = covid_data_2020_06.index.droplevel(1)
    covid_data_2020_09.index = covid_data_2020_09.index.droplevel(1)

    covid_data_2020_12_select = covid_data_2020_12[
        [
            "vaccine_intention_jan",
            "vaccine_intention_jul",
            "p_2m_infected",
            "p_2m_acquaintance_infected",
            "p_2m_hospital_if_infect_self",
            "p_2m_infected_and_pass_on",
            "infection_diagnosed",
            "nervous_month",
            "depressed_month",
            "calm_month",
            "gloomy_month",
            "happy_month",
            "work_status",
            "support_childcare_open",
            "support_cafe_open",
            "support_transport_rules",
            "support_no_visitors",
            "support_curfew_high_risk",
            "support_curfew_non_crucial",
            "support_curfew",
            "support_close_cc_some_schools",
            "support_close_cc_all_schools",
            "covid_health_concern",
            "covid_vaccine_effective",
            "covid_vaccine_safe",
            "covid_test_comply",
            "hours_workplace",
            "hours_home",
        ]
    ]
    covid_data_2020_03_select = covid_data_2020_03[
        [
            "trust_gov",
            "avoid_busy_places",
            "avoid_public_places",
            "maintain_distance",
            "adjust_school_work",
            "quarantine_symptoms",
            "quarantine_no_symptoms",
            "effect_close_schools",
            "effect_close_sports",
            "effect_close_food_service",
            "effect_close_most_stores",
            "effect_forbid_hospital_visits",
            "effect_curfew_high_risk",
            "effect_curfew_non_crucial",
            "effect_mask",
            "effect_pray",
            "effect_wash_hands",
            "effect_doctor_ill",
            "effect_doctor_healthy",
            "effect_avoid_public_places",
            "effect_avoid_high_risk",
            "effect_avoid_hospital",
            "effect_avoid_gastro",
            "effect_avoid_public_transp",
            "contact_older_people",
            "comply_bc_civic",
            "comply_bc_fear_punished",
            "comply_bc_protect_self",
            "comply_bc_protect_close",
            "comply_bc_protect_all",
            "comply_bc_high_risk",
            "disobey_bc_freedom",
            "disobey_bc_not_punish",
            "disobey_bc_unjustified",
            "disobey_bc_obligations",
            "disobey_bc_ineffective",
            "disobey_bc_unaffected",
            "disobey_bc_bored",
            "comply_curfew_self",
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
            "n_contacts_personal",
            "n_contacts_distanced",
            "change_contacts_personal",
            "change_contacts_distanced",
        ]
    ]
    covid_data_2020_05_select = covid_data_2020_05[
        [
            "approp_gov_medical",
            "approp_gov_restrict_public_life",
            "approp_gov_econ_mitigation",
            "approp_gov_communication",
            "concern_4w_bored",
            "concern_4w_serious_ill",
            "concern_4w_infect_others",
            "concern_4w_loved_ill",
            "concern_4w_food",
            "concern_4w_health_care",
            "concern_4w_fav_shop_bancrupt",
            "health_comparison_6m",
        ]
    ]
    covid_data_2020_09_select = covid_data_2020_09[
        ["avoid_cafe", "avoid_theater", "avoid_public_transport"]
    ]

    if add_profession == "yes":
        background_data_select = background_data[
            [
                "age",
                "hh_members",
                "hh_children",
                "civil_status",
                "location_urban",
                "edu_4",
                "profession",
                "gender",
                "origin",
                "net_income",
            ]
        ]
    elif add_profession == "no":
        # alternatively, exploring the removal of "profession" with approx. 500 obs more:
        background_data_select = background_data[
            [
                "age",
                "hh_members",
                "hh_children",
                "civil_status",
                "location_urban",
                "edu_4",
                "gender",
                "origin",
                "net_income",
            ]
        ]

    if add_political == "yes":
        # this lets the final sample drop by another 500 obs:
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

    # prepare covid_data_2020_12_select
    covid_data_2020_12_select = covid_data_2020_12_select.dropna(
        subset=[
            "p_2m_infected",
            "p_2m_acquaintance_infected",
            "p_2m_hospital_if_infect_self",
            "p_2m_infected_and_pass_on",
            "vaccine_intention_jan",
            "vaccine_intention_jul",
            "infection_diagnosed",
            "nervous_month",
            "depressed_month",
            "calm_month",
            "gloomy_month",
            "happy_month",
            "work_status",
            "support_childcare_open",
            "support_cafe_open",
            "support_transport_rules",
            "support_no_visitors",
            "support_curfew_high_risk",
            "support_curfew_non_crucial",
            "support_curfew",
            "support_close_cc_some_schools",
            "support_close_cc_all_schools",
            "covid_health_concern",
            "covid_vaccine_effective",
            "covid_vaccine_safe",
            "covid_test_comply",
        ]
    )

    covid_data_2020_12_select_expand = covid_data_2020_12_select.merge(
        pd.get_dummies(
            covid_data_2020_12_select.work_status, prefix="work_status", drop_first=True
        ),
        how="left",
        on="personal_id",
    )
    covid_data_2020_12_select_expand = covid_data_2020_12_select_expand.merge(
        pd.get_dummies(
            covid_data_2020_12_select_expand.infection_diagnosed,
            prefix="infection_diagnosed",
            drop_first=True,
        ),
        how="left",
        on="personal_id",
    )
    covid_data_2020_12_select_expand = covid_data_2020_12_select_expand.drop(
        ["work_status", "infection_diagnosed"], axis=1
    )

    covid_data_2020_12_cleaned = covid_data_2020_12_select_expand.replace(
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
        }
    )

    # merging here into "employed" is necessary if we want to include many
    # other variables that are conditional on being employed.
    covid_data_2020_12_cleaned["work_status_employed"] = (
        covid_data_2020_12_cleaned["work_status_self-employed"]
        + covid_data_2020_12_cleaned["work_status_employed"]
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

    covid_data_2020_12_cleaned[
        "work_status_employed:hours_workplace"
    ] = covid_data_2020_12_cleaned["work_status_employed"] * covid_data_2020_12_cleaned[
        "hours_workplace"
    ].fillna(
        0
    )
    covid_data_2020_12_cleaned[
        "work_status_employed:hours_home"
    ] = covid_data_2020_12_cleaned["work_status_employed"] * covid_data_2020_12_cleaned[
        "hours_home"
    ].fillna(
        0
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
            "work_status_self-employed",
            "hours_home",
            "hours_workplace",
        ],
        axis=1,
    )

    covid_data_2020_12_cleaned = covid_data_2020_12_cleaned.dropna()

    # prepare covid_data_2020_03_select

    covid_data_2020_03_select = covid_data_2020_03_select.dropna(
        subset=[
            "trust_gov",
            "avoid_busy_places",
            "avoid_public_places",
            "maintain_distance",
            "adjust_school_work",
            "quarantine_symptoms",
            "quarantine_no_symptoms",
            "effect_close_schools",
            "effect_close_sports",
            "effect_close_food_service",
            "effect_close_most_stores",
            "effect_forbid_hospital_visits",
            "effect_curfew_high_risk",
            "effect_curfew_non_crucial",
            "effect_mask",
            "effect_pray",
            "effect_wash_hands",
            "effect_doctor_ill",
            "effect_doctor_healthy",
            "effect_avoid_public_places",
            "effect_avoid_high_risk",
            "effect_avoid_hospital",
            "effect_avoid_gastro",
            "effect_avoid_public_transp",
            "contact_older_people",
            "comply_curfew_self",
        ]
    )

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

    covid_data_2020_03_cleaned["subj_effect_behav_proven"] = covid_data_2020_03_cleaned[
        [
            "effect_mask",
            "effect_wash_hands",
            "effect_avoid_public_places",
            "effect_avoid_gastro",
            "effect_avoid_public_transp",
        ]
    ].sum(axis=1)

    covid_data_2020_03_cleaned[
        "subj_effect_behav_unproven"
    ] = covid_data_2020_03_cleaned[["effect_pray"]]

    covid_data_2020_03_cleaned = covid_data_2020_03_cleaned.drop(
        [
            "effect_mask",
            "effect_pray",
            "effect_wash_hands",
            "effect_doctor_ill",
            "effect_doctor_healthy",
            "effect_avoid_public_places",
            "effect_avoid_high_risk",
            "effect_avoid_hospital",
            "effect_avoid_gastro",
            "effect_avoid_public_transp",
        ],
        axis=1,
    )

    covid_data_2020_03_cleaned_expand = covid_data_2020_03_cleaned.merge(
        pd.get_dummies(
            covid_data_2020_03_cleaned.comply_curfew_self, prefix="comply_curfew_self"
        ),
        how="left",
        on="personal_id",
    )
    covid_data_2020_03_cleaned_expand = covid_data_2020_03_cleaned_expand.drop(
        ["comply_curfew_self", "comply_curfew_self_critical profession"], axis=1
    )

    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes:comply_bc_civic"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes"
    ] * covid_data_2020_03_cleaned_expand[
        "comply_bc_civic"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes:comply_bc_fear_punished"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes"
    ] * covid_data_2020_03_cleaned_expand[
        "comply_bc_fear_punished"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes:comply_bc_protect_self"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes"
    ] * covid_data_2020_03_cleaned_expand[
        "comply_bc_protect_self"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes:comply_bc_protect_close"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes"
    ] * covid_data_2020_03_cleaned_expand[
        "comply_bc_protect_close"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes:comply_bc_protect_all"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes"
    ] * covid_data_2020_03_cleaned_expand[
        "comply_bc_protect_all"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes:comply_bc_high_risk"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_yes"
    ] * covid_data_2020_03_cleaned_expand[
        "comply_bc_high_risk"
    ].fillna(
        0
    )

    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_freedom"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_freedom"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_not_punish"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_not_punish"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_unjustified"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_unjustified"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_obligations"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_obligations"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_ineffective"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_ineffective"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_unaffected"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_unaffected"
    ].fillna(
        0
    )
    covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no:disobey_bc_bored"
    ] = covid_data_2020_03_cleaned_expand[
        "comply_curfew_self_no"
    ] * covid_data_2020_03_cleaned_expand[
        "disobey_bc_bored"
    ].fillna(
        0
    )

    covid_data_2020_03_cleaned_expand = covid_data_2020_03_cleaned_expand.drop(
        [
            "comply_bc_civic",
            "comply_bc_fear_punished",
            "comply_bc_protect_self",
            "comply_bc_protect_close",
            "comply_bc_protect_all",
            "comply_bc_high_risk",
            "disobey_bc_freedom",
            "disobey_bc_not_punish",
            "disobey_bc_unjustified",
            "disobey_bc_obligations",
            "disobey_bc_ineffective",
            "disobey_bc_unaffected",
            "disobey_bc_bored",
        ],
        axis=1,
    )

    # prepare covid_data_2020_04_select

    covid_data_2020_04_select = covid_data_2020_04_select.dropna()

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

    # prepare covid_data_2020_05_select
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

    # prepare covid_data_2020_09_select
    covid_data_2020_09_select = covid_data_2020_09_select.dropna()

    conditions = [
        (
            covid_data_2020_09_select["avoid_cafe"]
            == "not before the outbreak and not now"
        )
        & (
            covid_data_2020_09_select["avoid_theater"]
            == "not before the outbreak and not now"
        )
        & (
            covid_data_2020_09_select["avoid_public_transport"]
            == "not before the outbreak and not now"
        )
    ]

    choices = [0.0]
    covid_data_2020_09_select["avoidance_determined"] = np.select(
        conditions, choices, default=1.0
    )

    covid_data_2020_09_select = covid_data_2020_09_select.replace(
        {
            "before the outbreak, but not now": 3,
            "much less often than before the outbreak": 2,
            "a little less often than before the outbreak": 1,
            "as often as before the outbreak": 0,
            "not before the outbreak and not now": np.nan,
            "more often than before the outbreak": -1,
        }
    )

    covid_data_2020_09_select["avoidance_score"] = covid_data_2020_09_select[
        ["avoid_cafe", "avoid_theater", "avoid_public_transport"]
    ].mean(axis=1)

    covid_data_2020_09_select[
        "avoidance_determined:avoidance_score"
    ] = covid_data_2020_09_select["avoidance_determined"] * covid_data_2020_09_select[
        "avoidance_score"
    ].fillna(
        0
    )

    covid_data_2020_09_select = covid_data_2020_09_select.drop(
        ["avoid_cafe", "avoid_theater", "avoid_public_transport", "avoidance_score"],
        axis=1,
    )

    # prepare background_data_select

    background_data_select = background_data_select.dropna()

    background_data_select_expand = background_data_select.merge(
        pd.get_dummies(
            background_data_select.civil_status, prefix="civil_status", drop_first=True
        ),
        how="left",
        on="personal_id",
    )

    if add_profession == "yes":
        background_data_select_expand = background_data_select_expand.merge(
            pd.get_dummies(
                background_data_select_expand.profession,
                prefix="profession",
                drop_first=True,
            ),
            how="left",
            on="personal_id",
        )
        background_data_select_expand = background_data_select_expand.drop(
            ["profession"], axis=1
        )

    background_data_select_expand = background_data_select_expand.merge(
        pd.get_dummies(
            background_data_select_expand.gender, prefix="gender", drop_first=True
        ),
        how="left",
        on="personal_id",
    )
    background_data_select_expand = background_data_select_expand.merge(
        pd.get_dummies(
            background_data_select_expand.origin, prefix="origin", drop_first=True
        ),
        how="left",
        on="personal_id",
    )

    background_data_select_expand = background_data_select_expand.drop(
        ["civil_status", "gender", "origin"], axis=1
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

    if add_political == "yes":
        dfs = [
            covid_data_2020_12_cleaned,
            covid_data_2020_03_cleaned_expand,
            covid_data_2020_04_cleaned,
            covid_data_2020_05_cleaned,
            covid_data_2020_09_select,
            background_data_cleaned,
            political_data_cleaned,
        ]
    else:
        dfs = [
            covid_data_2020_12_cleaned,
            covid_data_2020_03_cleaned_expand,
            covid_data_2020_04_cleaned,
            covid_data_2020_05_cleaned,
            covid_data_2020_09_select,
            background_data_cleaned,
        ]

    df_final = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="left"
        ),
        dfs,
    )
    df_final_cleaned = df_final.dropna()
    return df_final_cleaned


add_profession = ["yes", "no"]
add_political = ["yes", "no"]


@pytask.mark.parametrize(
    "case, depends_on, produces",
    [
        (
            {
                "add_profession": f"{profession}",
                "add_political": f"{political_data}",
            },
            {
                "covid_data_2020_12": SRC
                / "original_data"
                / "covid_data_2020_12.pickle",
                "covid_data_2020_03": SRC
                / "original_data"
                / "covid_data_2020_03.pickle",
                "covid_data_2020_04": SRC
                / "original_data"
                / "covid_data_2020_04.pickle",
                "covid_data_2020_05": SRC
                / "original_data"
                / "covid_data_2020_05.pickle",
                "covid_data_2020_06": SRC
                / "original_data"
                / "covid_data_2020_06.pickle",
                "covid_data_2020_09": SRC
                / "original_data"
                / "covid_data_2020_09.pickle",
                "background_data": SRC
                / "original_data"
                / "background_data_merged.pickle",
                "politics_values": SRC / "original_data" / "politics_values.pickle",
            },
            BLD
            / "data"
            / f"sparse_modelling_df_add_profession_{profession}_add_political_{political_data}.csv",
        )
        for profession, political_data in itertools.product(
            add_profession, add_political
        )
    ],
)
def task_process_sparse_modelling_real_data(case, depends_on, produces):

    covid_data_2020_12 = pd.read_pickle(depends_on["covid_data_2020_12"])
    covid_data_2020_03 = pd.read_pickle(depends_on["covid_data_2020_03"])
    covid_data_2020_04 = pd.read_pickle(depends_on["covid_data_2020_04"])
    covid_data_2020_05 = pd.read_pickle(depends_on["covid_data_2020_05"])
    covid_data_2020_06 = pd.read_pickle(depends_on["covid_data_2020_06"])
    covid_data_2020_09 = pd.read_pickle(depends_on["covid_data_2020_09"])
    background_data = pd.read_pickle(depends_on["background_data"])
    political_data = pd.read_pickle(depends_on["politics_values"])

    df_final_cleaned = clean_data(
        covid_data_2020_12=covid_data_2020_12,
        covid_data_2020_03=covid_data_2020_03,
        covid_data_2020_04=covid_data_2020_04,
        covid_data_2020_05=covid_data_2020_05,
        covid_data_2020_06=covid_data_2020_06,
        covid_data_2020_09=covid_data_2020_09,
        background_data=background_data,
        political_data=political_data,
        add_profession=case["add_profession"],
        add_political=case["add_political"],
    )

    df_final_cleaned.to_csv(produces)
