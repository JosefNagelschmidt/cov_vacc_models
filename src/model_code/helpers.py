import numpy as np


def selection_power(true_support, selected_support):
    count_true_support = np.sum(true_support)
    count_selected_support = np.sum(selected_support)
    res = np.logical_and(true_support, selected_support)
    truth_uncovered = np.sum(res)
    ratio_selected_true = count_selected_support / count_true_support
    share_of_truth_uncovered = truth_uncovered / count_true_support

    false_positives = np.full((len(true_support),), False, dtype=bool)
    for idx, val in enumerate(selected_support):
        if val:
            if not true_support[idx]:
                false_positives[idx] = True

    false_pos_share_true_support = np.sum(false_positives) / count_true_support
    if truth_uncovered != 0:
        false_pos_share_right_selection = np.sum(false_positives) / truth_uncovered
    else:
        false_pos_share_right_selection = np.nan

    return {
        "share_of_truth_uncovered": share_of_truth_uncovered,
        "ratio_total_select_coeffs_true_coeffs": ratio_selected_true,
        "false_pos_share_true_support": false_pos_share_true_support,
        "false_pos_share_right_selection": false_pos_share_right_selection,
    }


def true_params_in_conf_interval(true_theta_vec, conf_int_matrix):
    coverage = np.greater(
        true_theta_vec, conf_int_matrix[:, 0].reshape((conf_int_matrix.shape[0], 1))
    ) & np.less(
        true_theta_vec, conf_int_matrix[:, 1].reshape((conf_int_matrix.shape[0], 1))
    )
    return coverage.flatten()
