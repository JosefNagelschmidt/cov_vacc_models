import numpy as np


def selection_power(true_support, selected_support):

    """Calculates various metrics to evaluate the performance of different model-selection methods for a given artificial (and known) DGP.

    Args:
        true_support (np.ndarray): logical vector of shape (p, ), indicating which regressor variables are in fact relevant, which is known since the DGP is known
        selected_support (np.ndarray): logical vector of shape (p, ), indicating which regressor variables were chosen relevant (i.e. were selected) by an arbitrary model-selection procedure

    Returns:
        dict: dict containing:

            **share_of_truth_uncovered** (*float*): Correctly selected relevant variables relative to total number of relevant variables \n
            **ratio_total_select_coeffs_true_coeffs** (*float*): Number of selected variables relative to total number of relevant variables \n
            **false_pos_share_true_support** (*float*): Number of mistakenly selected variables (which are irrelevant) relative to total number of relevant variables \n
            **false_pos_share_right_selection** (*float*): Number of mistakenly selected variables (i.e. irrelevant ones) relative to correctly selected (i.e. relevant) variables

    """

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

    """Determines whether elements from the vector *true_theta_vec* are within certain bounds as specified in *conf_int_matrix*. The procedure works elementwise.

    Args:
        true_theta_vec (np.ndarray): Vector of true coefficients (in a partly linear model) from a given DGP. One can also pass only relevant (i.e. non-zero) coefficients from the true model, but the dimensions between *true_theta_vec* and *conf_int_matrix* must match
        conf_int_matrix (np.ndarray): confidence intervals for the elements in *true_theta_vec*. Lower bounds are in the first column, upper bounds in the second column

    Returns:
        (np.ndarray): logical vector, indicating which coefficients from *true_theta_vec* are within the bounds from *conf_int_matrix*
    """

    coverage = np.greater(
        true_theta_vec, conf_int_matrix[:, 0].reshape((conf_int_matrix.shape[0], 1))
    ) & np.less(
        true_theta_vec, conf_int_matrix[:, 1].reshape((conf_int_matrix.shape[0], 1))
    )
    return coverage.flatten()
