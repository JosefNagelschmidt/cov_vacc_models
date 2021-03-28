.. _analysis:

********
Analysis
********

Documentation of the code in *src/analysis*.


Latent profile analysis
=======================

For four predefined sets of variables of interest (see `src/model_specs/lpa_var_set_x.json`), the main task is to train
different types of mixed gaussian models, each defined by a combination of parameters in `src/model_specs/lpa_estimator_specs.json`.
The BIC criterion then allows a performance comparison across trained models, even with varying complexity (i.e. different number of profiles / classes).

The corresponding files are:

- `real_data_lpa_estimator.r` which trains mixed gaussian models (which determine the latent profiles) for given (and pre-preprocessed) input data and model specifications, and then returns the each models performance (BIC)
- `task_lpa_analysis.py` which simply runs the R script in `real_data_lpa_estimator.r` for all model specifications


Sparse modelling (adaptive lasso)
=================================

The goal here is to do inference and model-selection at the same time on a real dataset from LISS surveys, making use of the oracle properties of the adaptive lasso.
The dependent variable is participants' intention to take a vaccine in january (or july) 2021, the independent variables form a subset of all given answers by participants across several surveys.
The adaptive lasso used in this part is my own implementation.

The corresponding files are:

- `task_adaptive_lasso_real_data.py` which trains the cross-validated adaptive lasso on the pre-processed data from LISS (several sets of variables), and outputs estimated model parameters into a dataframe



Scientific computing benchmarking
=================================

The corresponding files are:

- `task_sim_data_benchmarking.py` which generates a performance overview of different post-model-selection inference strategies with respect to several artificial data generating processes
- `task_sim_real_data_benchmarking.py` which generates a performance overview of different post-model-selection inference strategies with respect to a data generating process that was motivated by the real LISS survey data

For further information on the different post-model-selection inference strategies and the data generating processes, have a look at the `model_code` documentation.