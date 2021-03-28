.. _model_code:

**********
Model code
**********

This directory contains most code related to the simulation study for the scientific computing course.
It specifies functions for the data generating processes, my own implementation of the lasso and the adaptive lasso,
and competing post-model-selection inference estimators based on sample splitting.
Additionally, some benchmark metrics are specified.

The data generating processes in the simulations
================================================

.. automodule:: src.model_code.dgp
    :members:


The custom lasso and adaptive lasso implementation
==================================================

.. automodule:: src.model_code.estimators
    :members:


External estimators used as benchmarks
======================================

.. automodule:: src.model_code.external_estimators
    :members:


Performance metrics
===================

.. automodule:: src.model_code.helpers
    :members:


Unit tests of custom implementation
===================================

.. automodule:: src.model_code.test_estimators
    :members: