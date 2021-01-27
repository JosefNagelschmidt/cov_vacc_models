'
The file requires to be called with a model specification, a set of primary variables, and
a set of auxiliary variables, as the arguments;
corresponding json-files must exist in PATH_IN_MODEL_SPECS. These files
need to define dictionaries with keys:
1. model spec. file:
    * n_profiles - the number of latent profiles
    * variances - specifies which variance components to estimate in the finite mixture model
    * covariances - specifies which covariance components to estimate in the finite mixture model
    * rounds - number of repetitions for controlling numerical issues during fitting
2. primary variables:
    * list of sets of variables, in the format {"set_1":["A","B",..], "set_2": ...}
3. auxiliary variables:
    * list of sets of aux. variables; same format as in (2.) above.

The r-file loops over various specifications as defined in PATH_IN_MODEL_SPECS/lpa_estimator_specs.json and
in PATH_IN_MODEL_SPECS/lpa_var_sets.json.
Finally, it stores a dataframe with estimation results; and saves the respective models for later use.
'


library("tidyLPA")
library("dplyr")
library("reshape2")
library("ggplot2")
library("gridExtra")
library("rjson")

rm(list = ls())


args = commandArgs(trailingOnly=TRUE)

main_vars <- fromJSON(file=args[1])
estimator_specs <- fromJSON(file=args[2])

# Load data
df <- read.csv(file=args[3], header=TRUE)

output = data.frame(matrix(nrow=0, ncol=18))
# train the optimal models (change ""n_profiles" and "models" depending on output above)
for (n_profiles in estimator_specs$n_profiles){
  for (cov in estimator_specs$covariances){
    estimated_model = df %>%
    select(all_of(main_vars[[1]])) %>%
    single_imputation() %>%
    estimate_profiles(n_profiles=n_profiles,
                      variances = estimator_specs$variances,
                      covariances = cov)
    # extract performance measures from the R object saved in "estimated_model"
    output = rbind(output, estimated_model[[1]][[2]])
  }
}
colnames(output) = names(estimated_model[[1]][[2]])
output["set_of_vars"] <- strsplit(names(main_vars), split="_")[[1]][[2]]

# Save data to disk.
write.csv(output, file=args[4], col.names=TRUE)
