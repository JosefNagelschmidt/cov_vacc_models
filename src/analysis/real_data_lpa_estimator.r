library("tidyLPA")
library("dplyr")
library("rjson")

rm(list = ls())


args = commandArgs(trailingOnly=TRUE)


estimator_specs <- fromJSON(file=args[1])

# Load data
df <- read.csv(file=args[2], header=TRUE)
df <- df[ , -which(names(df) %in% c("personal_id"))]


output = data.frame(matrix(nrow=0, ncol=18))
# train a mixed gaussian models for each combination of estimator_specs
for (n_profiles in estimator_specs$n_profiles){
  for (cov in estimator_specs$covariances){
    estimated_model = df %>%
    # select(all_of(main_vars[[1]])) %>%
    single_imputation() %>%
    estimate_profiles(n_profiles=n_profiles,
                      variances = estimator_specs$variances,
                      covariances = cov)
    # extract performance measures from the R object saved in "estimated_model"
    output = rbind(output, estimated_model[[1]][[2]])
  }
}
colnames(output) = names(estimated_model[[1]][[2]])

# Save data to disk.
write.csv(output, file=args[3], col.names=TRUE)
