library("tidyLPA")
library("dplyr")
library("rjson")
library("ggplot2")

rm(list = ls())


args = commandArgs(trailingOnly=TRUE)


estimator_specs <- fromJSON(file=args[1])

# Load data
df <- read.csv(file=args[2], header=TRUE)
df <- df[ , -which(names(df) %in% c("personal_id"))]


output = data.frame(matrix(nrow=0, ncol=18))
# train the optimal model
estimated_model = df %>%
single_imputation() %>%
estimate_profiles(n_profiles=estimator_specs$Classes,
                  models=estimator_specs$Model)


profile_plot_model = plot_profiles(estimated_model, sd = FALSE, rawdata=FALSE, add_line=TRUE)

# save data to disk
ggsave(args[3], profile_plot_model, width=15, height=8.5)
