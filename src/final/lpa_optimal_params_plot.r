library("tidyLPA")
library("dplyr")
library("rjson")
library("ggplot2")
library("reshape2")

rm(list = ls())


args = commandArgs(trailingOnly=TRUE)
estimator_specs <- fromJSON(file=args[1])
aux_specs <- fromJSON(file=args[4])

normalize = function(vec){
  res = vector(mode = "numeric")
  min = min(vec)
  max = max(vec)
  for (i in vec) {
    val = (i) / (sum(vec))
    res = c(res, val)
  }
  return(res)
}

make_barplot = function(model, df, df_aux_vars, group_size, aux_vars){
  #group_shares = vector(mode = "numeric")
  #for (i in 1:group_size) {
  #  group_shares = c(group_shares, sum(model[[1]][[1]][[15]] == 1) / length(model[[1]][[1]][[15]]))
  #}
  df["model_groups"] = model[[1]][[1]][[15]]
  df_merged <- merge(df, df_aux_vars, by = "personal_id")
  model_group_averages = aggregate(df_merged[, aux_vars], list(df_merged$model_groups), mean)[,-1]
  for (name in aux_vars){
    model_group_averages[name] = normalize(model_group_averages[name])
  }
  model_group_averages = as.data.frame(t(model_group_averages))
  colnames(model_group_averages) = paste0(rep(paste("G"), group_size), as.character(1:group_size))
  model_group_averages$aux_vars <- rownames(model_group_averages)
  model_group_averages <- melt(model_group_averages, id.vars=c("aux_vars"))
  ggplot(model_group_averages, aes(factor(aux_vars), value, fill = variable)) +
    geom_bar(stat="identity", position = "dodge") +
    scale_fill_brewer(palette = "Set1") + coord_flip()
}

# Load data
df <- read.csv(file=args[2], header=TRUE)
df_aux_vars <- read.csv(file=args[3], header=TRUE)

#output = data.frame(matrix(nrow=0, ncol=18))
# train the optimal model
estimated_model = df[ , -which(names(df) %in% c("personal_id"))] %>%
single_imputation() %>%
estimate_profiles(n_profiles=estimator_specs$Classes,
                  models=estimator_specs$Model)


profile_plot_model = plot_profiles(estimated_model, sd = FALSE, rawdata=FALSE, add_line=TRUE)
aux_var_barplot = make_barplot(model=estimated_model, df = df, df_aux_vars=df_aux_vars, group_size=estimator_specs$Classes, aux_vars = aux_specs$aux_var_set)

# save data to disk
ggsave(args[5], profile_plot_model, width=15, height=8.5)
ggsave(args[6], aux_var_barplot, width=8.5, height=15)
