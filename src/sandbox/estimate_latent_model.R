library("tidyLPA")
library("dplyr")

rm(list = ls())

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
# if using RStudio you can use:
script_path = dirname(rstudioapi::getSourceEditorContext()$path)
set_1 = read.csv(paste0(dirname(dirname(script_path)), "/bld/data/lpa_df_var_subset_1.csv"))
set_1 <- set_1[ , -which(names(set_1) %in% c("personal_id"))]

# this selects the best model (BIC) for a rel. small set of regressors:
set_1 %>%
  single_imputation() %>%
  estimate_profiles(n_profiles=c(3,4,5), models = c(1,3)) %>%
  compare_solutions()

# this selects the best model for a larger set of regressors:
df %>%
  select(covid_vaccine_safe, flu_vaccine_safe,
         covid_vaccine_effective, flu_vaccine_effective,
         flu_health_concern, covid_health_concern,
         confidence_healthcare, confidence_science,
         confidence_media, confidence_pol_parties) %>%
  single_imputation() %>%
  estimate_profiles(n_profiles=c(3,4,5,6,7), models = c(1,3)) %>%
  compare_solutions()

# this one is between model_1 and model_2:
df %>%
  select(covid_vaccine_safe, covid_vaccine_effective,
         covid_health_concern, confidence_science,
         confidence_healthcare,
         confidence_media, confidence_pol_parties) %>%
  single_imputation() %>%
  estimate_profiles(n_profiles=c(3,4,5,6,7), models = c(1,3)) %>%
  compare_solutions()


# train the optimal models (change ""n_profiles" and "models" depending on output above)
model_small = df %>%
  select(covid_vaccine_safe, covid_vaccine_effective,
         covid_health_concern, confidence_science,
         confidence_media, confidence_pol_parties) %>%
  single_imputation() %>%
  estimate_profiles(n_profiles=c(5), models = c(1))

model_large = df %>%
  select(covid_vaccine_safe, flu_vaccine_safe,
         covid_vaccine_effective, flu_vaccine_effective,
         flu_health_concern, covid_health_concern,
         confidence_healthcare, confidence_science,
         confidence_media, confidence_pol_parties) %>%
  single_imputation() %>%
  estimate_profiles(n_profiles=c(5), models = c(1))

model_medium = df %>%
  select(covid_vaccine_safe, covid_vaccine_effective,
         covid_health_concern, confidence_science,
         confidence_healthcare,
         confidence_media, confidence_pol_parties) %>%
  single_imputation() %>%
  estimate_profiles(n_profiles=c(5), models = c(1))

# print some summary statistics:
model_small
model_large
model_medium

# plot the latent profiles:
profile_plot_small_model = plot_profiles(model_small, sd = FALSE, rawdata=FALSE, add_line=TRUE)
profile_plot_large_model = plot_profiles(model_large, sd = FALSE, rawdata=FALSE, add_line=TRUE)
profile_plot_medium_model = plot_profiles(model_medium, sd = FALSE, rawdata=FALSE, add_line=TRUE)

# this function calculates group-means for the aux_variables provided in the dataframe,
# groups are based on latent profiles by the models above, group_size is the number
# of latent profiles.
make_barplot = function(model, df, group_size){
  group_shares = vector(mode = "numeric")
  for (i in 1:group_size) {
    group_shares = c(group_shares, sum(model[[1]][[1]][[15]] == 1) / length(model[[1]][[1]][[15]]))
  }
  df["model_groups"] = model[[1]][[1]][[15]]
  aux_vars = c("vaccine_intention_jan","how_rightwing", "location_urban","nervous_month","depressed_month","happy_month","calm_month","gloomy_month","net_income_hh", "age","edu_4")
  model_group_averages = aggregate(df[, aux_vars], list(df$model_groups), mean)[,-1]
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

# visualize the barplots:
aux_var_barplot_small_model = make_barplot(model= model_small, df=df, group_size = 5)
aux_var_barplot_large_model = make_barplot(model= model_large, df=df, group_size = 5)
aux_var_barplot_medium_model = make_barplot(model= model_medium, df=df, group_size = 5)

# only works in RStudio, otherwise specify absolute path to save manually
save_plots_dir = paste0(dirname(dirname(script_path)), "/bld/latent_models/")
setwd(save_plots_dir)

profile_plots = list(profile_plot_small_model,
                     profile_plot_medium_model,
                     profile_plot_large_model)

aux_plots = list(aux_var_barplot_small_model,
                 aux_var_barplot_medium_model,
                 aux_var_barplot_large_model)

p = do.call(marrangeGrob, c(profile_plots,ncol=1,nrow=1))
p = do.call(marrangeGrob, args = list(grobs = profile_plots, ncol=1, nrow=1))
ggsave("profile_plots.pdf", p, width=15, height=8.5)

p1 = do.call(marrangeGrob, c(aux_plots,ncol=1,nrow=1))
p1 = do.call(marrangeGrob, args = list(grobs = aux_plots, ncol=1, nrow=1))
ggsave("aux_plots.pdf", p1, width=10, height=14)
