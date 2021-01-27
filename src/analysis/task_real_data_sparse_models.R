## Benchmarking feature selection algorithms:

# 1. Lasso
# 2. Adaptive Lasso (on whole dataset and split setting) (first stage: ridge)
# 3. Elastic Net
# 4. Cluster representative Lasso (CRL)
# 5. Boruta
# 6. Adaptive Group Lasso

###### Performance measure: #######
## (i): true positive rate (sensitivity of the algorithm)
## (ii): falsely discovered relevant variables
## (iii): stability if sample fluctuates
## (iv): RMSE of linear model of selected variables out-of-sample
###
## https://link.springer.com/chapter/10.1007/978-3-662-45620-0_2 : Boruta
## Correlated variables in regression: Clustering and sparse estimation (Bühlmann): CRL
## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3209717/#S11title : Adaptive Group Lasso
##

rm(list = ls())
## import libraries:
library(Boruta)
library(glmnet)
library(caret)
library(MASS)
library(clusterGeneration)
library(ranger)

# load the data:

df = read.csv(file="/Users/light/Documents/Code/Git/vacc_intention_models/bld/data/sparse_modelling_data.csv")

y = df$vaccine_intention_jan
X = df[ , !names(df) %in% c("personal_id","vaccine_intention_jan", "vaccine_intention_jul", "comply_curfew_self_yes",
                            "comply_curfew_self_no")] ## works as expected

adaLasso = function(X_mat, y_vals, firstStage, k){
  # This initial process gives us the set of coefficients from the best ridge regression model chosen based on the
  # 10-fold cross-validation using the squared error metric as the model performance metric.
  ## Perform ridge regression with 10-fold CV
  if(firstStage == "ridge"){
    first_stage_cv <- cv.glmnet(x = X_mat, y = y_vals,
                                ## type.measure: loss to use for cross-validation.
                                type.measure = "mse",
                                standardize = TRUE,
                                nfold = k,
                                ## ‘alpha = 1’ is the lasso penalty, and ‘alpha = 0’ the ridge penalty.
                                alpha = 0)

  }
  else if(firstStage == "lasso"){
    first_stage_cv <- cv.glmnet(x = X_mat, y = y_vals,
                                ## type.measure: loss to use for cross-validation.
                                type.measure = "mse",
                                standardize = TRUE,
                                nfold = k,
                                ## ‘alpha = 1’ is the lasso penalty, and ‘alpha = 0’ the ridge penalty.
                                alpha = 1)
  }
  else{
    print("Please specify first stage method for generating weights.")
  }
  ## Extract coefficients at the error-minimizing lambda
  #lambda_min = first_stage_cv$lambda.min
  #best_first_stage_coef <- as.numeric(coef(first_stage_cv, s = lambda_min))

  # now, perform adaptive lasso:

  penalty <- 1/abs(matrix(coef(first_stage_cv, s=first_stage_cv$lambda.min)[, 1][2:(ncol(X_mat)+1)] ))^1 ## Using gamma = 1
  penalty[penalty[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999

  alasso_cv <- cv.glmnet(x = X_mat, y = y_vals,
                         ## type.measure: loss to use for cross-validation.
                         type.measure = "mse",
                         ## K = 10 is the default.
                         nfold = k,
                         standardize = TRUE,
                         ## ‘alpha = 1’ is the lasso penalty, and ‘alpha = 0’ the ridge penalty.
                         alpha = 1,
                         ##
                         ## penalty.factor: Separate penalty factors can be applied to each
                         ##           coefficient. This is a number that multiplies ‘lambda’ to
                         ##           allow differential shrinkage. Can be 0 for some variables,
                         ##           which implies no shrinkage, and that variable is always
                         ##           included in the model. Default is 1 for all variables (and
                         ##           implicitly infinity for variables listed in ‘exclude’). Note:
                         ##           the penalty factors are internally rescaled to sum to nvars,
                         ##           and the lambda sequence will reflect this change.
                         penalty.factor = penalty,
                         ## prevalidated array is returned
                         keep = TRUE)

  #best_alasso_coef <- coef(alasso_cv, s = alasso_cv$lambda.min)
  return(alasso_cv)
}

lasso = function(X_mat, y_vals, k){
  estimator <- cv.glmnet(x = X_mat, y = y_vals,
                         ## type.measure: loss to use for cross-validation.
                         type.measure = "mse",
                         standardize = TRUE,
                         nfold = k,
                         ## ‘alpha = 1’ is the lasso penalty, and ‘alpha = 0’ the ridge penalty.
                         alpha = 1)
  return(estimator)
}

elasNet = function(X_mat, y_vals, k){
  # Make a custom trainControl - use ROC as a model selection criteria
  model <- train(x= X_mat, y = y_vals , method = "glmnet", metric =  "RMSE", maximize = FALSE, tuneLength = k)
  #Check the model
  return(model)
}



#a = glmnet(x = as.matrix(X), y = y, alpha = 1, lambda = 0.08848323)
#b = lasso(X_mat = as.matrix(X), y_vals = y, k = 5)
c = adaLasso(X_mat = as.matrix(X), y_vals = y, k = 10, firstStage = "lasso")

#coef.glmnet(a)
#coef(b, s = b$lambda.min)
paste(coef(c, s = c$lambda.min))

#bor = Boruta(x = X,  y = y, holdHistory = FALSE, maxRuns=450, getImp = getImpExtraZ)
#bor
#bor$finalDecision

