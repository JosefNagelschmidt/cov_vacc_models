library(glmnet)

df = read.csv(file="/home/helios/Code/Github/cov_vacc_models/src/sandbox/test.csv", header = FALSE)
fit <- glmnet(scaled.dat[,-1], scaled.dat[,1], standardize = FALSE)
coef(fit, s = 0.01, exact = TRUE, x = scaled.dat[,-1] , y= scaled.dat[,1],standardize = FALSE)

scaled.dat = scale(df)

fit <- glmnet(df[,-1], df[,1])
coef(fit, s = 0.01, exact = TRUE, x = df[,-1] , y= df[,1])
