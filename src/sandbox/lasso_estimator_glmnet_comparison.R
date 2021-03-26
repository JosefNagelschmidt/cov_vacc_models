library(glmnet)

df = read.csv(file="/home/helios/Code/Github/cov_vacc_models/src/sandbox/test.csv", header = FALSE)
fit <- glmnet(scaled.dat[,-1], scaled.dat[,1], standardize = FALSE)
coef(fit, s = 0.01, exact = TRUE, x = scaled.dat[,-1] , y= scaled.dat[,1],standardize = FALSE)

scaled.dat = scale(df)

fit <- glmnet(df[,-1], df[,1])
coef(fit, s = 0.01, exact = TRUE, x = df[,-1] , y= df[,1])


library("clusterGeneration")

data <- genRandomClust(numClust = 10,sepVal = 0.15,numNonNoisy = 90,numNoisy = 10,numReplicate = 1,fileName = "chk1")
cluster_data = data$datList$chk1_1

genPositiveDefMat$datList

cov_mat = genPositiveDefMat(dim = 100,covMethod = "eigen")
cov = cov_mat$Sigma
a = diag(cov)
diag(cov) <- a / 6
cor = cov2cor(cov)

library("MASS")
highCor<-matrix(seq(from=0.9, to= 0.0001, length.out = 100),100,100)


n <- 100
A <- matrix(runif(n^2)*4-1, ncol=n)
cov <- t(A) %*% A

