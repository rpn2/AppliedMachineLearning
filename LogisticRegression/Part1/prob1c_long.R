#Clear environemnt 
rm(list=ls())
library(glmnet)

#Read data 
datainp<-read.csv('default_plus_chromatic_features_1059_tracks.txt',sep = ",",header = FALSE, stringsAsFactors=F)
set.seed(2)

features = datainp[, 1:116]
longitude = datainp[, 118]


#unregularized

long.unreg = lm(V118 ~ . - V118 - V117 ,data=datainp)
cvunlong <- cv.lm(data=datainp, long.unreg, m=20)

#Ridge
long.fit = cv.glmnet(as.matrix(features), longitude,alpha=0, nfolds = 20)
plot(long.fit, main="Longtitude Ridge regression",cex.main=1.0)
lambda.long <- long.fit$lambda.min
longmse.min <- min(long.fit$cvm)
cat(lambda.long,longmse.min)


#Lasso
long.lasso = cv.glmnet(as.matrix(features), longitude,alpha=1, nfolds = 20)
plot(long.lasso, main="Longtitude Lasso regression",cex.main=1.0)
lambda.lasso <- long.lasso$lambda.min
lassomse.min <- min(long.lasso$cvm)
cat(lambda.lasso,lassomse.min)
coeficient=coef(long.lasso, s=lambda.lasso)
num = nnzero(coeficient)
cat("Including intercept", num)

#Elastic, 0.25
long.e1 = cv.glmnet(as.matrix(features), longitude,alpha=0.25, nfolds = 20)
plot(long.e1, main="Longtitude elastic regression, alpha = 0.25 ",cex.main=1.0)
lambda.e1 <- long.e1$lambda.min
e1mse.min <- min(long.e1$cvm)
cat(lambda.e1,e1mse.min)
coeficient1=coef(long.e1 , s=lambda.e1)
num1 = nnzero(coeficient1)
cat("Including intercept", num1)


#Elastic, 0.5
long.e2 = cv.glmnet(as.matrix(features), longitude,alpha=0.5, nfolds = 20)
plot(long.e2, main="Longtitude elastic regression, alpha = 0.5 ",cex.main=1.0)
lambda.e2 <- long.e2$lambda.min
e2mse.min <- min(long.e2$cvm)
cat(lambda.e2,e2mse.min)
coeficient2=coef(long.e2 , s=lambda.e2)
num2 = nnzero(coeficient2)
cat("Including intercept", num2)

#Elastic, 0.75
long.e3 = cv.glmnet(as.matrix(features), longitude,alpha=0.75, nfolds = 20)
plot(long.e3, main="Longtitude elastic regression, alpha = 0.75 ",cex.main=1.0)
lambda.e3 <- long.e3$lambda.min
e3mse.min <- min(long.e3$cvm)
cat(lambda.e3,e3mse.min)
coeficient3=coef(long.e3 , s=lambda.e3)
num3 = nnzero(coeficient3)
cat("Including intercept", num3)

