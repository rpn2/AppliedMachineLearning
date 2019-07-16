#Clear environemnt 
rm(list=ls())
library(glmnet)
library(DAAG)

#Read data 
datainp<-read.csv('default_plus_chromatic_features_1059_tracks.txt',sep = ",",header = FALSE, stringsAsFactors=F)
set.seed(2)

features = datainp[, 1:116]
latitude = datainp[, 117]
longitude = datainp[, 118]

#unregularized

lat.unreg = lm(V117 ~ . - V118 - V117 ,data=datainp)
cvunreg <- cv.lm(data=datainp, lat.unreg, m=20)


#Ridge
lat.fit = cv.glmnet(as.matrix(features), latitude,alpha=0, nfolds = 20)
plot(lat.fit, main="Latitude Ridge regression",cex.main=1.0)
lambda.lat <- lat.fit$lambda.min
latmse.min <- min(lat.fit$cvm)
cat(lambda.lat,latmse.min)


#Lasso
lat.lasso = cv.glmnet(as.matrix(features), latitude,alpha=1, nfolds = 20)
plot(lat.lasso, main="Latitude Lasso regression",cex.main=1.0)
lambda.lasso <- lat.lasso$lambda.min
lassomse.min <- min(lat.lasso$cvm)
cat(lambda.lasso,lassomse.min)
coeficient=coef(lat.lasso, s=lambda.lasso)
num = nnzero(coeficient)
cat("Including intercept", num)

#Elastic, 0.25
lat.e1 = cv.glmnet(as.matrix(features), latitude,alpha=0.25, nfolds = 20)
plot(lat.e1, main="Latitude elastic regression, alpha = 0.25 ",cex.main=1.0)
lambda.e1 <- lat.e1$lambda.min
e1mse.min <- min(lat.e1$cvm)
cat(lambda.e1,e1mse.min)
coeficient1=coef(lat.e1 , s=lambda.e1)
num1 = nnzero(coeficient1)
cat("Including intercept", num1)


#Elastic, 0.5
lat.e2 = cv.glmnet(as.matrix(features), latitude,alpha=0.5, nfolds = 20)
plot(lat.e2, main="Latitude elastic regression, alpha = 0.5 ",cex.main=1.0)
lambda.e2 <- lat.e2$lambda.min
e2mse.min <- min(lat.e2$cvm)
cat(lambda.e2,e2mse.min)
coeficient2=coef(lat.e2 , s=lambda.e2)
num2 = nnzero(coeficient2)
cat("Including intercept", num2)

#Elastic, 0.75
lat.e3 = cv.glmnet(as.matrix(features), latitude,alpha=0.75, nfolds = 20)
plot(lat.e3, main="Latitude elastic regression, alpha = 0.75 ",cex.main=1.0)
lambda.e3 <- lat.e3$lambda.min
e3mse.min <- min(lat.e3$cvm)
cat(lambda.e3,e3mse.min)
coeficient3=coef(lat.e3 , s=lambda.e3)
num3 = nnzero(coeficient3)
cat("Including intercept", num3)

