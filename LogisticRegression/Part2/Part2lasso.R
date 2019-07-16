#Clear environemnt 
rm(list=ls())
library(glmnet)
library(caret)
#Read data 
datainp<-read.csv('default of credit card clients.csv',sep = ",",header = TRUE, stringsAsFactors=F)
set.seed(2)
data <-datainp[ ,-1]
ccfeatures <- data[, 1:23]
cclabel <- datainp$Y

kfolds = 10
accuracy <-array(dim = kfolds)
lmin <-array(dim = kfolds)
error <- array(dim = kfolds)
numc <- array(dim = kfolds)

#indices for k-fold cross-validation
folds<-createFolds(cclabel,kfolds,returnTrain=FALSE)

for (trial in 1:kfolds){
  test_index <- unlist(folds[trial],use.names=FALSE)
  #Get training features and label
  trdata <- data[-test_index,]
  #Get test features and label
  testdata <-data[test_index,]
  
  # Run cross validation to get good lambda
  cvfit = cv.glmnet(as.matrix(trdata[,1:23]), trdata[,24],alpha=1, nfolds = 10,type.measure="class",family="binomial")
  plot(cvfit, main="Lasso regression",cex.main=1.0)
  lambda.cc <- cvfit$lambda.min
  
  #Find minimum lambda
  lambda.cc <- cvfit$lambda.min
  lmin[trial] <- lambda.cc
  
  
  #calculate misclassification error and accuracy on test
  fitted.results <- predict(cvfit,newx=as.matrix(testdata[,1:23]), s = lambda.cc,type = "class")
  fitted.results_t <- ifelse(fitted.results > 0.5,1,0)
  misClasificError <- mean(fitted.results_t != testdata$Y)
  accuracy[trial] = 1-misClasificError
  error[trial] = misClasificError
  
  #Number of co-efficients
  coeficient=coef(cvfit, s=lambda.cc)
  numc[trial] = nnzero(coeficient)
  
}  


#Calculate average of 10 trials
AverageAccuracy<- mean(accuracy)
print(AverageAccuracy)  
print(error)
print(lmin)
print(numc)