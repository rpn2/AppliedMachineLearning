#Clear environemnt 
rm(list=ls())
library(glmnet)
library(caret)

#Seed setting
set.seed(1234)
#Read data 
datainp<-read.csv('default of credit card clients.csv',sep = ",",header = TRUE, stringsAsFactors=F)
set.seed(2)
data <-datainp[ ,-1]
ccfeatures <- data[, 1:23]
cclabel <- datainp$Y

kfolds = 10
accuracy <-array(dim = kfolds)
error <-array(dim = kfolds)
#indices for k-fold cross-validation
folds<-createFolds(cclabel,kfolds,returnTrain=FALSE)


for (trial in 1:kfolds){
  test_index <- unlist(folds[trial],use.names=FALSE)
  #Get training features and label
  trdata <- data[-test_index,]
  #Get test features and label
  testdata <-data[test_index,]
  model <- glm(Y ~ . - Y, data = trdata, family=binomial(link='logit'))
  fitted.results <- predict(model,newdata=testdata,type='response')
  fitted.results_t <- ifelse(fitted.results > 0.5,1,0)
  misClasificError <- mean(fitted.results_t != testdata$Y)
  accuracy[trial] = 1-misClasificError
  error[trial] = misClasificError
}


#Calculate average of kfolds trials
AverageAccuracy<- mean(accuracy)
print(AverageAccuracy)
print(error)