#Clear environemnt 
rm(list=ls())
#Read data 
pimadata<-read.csv('pima-indians-diabetes.data.txt',header = FALSE)
#Load libraries
library(caret)
library(klaR)
#separate features and label from data
pimafeatures <-pimadata[,-c(9)]
pimalabel <-as.factor(pimadata[,9])
trdataindex<-createDataPartition(y=pimalabel,p=0.8,list =FALSE)
#Get training features and label 
trpf <-pimafeatures[trdataindex,]
trpl <-pimalabel[trdataindex]
#Get test features and label
testpf <-pimafeatures[-trdataindex,]
testpl <-pimalabel[-trdataindex]
#Biuld a Naive Bayes model, Use Cross-Validation of 10
nbmodel<-train(trpf,trpl,'nb',trControl=trainControl(method='cv',number=10))
testresults<-predict(nbmodel,newdata = testpf)
correcttepr <- testresults == testpl
pimatescore <- sum(correcttepr)/(sum(correcttepr) + sum(!correcttepr))
print(pimatescore)