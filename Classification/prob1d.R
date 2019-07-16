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
#Biuld a SVM model,
svmmodel<-svmlight(trpf,trpl,pathsvm='/Users/ramya/Documents/CS498aml/Rwork/svm_light_osx.8.4_i7/')
testresults<-predict(svmmodel,newdata = testpf)
prclass<-testresults$class
correcttepr <- prclass == testpl
pimatescore <- sum(correcttepr)/(sum(correcttepr) + sum(!correcttepr))
print(pimatescore)