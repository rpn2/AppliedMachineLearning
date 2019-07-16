#Clear environemnt 
rm(list=ls())
#Read data 
pimadata<-read.csv('pima-indians-diabetes.data.txt',header = FALSE)
#Load libraries
library(caret)
library(klaR)
#separate features and label from data
pimafeatures <-pimadata[,-c(9)]
pimalabel <-pimadata[,9]
#create arrays for resulst storage
pimatrscore <-array(dim = 10)
pimatescore <-array(dim = 10)
#Average for 10 times

#Replace 0's by NA for columns 3,4,6,8
pimafeatures$V3[pimafeatures$V3 == 0] <- NA
pimafeatures$V4[pimafeatures$V4 == 0] <- NA
pimafeatures$V6[pimafeatures$V6 == 0] <- NA
pimafeatures$V8[pimafeatures$V8 == 0] <- NA

for (trial in 1:10){
  #Create a stratified partition of the data : 80% training, 20% test, return index
  trdataindex<-createDataPartition(y=pimalabel,p=0.8,list =FALSE)
  localpf <- pimafeatures
  #Get training features and label 
  trpf <-localpf[trdataindex,]
  trpl <-pimalabel[trdataindex]
  #Get test features and label
  testpf <-localpf[-trdataindex,]
  testpl <-pimalabel[-trdataindex]
  #Create logic index for positive samples
  poslabelindex<- trpl > 0 
  #Separate training data into positive and negative 
  pospf <-trpf[poslabelindex,]
  negpf <-trpf[!poslabelindex,]
  #priors calculaion
  totaltrsize = length(trpl)
  totalpostrsize = length(poslabelindex[poslabelindex == TRUE])
  totalnegtrsize = length(poslabelindex[poslabelindex == FALSE])
  postrratio = totalpostrsize/totaltrsize
  negtrratio = totalnegtrsize/totaltrsize
  logposprior = log(postrratio)
  lognegprior = log(negtrratio)
  
  #Calculate respective means
  pospfmean <-sapply(pospf,mean,na.rm=TRUE)
  negpfmean <-sapply(negpf,mean,na.rm=TRUE)
  #Calculate respective standard deviation
  pospfsd <-sapply(pospf,sd,na.rm=TRUE)
  negpfsd <-sapply(negpf,sd,na.rm=TRUE)
  
  #Subract mean from every element and divide by std for test data 
  pospfoffsets <-t(t(testpf)-pospfmean)
  pospfscales<-t(t(pospfoffsets)/pospfsd)
  #Calculate log likelihood
  poslogs<--(1/2)*rowSums(apply(pospfscales,c(1, 2), function(x)x^2),na.rm=TRUE)-sum(log(pospfsd))
  #calculate posterior log
  posteriorposlog <- poslogs + logposprior
  
  #Subract mean from every element and divide by std for test data 
  negpfoffsets <-t(t(testpf)-negpfmean)
  negpfscales<-t(t(negpfoffsets)/negpfsd)
  #Calculate log likelihood
  neglogs<--(1/2)*rowSums(apply(negpfscales,c(1, 2), function(x)x^2),na.rm=TRUE)-sum(log(negpfsd))
  #calculate posterior log
  posteriorneglog <- neglogs + lognegprior
  
  #Test data accuracy calculations
  testlabelpr <- posteriorposlog >  posteriorneglog
  correcttepr <- testlabelpr == testpl
  pimatescore[trial] <- sum(correcttepr)/(sum(correcttepr) + sum(!correcttepr))
  
}

#Calculate average of 10 trials
AverageAccuracy<- mean(pimatescore)
print(AverageAccuracy)

