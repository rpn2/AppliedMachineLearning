#Clear the environemnt 
rm(list=ls())
#Use Rcurl to read adult data from URL
library(RCurl)
library(caret)
read_data <- function() {
  # specify the URL for the Iris data CSV
  urlfile <-'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
  # download the file
  downloaded <- getURL(urlfile, ssl.verifypeer=FALSE)
  # treat the text data as a steam so we can read from it
  connection <- textConnection(downloaded)
  # parse the downloaded data as CSV
  dataset1 <- read.csv(connection, header=FALSE)
}

read_data_test <- function() {
  # specify the URL for the Iris data CSV
  urlfile <-'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
  # download the file
  downloaded <- getURL(urlfile, ssl.verifypeer=FALSE)
  # treat the text data as a steam so we can read from it
  connection <- textConnection(downloaded)
  # parse the downloaded data as CSV
  dataset <- read.csv(connection, header=FALSE)
  
}
adult_data <-read_data()
adult_data_test_temp <- read_data_test()
adult_data_test <- adult_data_test_temp[2:nrow(adult_data_test_temp), ]
adult_data_test[,1] <-as.numeric(as.character(adult_data_test[,1]))
#The website has labels with a '.' in test data, removed '.'
adult_data_test[,15] <- gsub('<=50K.', '<=50K', adult_data_test[,15])
adult_data_test[,15] <- gsub('>50K.', '>50K', adult_data_test[,15])

#Combine all data entries : total 48842
totaldata  <- rbind(adult_data,adult_data_test) 


#Get columns with continuous data (numeric) only
nums <- sapply(totaldata, is.numeric)
#Get feature of that correspond to numeric type only
totaldata_features <- totaldata[, nums]
totaldata_label <- totaldata[, 15]
#Convert the label to 1 and -1 numeric data  >50k is 1 ; <=50K is -1
totaldata_lc0 <- totaldata_label == " >50K"
totaldata_lc <- 1*totaldata_lc0
totaldata_lc[totaldata_lc == 0] <- -1

#Omit rows with missing data, no NA's found : below statement is redundant 
totaldata_clean <- na.omit(totaldata_features)
totaldata_scaled <- scale(totaldata_clean,center = TRUE, scale = TRUE)

# 90%
trvaldataindex <- createDataPartition(y=totaldata_lc, p=0.9, list = FALSE)
trvalfeat <- totaldata_scaled[trvaldataindex, ]
trvallabel <- totaldata_lc[trvaldataindex]
# 10% test
testfeat <- totaldata_scaled[-trvaldataindex, ]
testlabel <- totaldata_lc[-trvaldataindex]
# 80% train
trdataindex <- createDataPartition(y=trvallabel, p=0.89, list = FALSE)
trfeat <- trvalfeat[trdataindex, ]
trlabel <- trvallabel[trdataindex]
# 10% validation
valfeat <- trvalfeat[-trdataindex, ]
vallabel <- trvallabel[-trdataindex]

#Declare constants 
#lamdavals <- c(1e-3, 1e-2, 1e-1, 1)
lamdavals <- c(1e-4,1e-3,1e-2,1e-1,1)
# 50 epochs
totalepoch <- 50 #50
#300 steps per epoch
totalsteps <- 300
#check accuracy every 30 steps
totalcheck <- 30
#learning rate∂
m = 100
n = 5000

#initialize accuracy matrix
accu <- matrix(nrow =length(lamdavals), ncol=as.integer(totalepoch*totalsteps/totalcheck)) 
#initialize magnitude matrix
magofa <- matrix(nrow =length(lamdavals), ncol=as.integer(totalepoch*totalsteps/totalcheck)) 
#End of epoch statistics collection
accuepoch <-  matrix(nrow =length(lamdavals), ncol=as.integer(totalepoch))
magofaepoch <- matrix(nrow =length(lamdavals), ncol=as.integer(totalepoch)) 

#holds accuracy and magnitude for each lambda
accuracylamda <-vector(mode = "numeric", length(lamdavals))
magnitudelamda <-vector(mode = "numeric", length(lamdavals))

#Finding correct lamda
for (lamdaindex in 1: length(lamdavals)){
  k = 1
  cat("\n\n")
  set.seed(12345)
  #Assign random values to initial values of a and b
  b <- runif(1)
  a <- runif(ncol(trfeat))
  
  accuracy <-vector(mode = "numeric", length=as.integer(totalepoch*totalsteps/totalcheck))
  magnitude<-vector(mode = "numeric", length=as.integer(totalepoch*totalsteps/totalcheck))
  
  
  
  for (epochnum in 1: totalepoch){
    lr <- m / (n + epochnum )
    
    #Choose 50 random samples from training for plotting
    treval <-sample(1:nrow(trfeat), 50, replace=FALSE)
    trevalfeat <- trfeat[treval,]
    trevallabel <- trlabel[treval]
    trinnfeat <- trfeat[-treval,]
    trinnlabel <- trlabel[-treval]
    #Choose 300 indexes, sampling with replacement
    indexlst <- sample(1:nrow(trinnfeat), totalsteps, replace=TRUE)
    for (step in 1:totalsteps) {
      ftunit <- trinnfeat[indexlst[step],]
      lbunit <- trinnlabel[indexlst[step]]
      #Calculate Y(ax +b)
      localsum <- lbunit * ((sum(t(a)*ftunit) + b))
      # Update co-efficients based on gradient rule
      if (localsum >= 1) {
        a = a - lr*lamdavals[lamdaindex]*a
        b = b + 0
      }
      else {
        a = a - lr*(lamdavals[lamdaindex]*a -lbunit*ftunit )
        b = b - lr*(-lbunit)
        
      }
      #Every 30 steps, plot the accuracy and maginitude for report purposes
      if(step%%totalcheck == 0){
        #Check accuracy of validation
        corr = 0
        for (evalin in 1 : nrow(trevalfeat)){
          rescalc = sum(t(a)*trevalfeat[evalin,]) + b
          res = 1*(rescalc >=0)
          if (res == 0){
            res = -1
          }
          #cat("\n",step, rescalc, step,res,trevallabel[evalin])  
          if (res == trevallabel[evalin]) {
            corr = corr + 1
          } 
        }
        #Plotting purposes
        accuracy[k] = corr*100/nrow(trevalfeat)
        magnitude[k] = sum(t(a)*a)
        k = k + 1
        
      }
      
    }
    
  }
  #Plotting purposes
  accu[lamdaindex,] = accuracy
  magofa[lamdaindex,] = magnitude
  
  #End of a lambda, find accuracy of 10% evaluation data for choosen lamba 
  corr = 0
  for (evalin in 1 : nrow(valfeat)){
    rescalc = sum(t(a)*valfeat[evalin,]) + b
    res = 1*(rescalc >=0)
    if (res == 0){
      res = -1
    }
    
    if (res == vallabel[evalin]) {
      corr = corr + 1
    } 
  }
  
  #Decision purposes
  accuracylamda[lamdaindex] = corr*100/nrow(valfeat)
  magnitudelamda[lamdaindex] = sqrt(sum(t(a)*a))
  
}

for (lamdaindex in 1:length(lamdavals)) {
  cat("for lamda: ", lamdavals[lamdaindex], "\n")
  cat("accuracy is: ", accuracylamda[lamdaindex], "\n")
  cat("magnitude is: ", magnitudelamda[lamdaindex], "\n\n")
}


#Plot accuracy 
yrange <- range(10:100) 
xrange <- range(1:totalepoch*totalsteps/totalcheck) 
plot(xrange,yrange,type="n",xlab ='Every 30 Steps', ylab ='Held-out Accuracy in %')
colors <- rainbow(length(lamdavals))

for (i in 1:length(lamdavals)) {
  lines(accu[i,],type="l", col=colors[i])

}
legend(100, 40, legend = lamdavals, cex=0.8, col=colors,lty=1:2)


#Plot magnitude

yrange <- range(magofa) 
xrange <- range(1:totalepoch*totalsteps/totalcheck) 
plot(xrange,yrange,type="n",xlab ='Every 30 Steps', ylab ='Magnitude of a')
colors <- rainbow(length(lamdavals))

for (i in 1:length(lamdavals)) {
  lines(magofa[i,],type="l", col=colors[i])
  
}
legend(100, 40, legend =lamdavals, cex=0.8, col=colors,lty=1:2)

#Retrain the classifier with 90% data and fixed Lamda value

#Combine the traning and evaluation data frames
#trfeat, trlabel;
#valfeat, vallabel;
trfeat <- cbind(trfeat,trlabel)
valfeat <- cbind(valfeat,vallabel)

trnew <- rbind(trfeat,valfeat)

trnewfeat <-trnew[, 1: ncol(trnew)-1]
trnewlabel <-trnew[, ncol(trnew)]

#Re-train on 90% dataset with fixed lambda
lamdavalsnew <- c(0.01)

set.seed(12345)
#Assign random values to initial values of a and b
b <- runif(1)
a <- runif(ncol(trnewfeat))
#Finding correct lamda
for (lamdaindex in 1: length(lamdavalsnew)){
  for (epochnum in 1: totalepoch){
    lr <- m / (n + epochnum )
    #Choose 300 indexes, sampling with replacement
    indexlst <- sample(1:nrow(trnewfeat), totalsteps, replace=TRUE)
    for (step in 1:totalsteps) {
      ftunit <- trnewfeat[indexlst[step],]
      lbunit <- trnewlabel[indexlst[step]]
      #Calculate Y(ax +b)
      localsum <- lbunit * ((sum(t(a)*ftunit) + b))
      # Update co-efficients based on gradient rule
      if (localsum >= 1) {
        a = a - lr*lamdavalsnew[lamdaindex]*a
        b = b + 0
      }
      else {
        a = a - lr*(lamdavalsnew[lamdaindex]*a -lbunit*ftunit )
        b = b - lr*(-lbunit)
        
      }
    }
  }
 
}

#Evaluate Final Accuracy on test set
corr = 0
for (evalin in 1 : nrow(testfeat)){
  rescalc = sum(t(a)*testfeat[evalin,]) + b
  res = 1*(rescalc >=0)
  if (res == 0){
    res = -1
  }
  
  if (res == testlabel[evalin]) {
    corr = corr + 1
  } 
}

#Decision purposes
FinalAccuracy <- corr*100/nrow(testfeat)

cat("Final Accuracy of Test Set" , FinalAccuracy)

