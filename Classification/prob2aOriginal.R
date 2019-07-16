#Clear environemnt 
rm(list=ls())
# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
trainfeatures = load_image_file("train-images-idx3-ubyte")
testfeatures  = load_image_file("t10k-images-idx3-ubyte")

# load labels
trainlabel = as.factor(load_label_file("train-labels-idx1-ubyte"))
testlabel  = as.factor(load_label_file("t10k-labels-idx1-ubyte"))

#Apply threshold
trainfeatures[trainfeatures < 35] <- 0
trainfeatures[trainfeatures >= 35] <- 1
testfeatures[testfeatures < 35] <- 0
testfeatures[testfeatures >= 35] <- 1


#Gaussian
library(naivebayes)
nbmnist <- naive_bayes(trainfeatures,trainlabel,usekernel = FALSE)
nbpredict <- predict(nbmnist,testfeatures)
correcttepr <- nbpredict == testlabel
mnisttescore <- sum(correcttepr)/(sum(correcttepr) + sum(!correcttepr))
cat("Accuracy of Gaussian Naive Bayes = ", mnisttescore)

#bernoulli

#Convert features values to categorical
trfnew <- data.frame(lapply(trainfeatures, factor,levels=c(0,1)))
tefnew <- data.frame(lapply(testfeatures, factor,levels=c(0,1)))


nbmnistb <- naive_bayes(trfnew,trainlabel,usekernel = FALSE,laplace = 3)
nbpredictb <- predict(nbmnistb,tefnew)
correctteprb <- nbpredictb == testlabel
mnisttescoreb <- sum(correctteprb)/(sum(correctteprb) + sum(!correctteprb))
cat("Accuracy of Bernoulli Naive Bayes = ", mnisttescoreb)


