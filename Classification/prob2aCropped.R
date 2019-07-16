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

#Resize the image by 20x 20
library(imager)

#Resize training 
#Convert Data Frame to a matrix 
tempdf = data.matrix(trainfeatures)
#Memory allocation for resized resultant matrix
newdfrow = matrix(nrow =nrow(trainfeatures), ncol = 400,byrow = TRUE )
# For each row of matrix, create a 20 x20 image
for (num in 1:nrow(trainfeatures)){
  
  tempdf1 <- matrix(tempdf[num,], ncol = 28, byrow = TRUE)
  tempdf2 <- resize(autocrop(as.cimg(tempdf1)), size_x = 20, size_y = 20)[, , 1, 1]
  tempdf3 <- as.integer(as.vector(t(tempdf2)))
  newdfrow[num, ] <- tempdf3
  }


croppedtf <-data.frame(newdfrow)
#Resize testing 

tempdf = data.matrix(testfeatures)
newdfrow = matrix(nrow =nrow(testfeatures), ncol = 400,byrow = TRUE )

for (num in 1:nrow(testfeatures)){
  tempdf1 <- matrix(tempdf[num,], ncol = 28, byrow = TRUE)
  tempdf2 <- resize(autocrop(as.cimg(tempdf1)), size_x = 20, size_y = 20)[, , 1, 1]
  tempdf3 <- as.integer(as.vector(t(tempdf2)))
  newdfrow[num, ] <- tempdf3
}

croppedtestf <-data.frame(newdfrow)


#Gaussian
library(naivebayes)
nbmnist <- naive_bayes(croppedtf,trainlabel,usekernel = FALSE)
nbpredict <- predict(nbmnist,croppedtestf)
correcttepr <- nbpredict == testlabel
mnisttescore <- sum(correcttepr)/(sum(correcttepr) + sum(!correcttepr))
cat("Accuracy of Gaussian Naive Bayes = ", mnisttescore)

#bernoulli

#Convert features values to categorical
trfnew <- data.frame(lapply(croppedtf,factor,levels=c(0,1)))
tefnew <- data.frame(lapply(croppedtestf,factor,levels=c(0,1)))


nbmnistb <- naive_bayes(trfnew,trainlabel,usekernel = FALSE,laplace = 3)
nbpredictb <- predict(nbmnistb,tefnew)
correctteprb <- nbpredictb == testlabel
mnisttescoreb <- sum(correctteprb)/(sum(correctteprb) + sum(!correctteprb))
cat("Accuracy of Bernoulli Naive Bayes = ", mnisttescoreb)


