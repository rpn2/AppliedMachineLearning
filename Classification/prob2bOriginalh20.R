
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

trainfeatures$label = trainlabel
testfeatures$label = testlabel

#Use randomForest from h2o package
library(h2o)

h2o.init()
#COnvert the dataframe to h20 objects
testfeaturesh2o <- as.h2o(testfeatures, destination_frame="testfeaturesh2o")
trainfeaturesh2o <- as.h2o(trainfeatures, destination_frame="trainfeaturesh2o")

#Loop over number of trees and depth
results = matrix(nrow = 9, ncol = 3)
rowcnt = 1
for (treecntindex in 1:3) {
  for (maxnodesindex in 0:2){
    treecntcalc <- treecntindex*10
    depthcalc <- 4 * ( 2 ^ maxnodesindex)
    #Fit a Random Forest Model
    rfmodel <- h2o.randomForest(y=785,x=1:784,training_frame = trainfeaturesh2o, max_depth=depthcalc,ntrees=treecntcalc)
    #Use the model for test data
    finalRf_predictions<-h2o.predict(object = rfmodel,newdata = testfeaturesh2o)
    #Calculate Average accuracy 
    mnisttescore <- mean(finalRf_predictions$predict==testfeaturesh2o$label)
    cat("\nAccuracy of Random Forest, numtrees, depth = ", mnisttescore,treecntcalc,depthcalc)
    results[rowcnt,1] <- mnisttescore
    results[rowcnt,2] <- treecntcalc
    results[rowcnt,3] <- depthcalc
    rowcnt <- rowcnt + 1
    
  }
}
print(results)
h2o.shutdown(prompt=FALSE)
