rm(list=ls())
start_time <- Sys.time()
library(grid)
#Read the label to image category mapping
labels <- read.table("cifar-10-batches-bin/batches.meta.txt")
#Set constants for image reading and pre-processing
num.images = 10000 
num.files = 6
num.colorf = 32*32
num.features = 32*32*3
num.categories = 10


# Create an 3Darray to get each catgeory: 6000 images in each category
# Dimesion is 6000 rows, 3072 features per category
imagesarr.rgb <-array(0,dim=c(6000,num.features,num.categories))
imagesarr.lab <- array(0,dim=c(6000,1,num.categories))

imagesarr.rgb.center <-array(0,dim=c(6000,num.features,num.categories))

#Keep track of row index of 10 categories 
indexvector <- vector(mode = "numeric", length=num.categories)

#Matrix to hold the sum of each category, 10 rows represent 10 categories
imagessum.rgb <- matrix(0L, nrow =num.categories, ncol = num.features,byrow = TRUE)
#Matrix to count images in each category
imagescnt.rgb <- matrix(0L, nrow =num.categories, ncol = 1,byrow = TRUE)
#Matrix to hold mean of each category
imagesmean.rgb <- matrix(0L, nrow =num.categories, ncol = num.features,byrow = TRUE)


# Cycle through all 6 binary files and capture all images features in every file

for (f in 1:num.files) {
  if (f < num.files) {
    to.read <- file(paste("cifar-10-batches-bin/data_batch_", f, ".bin", sep=""), "rb")
  }
  else {
    to.read <- file("cifar-10-batches-bin/test_batch.bin", "rb")
  }
  for(i in 1:num.images) {
    l <- as.integer(readBin(to.read, integer(), size=1, n=1, endian="big",signed = FALSE))
    r <- as.integer(readBin(to.read, raw(), size=1, n=num.colorf, endian="big",signed = FALSE))
    g <- as.integer(readBin(to.read, raw(), size=1, n=num.colorf, endian="big",signed = FALSE))
    b <- as.integer(readBin(to.read, raw(), size=1, n=num.colorf, endian="big",signed = FALSE))
    index <- num.images * (f-1) + i
    totalimage <- c(r, g, b)
    #images.rgb[index,] <- totalimage
    category = l + 1
    #images.lab[index] <- category
    
    #To calculate sum, populate the sum matrix
    imagessum.rgb[category,] <-imagessum.rgb[category,] + totalimage
    imagescnt.rgb[category,] <- imagescnt.rgb[category,] + 1
    
    #Populate arrays
    indexvector[category] = indexvector[category] + 1
    imagesarr.rgb[indexvector[category],,category] <-totalimage
    imagesarr.lab[indexvector[category],,category] <-category
    
  }
  close(to.read)
  remove(l,r,g,b,index,to.read,totalimage)
}



# function to run sanity check on photos  import
drawImage <- function(index,category) {
  
  
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  
  img <- imagesarr.rgb[index,,category]
  img.r.mat <- matrix(img[1:1024], ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img[1025:2048], ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img[2049:3072], ncol=32, byrow = TRUE)
  img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- c(32,32)
  img.col.mat <- t(img.col.mat)
  grid.raster(t(img.col.mat), interpolate=T)
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  
  
}

#Sample Airplane
drawImage(2,1)
#Sample Automobile
drawImage(1,2)
#Sample Bird
drawImage(1,3)
#Sample Cat
drawImage(2,4)
#Sample Deer
drawImage(4,5)
#Sample Dog
drawImage(2,6)
#Sample Frog
drawImage(2,7)
#Sample Horse
drawImage(1,8)
#Sample Ship
drawImage(5,9)
#Sample truck
drawImage(2,10)

#Find the mean of pixel values of every category
for (i in 1:num.categories) {
  imagesmean.rgb[i, ] <- imagessum.rgb[i, ]/imagescnt.rgb[i]
}


drawMeanImage <- function(index) {
  mfrow=c(1, 1)
  img <- imagesmean.rgb[index,]
  img.r.mat <- matrix(img[1:1024], ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img[1025:2048], ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img[2049:3072], ncol=32, byrow = TRUE)
  img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- c(32,32)
  img.col.mat <- t(img.col.mat)
  grid.raster(img.col.mat, interpolate=TRUE)
  
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  
}
##Get each category
#Chang ethe number from 1 to 10 for each category
dev.off()
drawMeanImage(1)
drawMeanImage(2)
drawMeanImage(3)
drawMeanImage(4)
drawMeanImage(5)
drawMeanImage(6)
drawMeanImage(7)
drawMeanImage(8)
drawMeanImage(9)
drawMeanImage(10)



#Center the data
for ( i in 1: 10) {
  imagesarr.rgb.center[,,i] <- scale(imagesarr.rgb[,,i], center = TRUE, scale = FALSE)
}


library(rARPACK)
#Vector to store reconstruction error
ReconsError <- vector("numeric", 10)

#Calculate  20 principal components using SVD function, reconstruct the image and calcuate reconstruction error
for (i in 1:10){
  
  imagesarr.svd <- svds( imagesarr.rgb.center[,,i], k=20)
  diagd <- diag(imagesarr.svd$d)
  recons <- (imagesarr.svd$u %*% diagd) %*% t(imagesarr.svd$v)
  ErrorMat <- (recons - imagesarr.rgb.center[,,i]) ^ 2 
  ErrorRowSum <- rowSums(ErrorMat)
  ReconsError[i] <- sum(ErrorRowSum)/6000.0
  
}

remove(imagesarr.svd,diagd,recons,ErrorMat,ErrorRowSum)

##Part A plots
par(las=2) # make label text perpendicular to axis
par(mar=c(5,5,4,2)) # increase y-axis margin.
xnames = c('airpl', 'autom', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#BarPLot
xx <- barplot(ReconsError, main="Absolute Error of reconstruction",names.arg =xnames,cex.names=0.6, xlim = c( 0,5e+06 ), horiz = TRUE) 
#xx <- barplot(ReconsError, main="Absolute Error of reconstruction",names.arg =xnames,cex.names=0.6, ylim = c( 0,5e+06 )) 
#barplot(ReconsError, main="Absolute Error of reconstruction") 
text(y = xx, x = ReconsError, label=ceiling(ReconsError), pos = 2, cex = 0.6, col = "red")      

###Part B 
dev.off()

###Distance between mean images
MeanDistance <- dist(imagesmean.rgb, method = "euclidean", diag = TRUE, upper = TRUE)
pcomean <- cmdscale(MeanDistance, k = 2, eig = TRUE)
plot(pcomean$points[,1],pcomean$points[,2],main="PCO", xlab="PCO 1", ylab="PCO 2")
text(pcomean$points[,1],pcomean$points[,2], labels=labels$V1,cex = 0.8, xpd = TRUE, pos = 4)

write.table(as.matrix(signif(MeanDistance,7)), file="Dismatrix.txt", row.names=FALSE, col.names=FALSE)

end_time <- Sys.time()
difference <- difftime(end_time, start_time, units='mins')
cat("\n ",difference)

write.table(MeanDistance, file="Meanmatrix.txt", row.names=FALSE, col.names=FALSE)


