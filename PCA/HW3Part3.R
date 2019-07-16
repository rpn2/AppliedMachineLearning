rm(list=ls())

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


#Find the mean of pixel values of every category
for (i in 1:num.categories) {
  imagesmean.rgb[i, ] <- imagessum.rgb[i, ]/imagescnt.rgb[i]
}

#Center the data
for ( i in 1: 10) {
  imagesarr.rgb.center[,,i] <- scale(imagesarr.rgb[,,i], center = TRUE, scale = FALSE)
}


library(rARPACK)

start_time <- Sys.time()
#Matrix for capturing errors
ErrorMat <- matrix(0,nrow=10,ncol=10)

#Calculate  20 principal components using SVD function, reconstruct the image and calcuate reconstruction error
for (catg in 1:10){
  imagesarr.svd <- svds( imagesarr.rgb.center[,,catg], k=20)
  for (crossg in 1:10){
    ErrorRowSum = 0 
    for ( i in 1:6000) {
      sumvector <- vector(mode = "numeric", length=3072)
      for (j in 1:20) {
        sumvector <- sumvector + drop((matrix(imagesarr.svd$v[, j], nrow =1 ) %*% matrix(imagesarr.rgb.center[i,,crossg], nrow = 3072))) * imagesarr.svd$v[, j] 
        
      } #j ends
      Error <- ((sumvector + imagesmean.rgb[crossg, ]) - imagesarr.rgb[i,,crossg]) ^ 2 
      ErrorRowSum <- ErrorRowSum + sum(Error)
      
    } # iends
    ErrorMat[catg,crossg] <- ErrorRowSum/6000
  }#crossg ends
} #catg ends

remove(imagesarr.svd,sumvector,Error,ErrorRowSum)
end_time <- Sys.time()
diff <- difftime(end_time, start_time, units='mins')
cat("\n Time taken =", diff)

save(ErrorMat, file = "ErrorMatrix.RData")
load(file = "ErrorMatrix.RData")

#Calculate similarity matrix : 1/2(E(A/B) + E(B/A))

SimMat <- matrix(0,nrow=10,ncol=10)

for ( i in 1:10){
  for (j in 1:10){
    SimMat[i,j] = (ErrorMat[i,j] + ErrorMat[j,i])/2
    
  }
}
dev.off()
#Plot the similarity matrix in 2D
pcosim = cmdscale(SimMat, k = 2, eig = TRUE)
plot(pcosim$points[,1],pcosim$points[,2],main="PCO", xlab="PCO 1", ylab="PCO 2")
text(pcosim$points[,1],pcosim$points[,2],labels=labels$V1,cex = 0.6, xpd = TRUE, pos = 3) 

write.table(signif(SimMat,7), file="Simmatrix.txt", row.names=FALSE, col.names=FALSE)


