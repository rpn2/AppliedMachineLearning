#Clear environemnt 
rm(list=ls())

#Set seed, change this seed for section 2
set.seed(50)

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  
  train <<- load_image_file('train-images-idx3-ubyte')
   
}
#Get MNIST data 
load_mnist()

#Create a matrix for 20 digits
imagesarr <-array(0,dim=c(28,28,20))
image_scale <-array(0,dim=c(28,28,20))

for (f in 1:20) {
  imagesarr[,,f] = matrix(train$x[f,],nrow=28)[,28:1]
}
#Remove total data 
remove(train)

#Debug, check random data
image(imagesarr[,,1],col=gray(12:1/12))
dev.off()

for (f in 1:20) {
  image_scale[,,f] = t(imagesarr[,28:1,f])
}

#Get binarized image
image_scale[image_scale < 128] = -1
image_scale[image_scale >= 128] = 1
#Debug, check random data
image(t(image_scale[28:1,,1]),col=gray(12:1/12))
dev.off()

###Add Noise
noise_location <- as.matrix(read.csv(file="NoiseCoordinates.csv", header=TRUE, sep=","))

imagesarrflip <-image_scale
img_index = 1
for (f in seq(1, 40, 2)) {
  
for (i in 2:16) {
    row = as.numeric(noise_location[f,i]) + 1
    col = as.numeric(noise_location[f+1,i]) + 1
    imagesarrflip[row,col,img_index] = imagesarrflip[row,col,img_index] * -1
  }
  img_index = img_index + 1
}
#Debug, check random data
image(t(imagesarrflip[28:1,,1]),col=gray(12:1/12))
dev.off()
### Read order

read_order <- as.matrix(read.csv(file="UpdateOrderCoordinates.csv", header=TRUE, sep=","))
img_index = 1
row_order = matrix(0, nrow= 20, ncol =784 )
clm_order = matrix(0, nrow= 20, ncol =784 )
for (f in seq(1, 40, 2)) {
  for (i in 2:785) {
    row_order[img_index,i-1] = as.numeric(read_order[f,i]) + 1
    clm_order[img_index,i-1] = as.numeric(read_order[f+1,i]) + 1
  }
  img_index = img_index + 1
  
}

###Parameters
init_q <- as.matrix(read.csv(file="InitialParametersModel.csv", header=FALSE, sep=","))

qval <- array(0,dim=c(28,28,20))

for (f in 1:20) {
  qval[,,f] = init_q
}

#Initial Energy calculation
Energy = matrix(0, nrow= 20, ncol =11)
epsilon = 1e-10
EQtemp = qval*log(qval+epsilon) +(1-qval)*log((1-qval)+epsilon) 
thetah = 0.8
thetax = 2
qvalh = 2*qval - 1
qvalhsum  <- array(0,dim=c(28,28,20))
for (f in 1:20) {
      z = qvalh[,,f]
      k = rbind(z[-1,],0) + rbind(0,z[-nrow(z),]) + cbind(z[,-1],0) + cbind(0,z[,-ncol(z)])
      qvalhsum[,,f] = k
}

Elogph_part1 = 0.8 * qvalhsum*qvalh
Elogph_part2 = 2 * qvalh * imagesarrflip

for (f in 1:20) {
  Energy[f,1]= sum(EQtemp[,,f]) - (sum(Elogph_part1[,,f]) + sum(Elogph_part2[,,f]) )
}

##Denoise logic, 10 iterations
for (iter in 1: 10) {
  for (imagenum in 1:20){
    for(pixel in 1:784){
      row = row_order[imagenum,pixel]
      clm = clm_order[imagenum,pixel]
      sumofH = 0
      if (row > 1){
        sumofH = sumofH + (2*qval[row-1,clm,imagenum] - 1)
      }
      if (row < 28){
        sumofH = sumofH + (2*qval[row+1,clm,imagenum] - 1)
      }
      if (clm > 1){
        sumofH = sumofH + (2*qval[row,clm-1,imagenum] - 1)
      }
      if(clm < 28){
        sumofH = sumofH + (2*qval[row,clm+1,imagenum] - 1)
      }
      sumofH = sumofH * 0.8
      sumofX = 2*imagesarrflip[row,clm,imagenum]
      Numofpi = sumofH + sumofX
      Denofpi = -(sumofH + sumofX)
      pi = exp(Numofpi)/(exp(Numofpi) + exp(Denofpi))
      #update Q
      qval[row,clm,imagenum] = pi
    }##End of all pixels
  }## All images

  ##Calculate Energy at every iteration
  qvalh = 2*qval - 1
  EQtemp = qval*log(qval+epsilon) +(1-qval)*log((1-qval)+epsilon) 
  for (f in 1:20) {
    z = qvalh[,,f]
    k = rbind(z[-1,],0) + rbind(0,z[-nrow(z),]) + cbind(z[,-1],0) + cbind(0,z[,-ncol(z)])
    qvalhsum[,,f] = k
  }
  
  Elogph_part1 = 0.8 * qvalhsum*qvalh
  Elogph_part2 = 2 * qvalh * imagesarrflip
  
  for (f in 1:20) {
    Energy[f,iter+1]= sum(EQtemp[,,f]) - (sum(Elogph_part1[,,f]) + sum(Elogph_part2[,,f]) )
  }
}

#Result for auto-grader
write.table(Energy[11:12,1:2], "energy.csv", sep=",", col.names=FALSE, row.names=FALSE)
recons_image = qval

#Get denoised image
recons_image[recons_image < 0.5] = 0
recons_image[recons_image >= 0.5] = 1

#debug
image(t(recons_image[28:1,,10]),col=gray(12:1/12))
dev.off()

#Test if results match sampleDenoised
sample_image <- as.matrix(read.csv(file="SampleDenoised.csv", header=FALSE, sep=","))
final_image_test = matrix(0,nrow = 28,ncol=280)
for (f in 1:10) {
  start = 28*(f-1) + 1
  end =  28*f
  final_image_test[,start:end] = recons_image[,,f]
}

result = sample_image == final_image_test

fail_index = which(result == FALSE)
print(fail_index)

final_image = matrix(0,nrow = 28,ncol=280)
for (f in 1:10) {
  start = 28*(f-1) + 1
  end =  28*f
  final_image[,start:end] = recons_image[,,f+10]
}

#Result for auto-grader
write.table( final_image,"GenDenoisedSamplesNew.csv", sep=",",  col.names=FALSE, row.names=FALSE)


