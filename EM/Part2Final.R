#Clear environemnt 
rm(list=ls())

#Set seed, change this seed for section 2
set.seed(50000)


library(analogue)
library(grid)
library(gridExtra)
library(OpenImageR)
library(matrixStats)

num_clusters = 20



img1 <- readImage("smallsunset.jpg")

total_rows = dim(img1)[1] * dim(img1)[2] 

img0_matrix = matrix(0, nrow =total_rows , ncol =3 )



# Combine all pixels of Image 1 into a 2D-matrix 
k = 1
for (x in 1:dim(img1)[1]) {
  for (y in 1:dim(img1)[2] ) {
    img0_matrix[k, ] = img1[x,y,]
    k = k + 1
  }
}




img1_matrix = img0_matrix*255

#Initialze with k-means
initial_iter = kmeans(img1_matrix, num_clusters)
centers.n = initial_iter$centers
assignment.n = initial_iter$cluster

#Pi vector 
pi.n <- vector(mode = "numeric", length=num_clusters)

for (i in 1:num_clusters){
  pi.n[i] = length(which(assignment.n ==i))/nrow(img1_matrix)
}


#Place holder for distance matrix
modDistmatrix = matrix(nrow = nrow(img1_matrix), ncol = num_clusters)
Wijprev = matrix(0, nrow = nrow(img1_matrix), ncol = num_clusters)
num_iter = 0

#Debug
Wijdifftrk <- vector(mode = "numeric", length=50)
stop_iteration = 0

#Used to cut-off EM if convergence is not met, back-up option 
max_iterations = 100

#Run loop until convergence is met. stop_iteration controls convergence
#Num_iterations is a backup for computationally intensive tasks
while((num_iter < max_iterations) && (stop_iteration == 0) ) {
  num_iter = num_iter + 1
  
  # Expectation step

  Distmatrix <- distance(img1_matrix, centers.n, method = "SQeuclidean")
  
  #Subtract closest cluster center to avoid dealing with tiny numbers
  for (k in 1 : nrow(img1_matrix) ) {
    modDistmatrix[k,] =  Distmatrix[k,] - min(Distmatrix[k,])
  }
  
  Fdist <- -0.5*modDistmatrix 
  Fdistmod <- Fdist +  matrix(rep(log(pi.n),each=nrow(Distmatrix)),nrow=nrow(Distmatrix))
  Wijnum <- exp(Fdistmod)
  Wijden <-rowSums(Wijnum)
  Wijcal = Wijnum/Wijden
  
  #Calculate mean of absolute difference between current weight and previous weight
  Wijdiff = mean(round(abs(Wijcal - Wijprev),4))
  Wijdifftrk[num_iter] = round(Wijdiff,4)
  
  #Convergence condition
  if (Wijdifftrk[num_iter] <= 1e-04) {
    stop_iteration = 1
  }
  
  
 
  #Maximization step
  
  centers.n <- (t(t(img1_matrix) %*% Wijcal))/colSums(Wijcal)
  
  pi.n <- colSums(Wijcal)/nrow(img1_matrix)
  
  #track previous weight for convergence
  Wijprev = Wijcal
}

#print(Wijdifftrk)
print(num_iter)
#Assign closest cluster based on posterior probablity distribution
finalassignment <- vector(mode = "numeric", length=nrow(img1_matrix))
for (k in 1 : nrow(img1_matrix))  {
  finalassignment[k] =  which(Wijcal[k,] == max(Wijcal[k,]))
}

reconst_image <- array(0,dim=c(dim(img1)[1],dim(img1)[2],3))

#Generate Average of colors in each cluster
avg_matrix = matrix(0, nrow = num_clusters , ncol =3 )

for (k in 1 : nrow(avg_matrix))  {
  
  if (length(which(finalassignment == k)) > 1) {
    avg_matrix[k,] =  colMeans(img1_matrix[which(finalassignment == k), ])
  }
  else if (length(which(finalassignment == k)) == 1) {
    avg_matrix[k,] =  img1_matrix[which(finalassignment == k), ]
    
  }
  
}

#Replace each pixel by mean of colors in the respective cluster
for (i in 1:dim(img1)[1] ) {
  for (j in 1:dim(img1)[2] ) {
    index = dim(img1)[2] * (i-1) + j
    reconst_image[i,j,] = avg_matrix[finalassignment[index],]
  }
}

#Output IMage
grid.raster(reconst_image/255)
 

#dev.off()


                            