
Goal: Principle Component Ananlysis

Dataset: CIFAR-10 dataset : https://www.cs.toronto.edu/~kriz/cifar.html, consists of many 32x32 images in 10 catgeories.


HW3Part12.R: 

Part 1: Computed mean image of each category and first 20 principal components of each image. Reconstructed each image using mean image and 20 principal components. Plotted resconstruction error per category. Used SVD decomposition library

Part 2: Computed distance between mean images for every category and  used principal coordinate analysis to plot the distances, using cmdscale library 

HW3Part3.R:

Calculated a similarity measure between two classes.For class A and class B, define E(A → B) to be the average error obtained by representing all the images of class A using the mean of class A and the first 20 principal components of class B. Distance metric between classes is defined as (1/2)(E(A → B) + E(B → A)). Used principal coordinate analysis to make a 2D map of the classes and compared iwth Part2

report.pdf has analysis on results

