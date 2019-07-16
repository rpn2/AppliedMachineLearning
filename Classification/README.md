Problem 1:

Dataset : https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

Task : Classification to check whether a patient is diabetic or not using set of given attributes  
 
prob1a.R : Naive Bayes in R without using libraries, normal distribution was used to model class conditional distributions. Classification accuracy by averaging over 10 test-train splits.

prob1b.R : Same as prob1a.R, with attribute value of "0" considered as missing value for dimensions [3 (Diastolic blood pressure), attribute 4 (Triceps skinfold thickness), attribute 6 (Body mass index), and attribute 8 (Age)]. Classification accuracy by averaging over 10 test-train splits.

prob1c.R : Naive Bayes using inbuilt methods in R with cross-validation

prob1d.R : SVM classification using svm package




Problem 2:

MNIST digit classification : http://yann.lecun.com/exdb/mnist/ 

Additional useful links: 

https://stackoverflow.com/questions/21521571/how-to-read-mnist-database-in-r

https://stackoverflow.com/a/21524980


prob2aOriginal.R:  

Each class of a dataset was modeled using a)Normal Distribution and b)Bernoulli distribution for original images in MNIST dataset and Accuracy was calculated for both the cases


prob2aCropped.R:

The original images in MNIST were cropped to as smaller bounding box (20x20) and each image was rescaled to fill the bounding box. Repeated digit classification by modeling the dataset as a)Normal Distribution b)Bernoulli distribution


prob2bOriginal.R: 
 
Classify MNIST using random forests by varying number of trees (10,30) and depth (4,16)

prob2bCropped.R:

Repeat random forests on cropped and stretched images

 
 Additional resulst are in Report.pdf
