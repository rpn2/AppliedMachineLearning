
Part1:

Dataset : default_plus_chromatic_features_1059_tracks.txt from https://sites.google.com/site/icdm2014music/

Dataset has 116 features and geographic location as (latitude, longitude).

prob1.R: Linear regression to predict latitude and longtiude against features using lm()

prob1b.R : Apply boxcox transformation to the dependent variables latitude and longitude, and choose best parameter

prob1c_lat.R and prob1c_long.R : Apply regularization and compare cross-validated MSE for ridge, lasso and elastic regression (with different alphas). cv.glmnet() was used




Part2:

Dataset : https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

There are 24 attributes and logistic regression (with and without regularization) was used to predict if a customer would default payment.

10-fold cross-validation is used to get the prediction

Part2.R : Logictic regression with cross-validation

Part2elasX.R : Elastic logistic regression with multiple alphas

Part2lasso.R : Lasso logistic regression 

Part3ridge.r: Ridge logictic regression


Analysis and Results:

report.pdf has analysis on results 