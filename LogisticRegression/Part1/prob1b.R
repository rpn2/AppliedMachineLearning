#Clear environemnt 
rm(list=ls())
library(MASS)
#Read data 
datainp<-read.csv('default_plus_chromatic_features_1059_tracks.txt',sep = ",",header = FALSE, stringsAsFactors=F)
datafeatures = datainp[,1:116]

#Add constant to remove negative values
datainp$poslat<-datainp[,117] + 90
datainp$poslong<-datainp[,118] + 180



#Latitude Regression and box-cox
lm.latitude = lm(poslat ~ . - V118 - V117 - poslong - poslat ,data=datainp)
trans.latitude = boxcox(lm.latitude, lambda=seq(-10,10, by = 0.1))
trans_df.latitude = as.data.frame(trans.latitude)
optimal_lambda.lat =trans_df.latitude[which.max(trans.latitude$y),1]
print(optimal_lambda.lat)

#Regression against optimnal lambda from boxcox
datainp$tflat <- ((datainp$poslat ^ optimal_lambda.lat) - 1)/optimal_lambda.lat
lm.optlat = lm(tflat ~ . - V118 - V117 - poslong - poslat -tflat,data=datainp)

#Residual in original
 
fitted.lat = ((predict(lm.optlat) * (optimal_lambda.lat)) + 1) ^ (1/optimal_lambda.lat) 
fitted.lat <- fitted.lat - 90
res.lat = t(datainp[,117] - fitted.lat)
plot(fitted.lat, res.lat,ylab="Residuals", xlab="Fitted Latitude",main="Transformed to original co-ordinates") 
#r2lat = var(fitted.lat,na.rm=TRUE)/var(datainp[,117],na.rm=TRUE)
r2lat = var(fitted.lat)/var(datainp[,117])
print(r2lat)

#BIC 
lat.bic = BIC(lm.optlat)
print(lat.bic)


#Longitude Regression and box-cox
lm.longitude = lm(poslong ~ . - V118 - V117 - poslong - poslat - tflat ,data=datainp)
trans.longitude = boxcox(lm.longitude,lambda=seq(-10,10, by = 0.1))
trans_df.longitude = as.data.frame(trans.longitude)
optimal_lambda.long =trans_df.longitude[which.max(trans.longitude$y),1]
print(optimal_lambda.long)

#Regression against optimnal lambda from boxcox
datainp$tflong <- ((datainp$poslong ^ optimal_lambda.long) - 1 )/optimal_lambda.long
lm.optlong = lm(tflong ~ . - V118 - V117 - poslong - poslat - tflat - tflong,data=datainp)

#Residual in original
fitted.long = ((predict(lm.optlong) * (optimal_lambda.long)) + 1 ) ^(1/optimal_lambda.long)
fitted.long <- fitted.long - 180
res.long = t(datainp[,118] - fitted.long)
plot(fitted.long, res.long,ylab="Residuals", xlab="Fitted Longitude",main="Transformed to original co-ordinates") 
r2long = var(fitted.long)/var(datainp[,118])
print(r2long)

#BIC 
long.bic = BIC(lm.optlong)
print(long.bic)