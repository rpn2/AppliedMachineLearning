#Clear environemnt 
rm(list=ls())

#Read data 
datainp<-read.csv('default_plus_chromatic_features_1059_tracks.txt',sep = ",",header = FALSE, stringsAsFactors=F)
datafeatures = datainp[,1:116]

#Linear regression with latitude

#Latitude Regression and R2 calculation
lm.latitude = lm(V117 ~ . - V118 - V117 ,data=datainp)
latr2 = summary(lm.latitude)$r.squared 
print("Latitude R2")
print(latr2)

fitted.latitude = predict(lm.latitude)

#Latitude residual
res.latitude = resid(lm.latitude)
plot(fitted.latitude, res.latitude,ylab="Residuals", xlab="Fitted Latitude",main="Residual plot for Latitude") 

#BIC of latitude
lat.bic = BIC(lm.latitude)
print(lat.bic)

#Longitude Regression nad R2 calculation
lm.longitude = lm(V118 ~ . - V118 - V117 ,data=datainp)
longr2 = summary(lm.longitude)$r.squared  
print("Longitude R2") 
print(longr2)

fitted.longitude = predict(lm.longitude)

#Latitude residual
res.longitude = resid(lm.longitude)
plot(fitted.longitude, res.longitude,ylab="Residuals", xlab="Fitted longitude",main="Residual plot for longitude") 

#BIC of latitude
long.bic = BIC(lm.longitude)
print(long.bic)
