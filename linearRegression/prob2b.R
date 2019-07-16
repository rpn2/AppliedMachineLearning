#Clear environemnt 
rm(list=ls())

#Read data 
datainp<-read.csv('physical.txt',sep = "\t",header = TRUE, stringsAsFactors=F)
print(datainp)

#Regression
lm.mass = lm(((Mass)^(1/3))~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head,data=datainp)
summary(lm.mass)

fittedval = predict(lm.mass)

#Cube root residual
res = resid(lm.mass)
plot(fittedval, res,ylab="Residuals", xlab="Fitted Cuberoot of Mass",main="Residual plot for Regression") 

#Transform to original 
newres = t( datainp['Mass'] - (fittedval)^(3))

plot((fittedval)^(3), newres,ylab="Residuals", xlab="Mass",main="Transformed to original co-ordinates") 


