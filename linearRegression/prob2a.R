#Clear environemnt 
rm(list=ls())

#Read data 
datainp<-read.csv('physical.txt',sep = "\t",header = TRUE, stringsAsFactors=F)
print(datainp)

#Apply regression 
lm.mass = lm(Mass~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head, data=datainp)
summary(lm.mass)

fittedval = fitted.values(lm.mass)

res = resid(lm.mass)
plot(fittedval, res,ylab="Residuals", xlab="Fitted Mass",main="Residual plot for Regression") 

