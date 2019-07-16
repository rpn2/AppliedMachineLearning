#Clear environemnt 
rm(list=ls())

#Read data 
datainp<-read.csv('brunhild.txt',sep = "\t",header = TRUE, stringsAsFactors=F)
print(datainp)
datalog <-sapply(datainp,log) 
dflog <- as.data.frame(datalog)
colnames(dflog) <- c("LogHours", "LogSulfate")
print(dflog)

#Compute regression on log data
lmlog = lm(LogSulfate~LogHours, data=dflog, model = TRUE, x = TRUE, y = TRUE)
summary(lmlog)


#Plot regression line on log scale
with(dflog,plot(LogHours, LogSulfate,main="Log Regression Fit"))
abline(lmlog)


#Plot regression on normal scale
pred = predict(lmlog)
with(datainp,plot(Hours, Sulfate,main="Regression Fit"))
lines(t(datainp['Hours']), exp(pred))

#Plot residual on log scale
logres = resid(lmlog)
plot(pred, logres,ylab="Residuals", xlab="Fitted Log Sulphate",main="Residual plot for Log scale") 



#Plot residual in normal scale

newres = t(datainp['Sulfate'] - exp(pred))
plot(exp(pred), newres,ylab="Residuals", xlab="Fitted Sulphate",main="Residual plot for normal scale")


