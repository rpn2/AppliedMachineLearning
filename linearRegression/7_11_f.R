library(glmnet)
setwd('/home/max/Desktop/cs498aml/hw5/HW5_final/')
data = read.csv('abalone.data.txt', stringsAsFactors=FALSE)

data[data == 'M'] = 1
data[data == 'F'] = 0
data[data == 'I'] = -1
age = data[, ncol(data)]
gender = data[, 1]
logAge = log10(age)

# (a)
training = data[, 2:(ncol(data) - 1)]
fit = cv.glmnet(as.matrix(training), age,alpha=0)
png(file = "711f-a.png") 
plot(fit, main="error for (a)")

# (b)
training = data[, 1:(ncol(data) - 1)]
fit = cv.glmnet(data.matrix(training), age,alpha=0)
png(file = "711f-b.png") 

plot(fit, main="error for (b)")

# (c)
training = data[, 2:(ncol(data) - 1)]
fit = cv.glmnet(data.matrix(training), logAge,alpha=0)
png(file = "711f-c.png") 

plot(fit, main="error for (c)")

# (d)
training = data[, 1:(ncol(data) - 1)]
fit = cv.glmnet(data.matrix(training), logAge,alpha=0)
png(file = "711f-d.png") 

plot(fit, main="error for (d)")
