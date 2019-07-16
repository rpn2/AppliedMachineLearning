


import GenTrainingData
import GenTestData
import ReadData
import Classifier



train_acc = []
test_acc = []
cm = []

for i in range(10):
        segsamplesize = 8
        firstk = 20
        secondk = 6
        numrdsamples = firstk * 150  
        testtrainsplit = 0.2


        ##Read Files
        rd = ReadData.ReadData(testtrainsplit)
        rd.ReadFiles()
        rd.SeparateData()

        ##Generate Training data
        #Usage Argument : Training data, segment size, Outer k means, Inner K means, Random sample size for Outer K means
        trvq = GenTrainingData.GenTrainingData(rd.X_train,segsamplesize,firstk,secondk,numrdsamples)
        trvq.GenClusterData()
        trvq.GenClusters()
        trvq.VectorQuantization()

        ##Generate Training data
        #Usage Argument : Training data, firstcentroids, secondcentroids, segment size, Outer k means, Inner K means

        tevq = GenTestData.GenTestData(rd.X_test,trvq.firstcentroids, trvq.secondcentroids, segsamplesize,firstk,secondk)
        tevq.GenClusterData()
        tevq.IdentifyClusters()
        tevq.VectorQuantization()


        #Invoke classifer 
        #Usage Argument : Training data feature, training label, test data feature, test label
        rfclassifier = Classifier.Classifier(trvq.trfeature,rd.y_train, tevq.tefeature,rd.y_test)
        rfclassifier.RandomForest()
        
        train_acc.append(rfclassifier.trainaccuracy)
        test_acc.append(rfclassifier.testaccuracy)
        cm.append(rfclassifier.confusionmatrix)

avg = float(sum(test_acc))/len(test_acc)
print("Average Accuracy", avg)
print("Average Errorrate", 1-avg)
print(test_acc)
print(train_acc)
print(cm)
        
