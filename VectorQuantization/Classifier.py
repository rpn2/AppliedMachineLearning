from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

##http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/

class Classifier:

    def __init__(self, traindata, trainlabel, testdata, testlabel):
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testdata = testdata
        self.testlabel = testlabel


    def RandomForest(self):

        #print("Entering RandomForest in Classifier") 
        #Training Phase
        rf = RandomForestClassifier(n_estimators=65)
        rf.fit(self.traindata, self.trainlabel)

        #print("Random Forest Trained\n", rf)

        #Prediction phase
        predictions = rf.predict(self.testdata)

        labels = ['Pour_water','Drink_glass', 'Use_telephone','Descend_stairs','Climb_stairs','Standup_chair','Sitdown_chair','Getup_bed','Liedown_bed','Comb_hair', 'Brush_teeth', 'Eat_meat', 'Eat_soup', 'Walk']


             
        self.trainaccuracy = accuracy_score(self.trainlabel, rf.predict(self.traindata))
        self.testaccuracy = accuracy_score(self.testlabel, predictions)
        self.confusionmatrix = confusion_matrix(self.testlabel, predictions,labels)
        #print(self.testlabel)
     
   
    




