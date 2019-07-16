import os
import sys
import glob 
from sklearn.model_selection import train_test_split

class ReadData:

    def __init__(self, splitsize =0.2):
        self.inplist = []
        self.labels =  []
        self.splitsize = splitsize

    #Read the input files and populate data and labels

    def ReadFiles(self):
        #print("Entering ReadFiles")
        x = 0
        for dirname, subdirname, files in os.walk('HMP_Dataset'):        
            filterd = dirname.split("/")
            #print(dirname)            
            if len(filterd) == 2:
                for fname in files:
                    x = x + 1
                    if fname.endswith('.txt'):                                        
                        with open(os.path.join(dirname,fname), "r") as fn: 
                            inparr =  []                       
                            for line in fn:                            
                                linetemp = line.split()                            
                                inparr.append(float(linetemp[0]))
                                inparr.append(float(linetemp[1]))
                                inparr.append(float(linetemp[2])) 
                        self.inplist.append(inparr)
                        self.labels.append(filterd[1])
                       

    #Keep some data for classifier evaluation
    def SeparateData(self):
        #print("Entering SeparateData")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.inplist, self.labels,stratify=self.labels,test_size=self.splitsize)
        #print("training" , len(self.X_train))
        #print("test" , len(self.X_test))
        