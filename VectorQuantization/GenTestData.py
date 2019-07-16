import numpy as np
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class GenTestData:

    def __init__(self,Testdata,firstcentroids,secondcentroids,segsize = 32,firstk=10,secondk=5):

        self.segsize = segsize*3
        #Each segment is a list of 96 values, 32 each for x,y,z
        self.segments = []
        self.firstcentroids = firstcentroids
        #Capture second-level centroids, Dict with key as first centroid index and value as list of second centroids
        self.secondcentroids = secondcentroids
        self.X_test = Testdata
        self.firstk = firstk
        self.secondk = secondk
        self.segloc = {}
        #Generate feature list
        self.tefeature = np.zeros(shape=(len(self.X_test),self.firstk*self.secondk), dtype = int)



    def GenClusterData(self):
        #print("Entering GenClusterData in Test")        
        for traincnt in range(len(self.X_test)):            
            eachtrain = self.X_test[traincnt] 
            indexlst = []
            indexlst.append(len(self.segments))
            segtrain = [eachtrain[i * self.segsize:(i + 1) * self.segsize] for i in range((len(eachtrain) + self.segsize - 1) // self.segsize )]           
            #Convert segmented data in interleaved (x,y,z) to x followed by y followed by Z
            #Create data segments. Each segment is of length segsize*3
            for seglist in  segtrain:
                if len(seglist) == self.segsize:
                    arrx = [seglist[x] for x in range(0,self.segsize,3)]
                    arry = [seglist[x] for x in range(1,self.segsize,3)]
                    arrz = [seglist[x] for x in range(2,self.segsize,3)]
                    total = arrx + arry + arrz
                    self.segments.append(total)
                #Happens when last segment is not exactly segsize*3, get last segsize*3 elements from the data. Minor overlap
                '''else:
                    templist = eachtrain[-self.segsize:]
                    arrx = [templist[x] for x in range(0,self.segsize,3)]
                    arry = [templist[x] for x in range(1,self.segsize,3)]
                    arrz = [templist[x] for x in range(2,self.segsize,3)]
                    total = arrx + arry + arrz
                    self.segments.append(total)''' 
            indexlst.append(len(self.segments) - 1)
            self.segloc[traincnt] = indexlst

        #print(self.segloc)
        #print("Total Segments in test =", len(self.segments)) 


    def IdentifyClusters(self): 
        #print("Entering IdentifyClusters")

        self.segmentsarr = normalize(np.array(self.segments))

        #print("Shape of segarray = ", self.segmentsarr.shape)

        #Computer first level clusterid of all points, assigning them to closet cluster center 
        code_index, dist = vq(self.segmentsarr, self.firstcentroids)

        #print("Shape of firstcentrod in test = ", self.firstcentroids.shape)
        #Dictionary to store cluster membership, Key is index of array, value is cluster number
        self.clustermembership = np.zeros(code_index.shape, int)
        
        #For all the segments, compute appropriate second level membership
        
        for k  in range(len(code_index)):
            fcid = code_index[k]
            data = self.segmentsarr[k].reshape((1, self.segsize))
            getsecondcentroids = self.secondcentroids[fcid]
            
            sindex, dist2 = vq(data, getsecondcentroids)
            self.clustermembership[k] = self.secondk*fcid + sindex
            k = k + 1




    def VectorQuantization(self): 
        #print("Entering VectorQuantization in Test")
        for traincnt in range(len(self.X_test)):
            featurevect = np.zeros(self.secondk*self.firstk, int)
            #print("d1 =", featurevect.shape)
            index = self.segloc[traincnt]            
            clusterpersample = self.clustermembership[index[0]:index[1]+1]            

            for element in clusterpersample: 
                featurevect[element] = featurevect[element] + 1

            
            self.tefeature[traincnt] = featurevect
            


        #print(self.trfeature)
        np.savetxt("testdata.csv",  self.tefeature, fmt='%i', delimiter=" ")
















