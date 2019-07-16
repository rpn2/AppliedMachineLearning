import numpy as np
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class GenTrainingData:

    def __init__(self,Trdata, segsize = 32, firstk=10,secondk=5,rsamplesize=2000):
        
        self.segsize = segsize*3
        #Each segment is a list of 96 values, 32 each for x,y,z
        self.segments = []
        #K value for first and second level for K means
        self.firstk = firstk
        self.secondk = secondk
        #Sample size for first level of k-means
        self.samplesize = rsamplesize
        #To keep track of segment locations of each sample. Key is index, value is 2-elment list (start,end)
        self.segloc = {}
        self.X_train = Trdata
        #Generate feature list
        self.trfeature = np.zeros(shape=(len(self.X_train),self.firstk*self.secondk), dtype = int)
        #Capture centroids of first-level k
        self.firstcentroids = None
        #Capture second-level centroids, Dict with key as first centroid index and value as list of second centroids
        self.secondcentroids = {}


    def GenClusterData(self):
        #print("Entering GenClusterData")        
        for traincnt in range(len(self.X_train)):            
            eachtrain = self.X_train[traincnt] 
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
        #print("Total Segments=", len(self.segments))

    def GenClusters(self): 
        #print("Entering GenClusters")

        self.segmentsarr = normalize(np.array(self.segments))
        
        #Randomly sample a subset of data
        randomarr = self.segmentsarr[np.random.choice(self.segmentsarr.shape[0], self.samplesize, replace=False)]

        #print("Shape of rarray = ", randomarr.shape)
        
        #First level Clustering
        firstlevel = KMeans(self.firstk, n_init=15)
        firstlabels = firstlevel.fit_predict(randomarr)
        self.firstcentroids = firstlevel.cluster_centers_  

        #print("Shape of firstcentrod = ", self.firstcentroids.shape)

        #Computer first level clusterid of all points, assigning them to closet cluster center 
        code_index, dist = vq(self.segmentsarr, self.firstcentroids)
        
        #Dictionary to store high level cluster membership
        #Key is cluster id, value is indicies of self.segmentsarr
        firstclustermem = {}

        for i in range(self.firstk):
            itemindex = np.where(code_index==i)
            firstclustermem[i] = itemindex

        #Dictionary to store cluster membership, Key is index of array, value is cluster number   

        self.clustermembership = np.zeros(code_index.shape, int)
        
        #Apply second level of clustering
        for i in range(self.firstk):
            #Get all segments in a cluster
            localarr = self.segmentsarr[firstclustermem[i]]            
            secondlevel = KMeans(self.secondk, n_init=15)
            secondlabels = secondlevel.fit_predict(localarr)            
            centroids = secondlevel.cluster_centers_
            #print("Shape of secondcentrod = ", centroids.shape)
            self.secondcentroids[i] = centroids
            k = 0
            for element in np.nditer(firstclustermem[i]):
                self.clustermembership[element] = self.secondk*i + secondlabels[k]
                k = k + 1

        
    def VectorQuantization(self): 
        #print("Entering VectorQuantization")
        for traincnt in range(len(self.X_train)):
            featurevect = np.zeros(self.secondk*self.firstk, int)
            index = self.segloc[traincnt]            
            clusterpersample = self.clustermembership[index[0]:index[1]+1]            

            #Generate Histogram
            for element in clusterpersample: 
                featurevect[element] = featurevect[element] + 1

            #print("Training id", traincnt, featurevect)
            self.trfeature[traincnt] = featurevect
            #print(featurevect)


        #print(self.trfeature)
        np.savetxt("trdata.csv",  self.trfeature, fmt='%i', delimiter=" ")

