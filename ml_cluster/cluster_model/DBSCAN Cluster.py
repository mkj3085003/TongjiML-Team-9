import numpy as np
import math

class myDBSCAN :
    # initialize
    def __init__(self, eps, min_samples) :
        self.eps = eps
        self.min_samples = min_samples
        self.visit = []
        self.clusters = []
        self.list_of_cluster = []
        self.noise = []
        self.core = [] # for predict
        self.dataset = []
        
    def fit(self, dataset) : # list
        self.visit = dataset.values.tolist()  # 将 DataFrame 转换为列表
        self.dataset = dataset.values.tolist()
        C = -1 # create cluster
        for data in self.dataset :
            if data in self.visit :
                self.visit.remove(data)
                # find all neighbor for sample data
                data_neighbor = self.find_neighbors(data)
                if len(data_neighbor) < self.min_samples : self.noise.append(data)
                else :
                    C += 1
                    self.expand_cluster(data, data_neighbor, C)
        self.create_cluster()
                    
    def expand_cluster(self, sample, sample_neighbor, C) :
        # first delete clustered element before because it's not core
        for inst in self.clusters :
                if sample in inst : self.clusters[self.clusters.index(inst)].remove(sample)
        self.clusters.insert(C, [sample])
        self.core.append(sample)
        
        for data in sample_neighbor :
            if data in self.visit : # is not visited yet
                self.visit.remove(data)
                data_neighbor = self.find_neighbors(data)
                if len(data_neighbor) >= self.min_samples :
                    self.core.append(data)
                    for elmt in data_neighbor :
                        if elmt not in sample_neighbor : sample_neighbor.append(elmt)
            cluster = False
            for inst in self.clusters :
                if data in inst :
                    cluster = True
                    break
            if cluster == False : 
                self.clusters[C].append(data)
                if data in self.noise : self.noise.remove(data)
    
    def find_neighbors(self, sample) :
        neighbor = []
        for data in self.dataset :
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(sample, data)])) # calculate euclidian distance
            if distance <= self.eps : neighbor.append(data)        
        return neighbor
    
    def create_cluster(self) :
        for data in self.dataset :
            for i in range(len(self.clusters)) :
                if data in self.clusters[i] : 
                    self.list_of_cluster.append(i)
            if data in self.noise : 
                self.list_of_cluster.append(-1)
        self.list_of_cluster = np.array(self.list_of_cluster)
        
    def predict(self, dataset) :
        pred = []
        for data in dataset.values.tolist() :
            appended = False
            for core_ in self.core :
                distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(core_, data)])) # calculate euclidian distance
                #print("distance between ", data, " and ", core_, " is ", distance)
                if distance <= self.eps : 
                    #print("it is in cluster ", self.dataset.index(core_))
                    pred.append(self.list_of_cluster[self.dataset.index(core_)])
                    appended = True
                    break
            if appended == False : 
                pred.append(-1)
        return np.array(pred)