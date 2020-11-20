"""
@author: cnoble
"""

import numpy as np
from statistics import mean, mode, StatisticsError 
import Metrics

# function parameters: training dataset, test point, number of neighbors k, classification/regression T/F
class Knn:
    @staticmethod
    def knn(trainSet, testPoint, k, classification):
        # store indices and distances in 2D array
        distances = np.zeros(shape=(len(trainSet), 2))

        # loop through training set to find distances between test point and each training set point
        for i in range(len(trainSet)):    
            # use our own distance metric
            curDist = Metrics.Metrics().euclideanDistance(testPoint, trainSet[i], (len(testPoint) - 1))
            distances[i][0] = i
            distances[i][1] = curDist
   
        # sort by distance and subset to k neighbors' response values
        sortedDist = sorted(distances, key=lambda x: x[1])
        neighbors = np.zeros(k)
        for i in range(k):
            neighbors[i] = trainSet[int(sortedDist[i][0])][-1]
        
        # return predicted class or regression value
        return Knn.predict(neighbors, classification, trainSet, testPoint, k)

    # predict response variable from neighbors
    def predict(neighbors, classification, trainSet, testPoint, k):
        # choose most popular class for classification
        if classification:
            # in case of tie, repeatedly run knn with k-1 until most popular class is found
            try:
                return int(mode(neighbors))
            except StatisticsError:
                return Knn.knn(trainSet, testPoint, k-1, True)

        # find average of neighbors for regression
        else:
            return mean(neighbors)
    def fit(self, trainset, testset, k, classification):
            predicted = []
            for index, x in testset.iterrows():
                    predicted.append(Knn.knn(trainset, x, k, classification))
            return predicted
