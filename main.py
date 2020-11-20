# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:09:33 2019

@author: asadc
"""

import LoadDataset as ld
import Knn
import EditedNN
import Metrics
import pandas as pd

class Main:
        def __init__(self):
                #define variable to use inside class which may need tuning
                self.splitlength = 0.75
                self.editedNN_k_value = 3
                self.knn_k_values = [3]
                #load all dataset
                self.alldataset = ld.LoadDataset().load_data()          
                #check dataset is classification
                self.IsClassificationDict = ld.LoadDataset().IsClassificationDict() 
                #define dataframe to store all the results
                self.allresults = pd.DataFrame(columns=['dataset', 'isClassification', 'k', 'method',
                                                        'accuracy', 'precision', 'recall', 'RMSE'])
                
        def main(self):
                for dataset in self.alldataset:         #for each dataset call each algorithm
                        print('\n current dataset ::: {0} \n'.format(dataset))
                        data = self.alldataset.get(dataset)
                        isClassification = self.IsClassificationDict.get(dataset)
                        #k_for_cluster = 0                       #define k value for cluster                        
                        for k in self.knn_k_values:
                                trainset, testset = self.testtrainsplit(data, self.splitlength)
                                pd.DataFrame(trainset).to_csv("reduced_datasets/trainset.csv", header=None, index=None)
                                pd.DataFrame(testset).to_csv("reduced_datasets/testset.csv", header=None, index=None)
                                #call knn
                                predicted, labels = self.knn(trainset, testset, k, isClassification)
                                self.performance_measure(predicted, labels, dataset, isClassification, k, 'KNN')
                                #call edited_nn
                                if isClassification:
                                        k_for_cluster, predicted, labels = self.edited_nn(trainset, testset, k, isClassification)
                                        self.performance_measure(predicted, labels, dataset, isClassification, k, 'E-NN')
                # results
                pd.DataFrame(self.allresults).to_csv("results.csv")
                print('\n printing result\n')
                print(self.allresults)
                                
                return self.allresults
        
        def testtrainsplit(self, data, foldlen):
                data = data.sample(frac=1)                   #randomize the rows to avoid sorted data
                testlen = int(len(data) * foldlen)           #split according to fold lenth
                testset = data[testlen:]
                trainset = data[:testlen]
                return trainset, testset
        
        def knn(self, trainset, testset, k, isClassification):
                print('running knn')
                predicted = Knn.Knn().fit(trainset.values, testset, k, isClassification)
                return predicted, testset.iloc[:, -1]   #return predicted and actual labels
        
        def edited_nn(self, trainset, testset, k, isClassification):
                #get the reduced dataset using edited nn
                print('running enn')
                reduced_dataset = EditedNN.EditedNN().eknn(trainset.values, self.editedNN_k_value)
                #call knn with the reduced train set
                predicted = Knn.Knn().fit(reduced_dataset, testset, k, isClassification)
                return len(reduced_dataset), predicted, testset.iloc[:, -1]   #return predicted and actual labels
        
        def performance_measure(self, predicted, labels, dataset, isClassification, k, method):
                mtrx = Metrics.Metrics()
                if (isClassification):
                        acc, prec, recall = mtrx.confusion_matrix(labels.values, predicted)
                        self.update_result(dataset, isClassification, k, method, acc, prec, recall, 0)
                         
                else:
                        rmse = mtrx.RootMeanSquareError(labels.values, predicted)
                        self.update_result(dataset, isClassification, k, method, 0, 0, 0, rmse)
        
        def update_result(self, dataset, isClassification, k, method, acc, prec, recall, rmse):
                self.allresults = self.allresults.append({'dataset': dataset, 'isClassification': isClassification,
                                                'k': k, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'RMSE': rmse}, ignore_index=True)
        
results = Main().main()
results.to_csv('results.csv')