# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:21:36 2019

@author: asadc
"""

import pandas as pd

class LoadDataset:
        def __init__(self):
                self.directory = 'dataset/'
                #self.datafiles = ['abalone.data','car.data','segmentation.data', 'machine.data',
                #                 'forestfires.data', 'winequality-red.csv', 'winequality-white.csv']
                self.datafiles = ['segmentation.data', 'forestfires.data']
                self.alldataset = {}
                
        def load_data(self):
                for files in self.datafiles:       
                        #read each datafiles
                        data = pd.read_csv(self.directory + files)
                        #give filename without extension as dict key for each dataset
                        key = files.split('.')[0]
                        self.alldataset[key] = self.PreprocessingData(key, data)
                return self.alldataset
                
        def PreprocessingData(self, key, data):
                if key == 'abalone':
                        data = data.drop(['Sex'], axis= 1)
                elif key == 'forestfires':
                        data = data.drop(['month', 'day'], axis= 1)
                elif key == 'machine':
                        data = data.drop(['Vendor name', 'Model name', 'ERP'], axis= 1)
                elif key == 'segmentation':
                        data = data.replace({'CLASS': {'BRICKFACE': 0, 'SKY': 1, 'FOLIAGE': 2, 'CEMENT': 3,
                                                       'WINDOW': 4, 'GRASS': 5, 'PATH': 6 }})
                        class_d = data['CLASS']
                        data = data.drop(['CLASS'], axis= 1)
                        data['CLASS'] = class_d
                elif key == 'car':
                        data = data.replace({'low': 0, 'med': 1, 'high': 2, 'vhigh':3, '5more': 5,
                                             'more': 5, 'small': 0, 'big': 2, 'unacc': 0, 'acc': 1, 
                                             'good': 2, 'vgood': 3})
                        data[['doors', 'persons']] = data[['doors', 'persons']].astype(int)
                return data
        
        def IsClassificationDict(self):
                return {'abalone': True, 'car': True, 'segmentation': True, 'machine': False,
                        'forestfires': False, 'winequality-red': False, 'winequality-white': False}                
                
ld = LoadDataset().load_data()

                