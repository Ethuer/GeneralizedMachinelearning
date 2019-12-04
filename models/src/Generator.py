from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class Generator():
    def __init__(self, csv_location , train_split_ratio = 0.9, verbose = False):
        """ prefetches the data and returns batches of preprocessed samples on getbatch"""
        """ only recommendable for smallish datasets when prefetching is feasible """
        self.csv_location = csv_location
        self.verbose = verbose
        self.train_split_ratio  = train_split_ratio
        self.train, self.val = self._fetch_split()
        self.label_encoder = LabelEncoder()
        self.ohot_encoder = OneHotEncoder(sparse=False)
        self._fit_encoders()
        
        
        
    def specific_column_normalization(self,pandas_dataframe):
        nonbinary_columns = []
        for element in range(294):
            if len(pandas_dataframe[element].unique()) > 2:
                nonbinary_columns.append(element)
        
        for column in nonbinary_columns:
            
            # the better approach could be e.g binning and one hot encoding
            # this just puts the column to 0 at < median
            
            
            mask = pandas_dataframe[column]> pandas_dataframe[column].median()
            pandas_dataframe.loc[mask,column] = 1
            pandas_dataframe.loc[~mask,column] = 0
            
            # alternative 
            #pandas_dataframe[column] = np.round(pandas_dataframe[column] / pandas_dataframe[column].max(),0)
        
        return pandas_dataframe
        
    def _fit_encoders(self):
        self.label_encoder.fit(self.train[295])
        
        self.ohot_encoder.fit(self.label_encoder.transform(self.train[295]).reshape(-1, 1))
        
        if self.verbose:
            print("[STATUS] fitted encoders ohot and label")
        
        
    def _fetch_split(self):
        """ returns train and validation set of the df """
        """ for simplicity sake this is a prefetch, should be linked to a dbc and add a """
        """ preprocessing functionality """
        dataframe = pd.read_csv(self.csv_location , header=None)
        dataframe = self.specific_column_normalization(dataframe)
        mask = np.random.rand(len(dataframe)) < self.train_split_ratio
        
        # split the frame
        return dataframe[mask], dataframe[~mask]

    def getbatch(self, batchsize = 10, training = True):
        if training:
            batch = self.train.sample(batchsize)
        else:
            batch = self.val.sample(batchsize)
        
        # preprosessing potential, 
        return batch.iloc[:,0:295], self.label_encoder.transform(batch[295])

