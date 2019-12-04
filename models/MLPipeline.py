import sys
from time import time

import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
# unweighted mean of predicitons
from sklearn.metrics import precision_score
# try ROC_AUC as it combines recall with acc. additional information is necessary to decide whether e.g precision is more important.
# due to the varying scales , a scaler is necessary  Standardscaler (Min-max scaling aka normalization)
# there are alternatives, standardization, 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from utils.Generator import Generator

class Classical_ML_benchmark():
    def __init__(self, 
                 csv_file_location,
                 x_validate = True,
                 verbose = False,
                 versus='none', 
                 encode_labels=False, 
                 learning_rate = 0.1, 
                 max_depth = 3,
                 max_featues = 'none', 
                 bootstrap = False,
                 n_estimators_GB = 100,
                 n_estimators_ET_RF = 10,
                 n_estimators_ADA = 50,
                 C = 1.0,
                 tol = 0.0001,
                 max_iter = 1000,
                 alpha = 1.0,
                 eta0 = 1.0,
                 alpha_perc = 0.0001,
                 leaf_size = 30,
                 n_neighbors = 10,
                 min_impurity_decrease = 0.0):
        """
        
        set Hyperparameters for various clfs
        versus 'ovo' or 'none'
        
        Hyperparameters for CLFs
        """
        # a part could be passed on by inheritance between the ML DL pipeline,
        self.data = Generator(csv_file_location)
        self.x_validate = x_validate
        self.verbose = verbose
        
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_featues = max_featues
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.n_estimators_GB = n_estimators_GB
        self.n_estimators_ET_RF = n_estimators_ET_RF
        self.n_estimators_ADA = n_estimators_ADA
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.eta0 = eta0
        self.alpha_perc = alpha_perc
        self.leaf_size = leaf_size
        self.n_neighbors = n_neighbors
        self.versus = versus
        self.encode_labels = encode_labels
        self.classifiers = {}
        self.label_encoder = LabelEncoder()
        self.ohot_encoder = OneHotEncoder()
        
        self.populate_classes()
        
        
    def _encode_labels(self, labels):
        self.label_encoder.fit(labels)
    
    def _ohot_encode_labels(self,labels):
        self.ohot_encoder.fit(labels)
        
    def populate_classes(self):
        
        self.models_to_use_limited = (
        (RandomForestClassifier(n_estimators=self.n_estimators_ET_RF, bootstrap=self.bootstrap), "Randomforest"),
        (NearestCentroid(),"NearestCentroid"),
        )
        
        
        # make this more flexible about adding specific classifiers
        self.models_to_use = (
            
            (Lasso(alpha=self.alpha, max_iter=self.max_iter, tol = self.tol),"Lasso"),  # regressor
            
            
            # Rf
            (RandomForestClassifier(n_estimators=self.n_estimators_ET_RF,max_features='sqrt' ,
                                    bootstrap=self.bootstrap, oob_score=False), "Randomforest_sqrt"),
            
            (RandomForestClassifier(n_estimators=self.n_estimators_ET_RF,max_features='log2', 
                                    bootstrap=self.bootstrap, oob_score=False), "Randomforest_log2"),
            
            
            (RandomForestClassifier(n_estimators=self.n_estimators_ET_RF,max_features=None,
                                    bootstrap=self.bootstrap, oob_score=False), "Randomforest"),
            
            (BaggingClassifier(n_estimators=self.n_estimators_ET_RF, oob_score=False), "Bagging Tree Classifier"),
            
            # boosted trees
            (GradientBoostingClassifier(n_estimators= self.n_estimators_GB,
                                        learning_rate=self.learning_rate, 
                                        max_depth= self.max_depth, 
                                        min_impurity_decrease=self.min_impurity_decrease),"GradientBoosting"),
            
            (ExtraTreesClassifier(max_depth=self.max_depth,
                                  min_impurity_decrease=self.min_impurity_decrease,
                                  bootstrap= self.bootstrap, 
                                  n_estimators=self.n_estimators_ET_RF),"ExtraTreesClassifier"),
            (AdaBoostClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators_ADA),"AdaBoostClassifier"),
            
            
            # SVM 
            (LinearSVC(C=self.C, tol = self.tol),"LinearSVC"),
            
            
            
            # Passsive Agressive Class. http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
            (PassiveAggressiveClassifier(max_iter=self.max_iter, C=self.C), "Passive-Aggressive "), # hinge > squarehinge
            
            
            # ridgeclassifier does not converge
            #(RidgeClassifier(tol=self.tol, alpha=self.alpha,solver="sag"), "RidgeClassifier"),
            
            
            (Perceptron(alpha=self.alpha, eta0=self.eta0, max_iter=self.max_iter), "Perceptron"),
            
            # Nearest Centroid and K nearest neighbours 
            (NearestCentroid(),"NearestCentroid"),
            (KNeighborsClassifier(n_neighbors=self.n_neighbors,leaf_size=self.leaf_size ), "KNearestNeighbours"),
            
            # a binary (Bernoulli) based Naive Bayes, useful as most features are binary
            (BernoulliNB(binarize=False, fit_prior=True, alpha=self.alpha),"BernoulliNB"),

            )
        
    def fit_all_classifiers(self):
        
        if self.x_validate & self.verbose:
            print("[CAUTION] cross validation takes time")
        
        X_train, y = self.data.getbatch(batchsize=len(self.data.train))
        
        if self.encode_labels:
            _encode_labels(y)
            
            y = self.label_encoder.transform(y)
        
        
        for clf, name in self.models_to_use:

            #if self.versus == 'ovo':
            #    clf_pipeline = Pipeline([
            #        #('std_scaler',StandardScaler() ),
            #        ('OVO_%s' %name, OneVsOneClassifier(clf))  # class by class comparison
            #    ])
            #    
           # else:
            #     # there are no missing values in this data, but an imputer is impurtant.
            #    clf_pipeline = Pipeline([
            #        ('imputer', Imputer(strategy="median") ),
            #        #('std_scaler',StandardScaler() ), #  scalers are not doing so well with binary data
            #        (name,clf )
            #    ])
            
            clf_pipeline = clf
            
            if self.verbose:
                print('[STATUS] Fitting %s ' %name)
                    
            # fitting
            clf_pipeline.fit(X=X_train,y=y)
            
            # RMSE evaluation via cross validation
            if self.x_validate:
                cvs = cross_val_score(clf_pipeline,X_train,y,scoring='neg_mean_squared_error', cv=3)
                rmse_score = np.sqrt(-cvs)
            else:
                rmse_score = 0.0
            # storing the classifiers
            self.classifiers[name] = [clf_pipeline, np.mean(rmse_score) ]
                
        #return classifiers
    
    def _cw_roc_auc(self,y_true,y_pred):
        
        
        #data.ohot_encoder(y_true.reshape(-1, 1))
        
        
        # transform the predictions and labels to one hot encoding
        y_true = self.data.ohot_encoder.transform(y_true.reshape(-1, 1))
        y_pred = self.data.ohot_encoder.transform(y_pred.reshape(-1, 1))
        
        return np.mean([accuracy_score(y_true= np.transpose(y_true)[count],y_pred=col ) 
                        for count,col in enumerate(np.transpose(y_pred)) ])

    
    def report(self ):
        """ Predict the measure of accuracy """
        X_val, y_val = self.data.getbatch(batchsize=len(self.data.val))
        print("Class", "RMSE", " ROC AUC Score", "Accuracy","Recall")
        
        for name, clf in self.classifiers.items():
            try:
                y_pred = clf[0].predict(X_val)
                print( name, np.round(clf[1],3) , np.round(self._cw_roc_auc(y_val,y_pred ) ,3), 
                      np.round(accuracy_score(y_true=y_val,y_pred=y_pred),3 ), 
                      np.round(recall_score(y_true=y_val, y_pred=y_pred, average='macro') , 3) ) # micro average for label imbalance
            except:
                pass
                
                
    def predict(self,classifier,X_dat):
        """ Returns predictions for specific classifier"""
        return self.classifiers[classifier][0].predict(X_dat)
        
        
