# try 1: remove dd, drinking may assemble smoking more, and different from feeding
# try 2: generate fake data, the 10 fold results vary, maybe the data set is too small.
# 

from __future__ import division
import time
import datetime
import csv
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *
from scipy.signal import *
from numpy import *
from beyourself.data.label import read_SYNC
from beyourself.core.util import *
from pylatex import Document, LongTable, MultiColumn

import os
import re
import matplotlib
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from collections import Counter
from sklearn import preprocessing, svm
from sklearn.metrics import matthews_corrcoef
import numpy.polynomial.polynomial as poly
pd.set_option('display.max_rows', 500)
from sklearn import preprocessing
from sklearn.metrics import auc, silhouette_score
from collections import Counter
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score
from datetime import timedelta
from pylatex import Document, LongTable, MultiColumn
from utils import tt_split_pseudo_rand, calc_multi_cm, balanced_subsample, wacc_from_CM

import warnings
warnings.filterwarnings("ignore")

from scipy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold,KFold

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline



def standard_normalizer(x):
    x = x.T
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   
    # create standard normalizer function based on input data statistics
    normalizer = lambda data: ((data.T - x_means)/x_stds).T
    # return normalizer and inverse_normalizer
    return normalizer




fd_file='All/fd/fd_groundtruthsegmentation.csv'
dd_file='All/dd/dd_groundtruthsegmentation.csv'
sd_file='All/sd/sd_groundtruthsegmentation.csv'
cd_file='All/cd/cd_groundtruthsegmentation.csv'
null_file='All/null/null_groundtruthsegmentation.csv'

fd,dd=pd.read_csv(fd_file),pd.read_csv(dd_file)
sd,cd,null=pd.read_csv(sd_file),pd.read_csv(cd_file),pd.read_csv(null_file)

pos=pd.concat([fd,dd])
# neg=pd.concat([sd,cd,null])
neg = null

pos['label']=1
neg['label']=0
posneg = pd.concat([pos,neg])
XY_all = posneg.as_matrix()
# print(XY_all)

# XY_trn, XY_test = tt_split_pseudo_rand(, .7, seed=1)

fsc_list = []

skf = StratifiedKFold(n_splits=10)

first_flag = 1

for XY_trn_idx, XY_test_idx in skf.split(XY_all[:,:-1], XY_all[:,-1]):
    XY_trn = XY_all[XY_trn_idx]
    XY_test = XY_all[XY_test_idx]

    X_trn, Y_trn = XY_trn[:,:-1], XY_trn[:,-1]
    X_test, Y_test = XY_test[:,:-1], XY_test[:,-1]


    # return normalization functions based on input x
    normalizer = standard_normalizer(X_trn)
    # normalize input by subtracting off mean and dividing by standard deviation
    X_trn = normalizer(X_trn)
    X_test = normalizer(X_test)


    # X_trn, Y_trn = balanced_subsample(X_trn,Y_trn,subsample_size=1.0)
    X_trn,Y_trn = SMOTE().fit_sample(X_trn,Y_trn)
    X_trn,Y_trn = ADASYN().fit_sample(X_trn,Y_trn)

    # print( "Shape of training data: " + str(X_trn.shape))
    # print( "Shape of training label: " + str(Y_trn.shape))

    # pos_examples_counter = len(np.where(Y_trn == 1)[0])
    # neg_examples_counter = len(np.where(Y_trn == 0)[0])

    # print( "Positive Examples: " + str(pos_examples_counter))
    # print( "Negative Examples: " + str(neg_examples_counter))

    # clf = ExtraTreesClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_estimators=185)
    # clf = AdaBoostClassifier(n_estimators=185)
    # clf = KNeighborsClassifier(n_neighbors=2)
    # clf = svm.LinearSVC()
    #clf = GaussianNB()
    # clf = DecisionTreeClassifier()
    # clf = LogisticRegression()

    clf.fit(X_trn,Y_trn)

    X_test = XY_test[:,:-1]
    Y_test = XY_test[:,-1]
    predicted = clf.predict(X_test)

    # print("Shape of Predicted: " + str(predicted.shape))

    result, cm = calc_multi_cm(Y_test, predicted)
    # print(Y_test)
    # print(predicted.shape)

    if first_flag:
    	Y_test_all = Y_test
    	predicted_all = predicted
    	first_flag = 0
    else:
    	Y_test_all = np.hstack((Y_test_all, Y_test))
    	predicted_all = np.hstack((predicted_all, predicted))


    print(result['1']['f1'])
    print(cm)
    # fsc_list.append(result['1']['f1'])


# print(np.mean(np.array(fsc_list)))
result, cm = calc_multi_cm(Y_test_all, predicted_all)
print(result['1']['f1'])
print(result)
print(cm)




# print(wacc_from_CM(cm, 2))

