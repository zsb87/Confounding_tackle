from __future__ import division
import time
import datetime
import csv
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
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
from sklearn import preprocessing
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
import warnings
warnings.filterwarnings("ignore")
from utils import list_files_in_directory, create_folder, calc_fft




# input para: input_data , intervals_of_interest , timeString 
def genFeatsNames( feat_list , sensor_list, extension ):
    header = []
    
    for feat in feat_list:
        for key in sensor_list:
            one = key + "_" + feat
            header.extend([one])
    header.extend(extension)

    return header




feat_list = ["mean","var","skew","kurtosis","RMS","median","max","min","qurt3",'quart1','IRQ']
sensor_list = ['accX','accY','accZ','rotX','rotY','rotZ']
extension = ["FFTaccx0","FFTaccx1","FFTaccx2","FFTaccx3",\
            "FFTaccx4","FFTaccx5","FFTaccx6","FFTaccx7",\
            "FFTaccy0","FFTaccy1","FFTaccy2","FFTaccy3",\
            "FFTaccy4","FFTaccy5","FFTaccy6","FFTaccy7",\
            "FFTaccz0","FFTaccz1","FFTaccz2","FFTaccz3",\
            "FFTaccz4","FFTaccz5","FFTaccz6","FFTaccz7",\
            "ENGYX","ENGYY","ENGYZ","ENGY",\
            "cov_accXY","cov_accYZ","cov_accXZ",\
            "cov_rotXY","cov_rotYZ","cov_rotXZ",\
            "cov_accXrotX","cov_accYrotY","cov_accZrotZ"]



feature_names = genFeatsNames(feat_list, sensor_list, extension)
print(len(feature_names))


