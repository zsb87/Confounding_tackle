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
# print(feature_names)


settings = {}
settings["TIMEZONE"] = 'Etc/GMT+6'

SUBJS = ['202','205','209','211','212','215','218']
#  205 Eating, empty
sampling_rate = 16

ANNO_DIR = '/Volumes/Seagate/SHIBO/MD2K/ANNOTATION/'
DATA_DIR = '/Volumes/Seagate/SHIBO/MD2K/RESAMPLE/'
OUT_DIR = '/Volumes/Seagate/SHIBO/MD2K/DATA_LABEL/'
PLOT_DIR = '/Volumes/Seagate/SHIBO/MD2K/PLOT/'

DEVICE = 'RIGHT_WRIST'
SENSOR = 'ACC_GYR'
focus_act = ['fd','dd','cd','sd','null']


for SUBJ in SUBJS:

    if SUBJ == '212':
        ACTIVITIES = ['Smoking_Eating','False']
        fullname = {'Smoking_Eating':'inlab_eating_smoking', \
                    'False':'inlab_false_alarm'
                   }
        # activity_dict = {'Smoking_Eating': ['sync','fd','fn','ft','dd','dn','dt','fod','fon','sd','sn','st','ld','ln','cn','cd','ct'],\
        #                  'False': ['sync','cn','cd','ct']
        #                 }

        # activity_dict = {'Smoking_Eating': ['null'],\
        #                  'False': ['null']
        #                 }
    else:  
        ACTIVITIES = ['False','Eating','Smoking']
        fullname = {'Smoking':'inlab_smoking', \
                    'False':'inlab_false_alarm', \
                    'Eating':'inlab_eating'
                   }
        # activity_dict = {'Smoking': ['sd','sync','sn','st','ld','ln','cn','cd','ct'],\
        #                  'False': ['cd','cn','sync','ct'], \
        #                  'Eating':['fd','sync','fn','ft','dd','dn','dt','fod','fon','cn','cd','ct']
        #                 }
        # activity_dict = {'Smoking': ['null'],\
        #                  'False': ['null'], \
        #                  'Eating':['null']
        #                 }

    LABEL_FOLDER = os.path.join(ANNO_DIR, SUBJ)
    DATA_FOLDER = os.path.join(DATA_DIR, SUBJ, DEVICE, SENSOR, 'DATA')
    OUT_FOLDER = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR)
    PLOT_FOLDER = os.path.join(PLOT_DIR, SUBJ, DEVICE, SENSOR)    

    df_concat = []

    for act in focus_act:

        for ACTIVITY in ACTIVITIES:
            print(SUBJ)
            print(ACTIVITY)

            SAMPLE_DATA_FOLDER = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR, 'SAMPLE', 'RAW', ACTIVITY)
            SAMPLE_FEAT_FOLDER = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR, 'SAMPLE', 'FEATURE', ACTIVITY)

            if os.path.exists(os.path.join(SAMPLE_FEAT_FOLDER, act)):
                file = os.path.join(SAMPLE_FEAT_FOLDER, act, act+'_groundtruthsegmentation_all_features_lengthover1d8.csv')
                df = pd.read_csv(file, names=feature_names)
                df_concat.append(df)

        df = pd.concat(df_concat)
        folder = os.path.join(os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR, 'SAMPLE', 'FEATURE', 'All', act))
        create_folder(folder)
        df.to_csv(os.path.join(folder, act+'_groundtruthsegmentation_all_features_lengthover1d8.csv'),index=None)




