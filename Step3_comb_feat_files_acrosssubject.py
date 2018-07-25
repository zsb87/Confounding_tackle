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


for act in focus_act:
    df_concat = []

    for SUBJ in SUBJS:

        folder = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR, 'SAMPLE', 'FEATURE', 'All', act)
        df = pd.read_csv(os.path.join(folder, act+'_groundtruthsegmentation_all_features_lengthover1d8.csv'))
        df_concat.append(df)
    df = pd.concat(df_concat)
    print(len(df))

    folder = os.path.join(OUT_DIR, 'ALL', DEVICE, SENSOR, 'SAMPLE', 'FEATURE', 'All', act)
    create_folder(folder)
    df.to_csv(os.path.join(folder, act+'_groundtruthsegmentation_all_features_lengthover1d8.csv'),index=None)




