# ---------------------------------------------------------------------------------------------
# DICTIONARY:
# null:0
# feeding_drink:1
# smoking: 2
# confounding: 4
# ---------------------------------------------------------------------------------------------


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
import sys
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


for SUBJ in SUBJS:

    if SUBJ == '212':
        ACTIVITIES = ['Smoking_Eating','False']
        fullname = {'Smoking_Eating':'inlab_eating_smoking', \
                    'False':'inlab_false_alarm'
                   }
        activity_dict = {'Smoking_Eating': ['sync','fd','fn','ft','dd','dn','dt','fod','fon','sd','sn','st','ld','ln','cn','cd','ct','null'],\
                         'False': ['sync','cn','cd','ct','null']
                        }

        # activity_dict = {'Smoking_Eating': ['null'],\
        #                  'False': ['null']
        #                 }
    else:  
        ACTIVITIES = ['False','Eating','Smoking']
        fullname = {'Smoking':'inlab_smoking', \
                    'False':'inlab_false_alarm', \
                    'Eating':'inlab_eating'
                   }
        activity_dict = {'Smoking': ['sd','sync','sn','st','ld','ln','cn','cd','ct','null'],\
                         'False': ['cd','cn','sync','ct','null'], \
                         'Eating':['fd','sync','fn','ft','dd','dn','dt','fod','fon','cn','cd','ct','null']
                        }
        # activity_dict = {'Smoking': ['null'],\
        #                  'False': ['null'], \
        #                  'Eating':['null']
        #                 }

    LABEL_FOLDER = os.path.join(ANNO_DIR, SUBJ)
    DATA_FOLDER = os.path.join(DATA_DIR, SUBJ, DEVICE, SENSOR, 'DATA')
    OUT_FOLDER = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR)
    PLOT_FOLDER = os.path.join(PLOT_DIR, SUBJ, DEVICE, SENSOR)    



    for ACTIVITY in ACTIVITIES:
        print(SUBJ)
        print(ACTIVITY)

        SAMPLE_DATA_FOLDER = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR, 'SAMPLE', 'RAW', ACTIVITY)
        SAMPLE_FEAT_FOLDER = os.path.join(OUT_DIR, SUBJ, DEVICE, SENSOR, 'SAMPLE', 'FEATURE', ACTIVITY)

        for act in activity_dict[ACTIVITY]:
            folder = os.path.join(SAMPLE_DATA_FOLDER, act)
            files = list_files_in_directory(folder)
            start_flg = 0

            for file in files:

                if file.startswith(act):
                    print(file)
                    data_df = pd.read_csv(os.path.join(folder, file))

                    if len(data_df) > 36:
                        X_win = data_df.as_matrix()

                        """
                        overall

                        """

                        MEAN = mean(X_win,axis=0)
                        VAR = var(X_win,axis=0)
                        SK = stats.skew(X_win,axis=0)
                        KURT = stats.kurtosis(X_win,axis=0)
                        RMS = sqrt(mean(X_win**2,axis=0))
                        MED = median(X_win,axis=0)
                        MAX = X_win.max(axis=0)
                        MIN = X_win.min(axis=0)
                        Q3 = np.percentile(X_win,75,axis=0)
                        Q1 = np.percentile(X_win,25,axis=0)
                        IRQ = Q3- Q1
                        COV_M = np.cov(X_win.T)
                        COV1 = np.array([COV_M[0,1], COV_M[1,2], COV_M[0,2]])
                        COV2 = np.array([COV_M[3,4], COV_M[4,5], COV_M[3,5]])
                        COV3 = np.array([COV_M[0,3], COV_M[1,4], COV_M[2,5]])

                        FFTX = calc_fft(X_win[:,0], sampling_rate)
                        FFTY = calc_fft(X_win[:,1], sampling_rate)
                        FFTZ = calc_fft(X_win[:,2], sampling_rate)

                        FFTX8 = FFTX[:8]
                        FFTY8 = FFTY[:8]
                        FFTZ8 = FFTZ[:8]
                        
                        ENGYX = sum(i*i for i in FFTX[1:])
                        ENGYY = sum(i*i for i in FFTY[1:])
                        ENGYZ = sum(i*i for i in FFTZ[1:])
                        ENGY = np.array([ENGYX*ENGYX + ENGYY*ENGYY + ENGYZ*ENGYZ])


                        F_win = hstack((MEAN,VAR))
                        F_win = hstack((F_win,SK))
                        F_win = hstack((F_win,KURT))
                        F_win = hstack((F_win,RMS))
                        F_win = hstack((F_win,MED))
                        F_win = hstack((F_win,MAX))
                        F_win = hstack((F_win,MIN))
                        F_win = hstack((F_win,Q3))
                        F_win = hstack((F_win,Q1))
                        F_win = hstack((F_win,IRQ))
                        F_win = hstack((F_win,COV1))
                        F_win = hstack((F_win,COV2))
                        F_win = hstack((F_win,COV3))
                        F_win = hstack((F_win,FFTX8))
                        F_win = hstack((F_win,FFTY8))
                        F_win = hstack((F_win,FFTZ8))
                        F_win = hstack((F_win,ENGYX))
                        F_win = hstack((F_win,ENGYY))
                        F_win = hstack((F_win,ENGYZ))
                        F_win = hstack((F_win,ENGY))



                        """
                        head (first half)
                        """
                        X_head = X_win[:len(X_win)//2]

                        MEAN = mean(X_head,axis=0)
                        VAR = var(X_head,axis=0)
                        SK = stats.skew(X_head,axis=0)
                        KURT = stats.kurtosis(X_head,axis=0)
                        RMS = sqrt(mean(X_head**2,axis=0))
                        MED = median(X_head,axis=0)
                        MAX = X_head.max(axis=0)
                        MIN = X_head.min(axis=0)
                        Q3 = np.percentile(X_head,75,axis=0)
                        Q1 = np.percentile(X_head,25,axis=0)
                        IRQ = Q3- Q1
                        COV_M = np.cov(X_head.T)
                        COV1 = np.array([COV_M[0,1], COV_M[1,2], COV_M[0,2]])
                        COV2 = np.array([COV_M[3,4], COV_M[4,5], COV_M[3,5]])
                        COV3 = np.array([COV_M[0,3], COV_M[1,4], COV_M[2,5]])

                        FFTX = calc_fft(X_head[:,0], sampling_rate)
                        FFTY = calc_fft(X_head[:,1], sampling_rate)
                        FFTZ = calc_fft(X_head[:,2], sampling_rate)

                        FFTX8 = FFTX[:8]
                        FFTY8 = FFTY[:8]
                        FFTZ8 = FFTZ[:8]
                        
                        ENGYX = sum(i*i for i in FFTX[1:])
                        ENGYY = sum(i*i for i in FFTY[1:])
                        ENGYZ = sum(i*i for i in FFTZ[1:])
                        ENGY = np.array([ENGYX*ENGYX + ENGYY*ENGYY + ENGYZ*ENGYZ])

                        F_win = hstack((MEAN,VAR))
                        F_win = hstack((F_win,SK))
                        F_win = hstack((F_win,KURT))
                        F_win = hstack((F_win,RMS))
                        F_win = hstack((F_win,MED))
                        F_win = hstack((F_win,MAX))
                        F_win = hstack((F_win,MIN))
                        F_win = hstack((F_win,Q3))
                        F_win = hstack((F_win,Q1))
                        F_win = hstack((F_win,IRQ))
                        F_win = hstack((F_win,COV1))
                        F_win = hstack((F_win,COV2))
                        F_win = hstack((F_win,COV3))
                        F_win = hstack((F_win,FFTX8))
                        F_win = hstack((F_win,FFTY8))
                        F_win = hstack((F_win,FFTZ8))
                        F_win = hstack((F_win,ENGYX))
                        F_win = hstack((F_win,ENGYY))
                        F_win = hstack((F_win,ENGYZ))
                        F_win = hstack((F_win,ENGY))

                        """
                        tail (second half)
                        """
                        X_tail = X_win[len(X_win)//2:]

                        MEAN = mean(X_tail,axis=0)
                        VAR = var(X_tail,axis=0)
                        SK = stats.skew(X_tail,axis=0)
                        KURT = stats.kurtosis(X_tail,axis=0)
                        RMS = sqrt(mean(X_tail**2,axis=0))
                        MED = median(X_tail,axis=0)
                        MAX = X_tail.max(axis=0)
                        MIN = X_tail.min(axis=0)
                        Q3 = np.percentile(X_tail,75,axis=0)
                        Q1 = np.percentile(X_tail,25,axis=0)
                        IRQ = Q3- Q1
                        COV_M = np.cov(X_tail.T)
                        COV1 = np.array([COV_M[0,1], COV_M[1,2], COV_M[0,2]])
                        COV2 = np.array([COV_M[3,4], COV_M[4,5], COV_M[3,5]])
                        COV3 = np.array([COV_M[0,3], COV_M[1,4], COV_M[2,5]])

                        FFTX = calc_fft(X_tail[:,0], sampling_rate)
                        FFTY = calc_fft(X_tail[:,1], sampling_rate)
                        FFTZ = calc_fft(X_tail[:,2], sampling_rate)

                        FFTX8 = FFTX[:8]
                        FFTY8 = FFTY[:8]
                        FFTZ8 = FFTZ[:8]
                        
                        ENGYX = sum(i*i for i in FFTX[1:])
                        ENGYY = sum(i*i for i in FFTY[1:])
                        ENGYZ = sum(i*i for i in FFTZ[1:])
                        ENGY = np.array([ENGYX*ENGYX + ENGYY*ENGYY + ENGYZ*ENGYZ])

                        F_win = hstack((MEAN,VAR))
                        F_win = hstack((F_win,SK))
                        F_win = hstack((F_win,KURT))
                        F_win = hstack((F_win,RMS))
                        F_win = hstack((F_win,MED))
                        F_win = hstack((F_win,MAX))
                        F_win = hstack((F_win,MIN))
                        F_win = hstack((F_win,Q3))
                        F_win = hstack((F_win,Q1))
                        F_win = hstack((F_win,IRQ))
                        F_win = hstack((F_win,COV1))
                        F_win = hstack((F_win,COV2))
                        F_win = hstack((F_win,COV3))
                        F_win = hstack((F_win,FFTX8))
                        F_win = hstack((F_win,FFTY8))
                        F_win = hstack((F_win,FFTZ8))
                        F_win = hstack((F_win,ENGYX))
                        F_win = hstack((F_win,ENGYY))
                        F_win = hstack((F_win,ENGYZ))
                        F_win = hstack((F_win,ENGY))

                        F_win = F_win[np.newaxis,:]


                        if start_flg == 0:
                            F_all = F_win
                            start_flg = 1
                            print(F_all.shape)

                        else:
                            print(F_all.shape)
                            print(F_win.shape)

                            F_all = vstack((F_all,F_win))

            print(F_all.shape)

            out_file = os.path.join(SAMPLE_FEAT_FOLDER, act, act+'_groundtruthsegmentation_allheadtail_features.csv')
            create_folder(os.path.join(SAMPLE_FEAT_FOLDER, act))

            savetxt(out_file, F_all, delimiter=",")



                # print("")
                # print("Positive Examples: " + str(pos_examples_counter))
                # print("Negative Examples: " + str(neg_examples_counter))
                # print("")





                # print ""
                # print "Shape of training data: " + str(Z_T.shape)
                # print ""
                # print str(Z_T)
                # print ""

                # # Number of inputs
                # number_of_inputs = Z_T.shape[1]-1

                # predicted = clf.predict(X_T)


                # end = time.time()
                # print ""
                # print ""
                # print ""
                # print("time eclaped:")
                # print(end - ts)
                # print ""

                # print ""
                # print "Shape of Predicted: " + str(predicted.shape)
                # print ""

                # prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm = calc_cm(Y_T, predicted)
                # print(cm)
                # print(prec_pos)
                # print(f1_pos)



                    