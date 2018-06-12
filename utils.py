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



def list_files_in_directory(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def filter(df):
    flt_para = 10
    df.accx = pd.rolling_mean(df.accx, flt_para)
    df.accy = pd.rolling_mean(df.accy, flt_para)
    df.accz = pd.rolling_mean(df.accz, flt_para)
    
    df.rotx = pd.rolling_mean(df.rotx, flt_para)
    df.roty = pd.rolling_mean(df.roty, flt_para)
    df.rotz = pd.rolling_mean(df.rotz, flt_para)
    
    df.pitch_deg = pd.rolling_mean(df.pitch_deg, flt_para)
    df.roll_deg = pd.rolling_mean(df.roll_deg, flt_para)
    
    df = df.dropna()
    return df


def calc_fft(y, freq):
    # TEST CASE:
    # >>>print(calc_fft(np.array([1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2,1,2,3,4,5,4,3]), 16))
    # output:
    # >>>[ 0.10867213  0.22848475  1.67556733  0.1980655   0.11177658  0.08159451
    # 0.07137028  0.12458543  0.26419639  0.10726005]

    # Number of samplepoints
    N = y.shape[0]

    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])

    return amp


# return fft except the foundamental frequency component
def cal_energy_wo_bf(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp[1:])


# return the foundamental/basic frequency component
def cal_energy_bf(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(amp[0]*amp[0])


def cal_energy_all(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp)


def tt_split_pseudo_rand(XY, train_ratio, seed):
    # eg: train_ratio = 0.7

    numL = list(range(10))
    random.seed(seed)
    random.shuffle(numL)

    length = len(XY)
    test_enum = numL[0:10-int(10*train_ratio)]
    test_ind = []

    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys



def calc_multi_cm(y_gt, y_pred):    
    # ct = pd.crosstab(y_gt, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_gt, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    # print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    multi_cm = confusion_matrix(y_gt, y_pred)
    
    accuracy = sum(multi_cm[i,i] for i in range(len(set(y_gt))))/sum(sum(multi_cm[i] for i in range(len(set(y_gt)))))
    recall_all = sum(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(len(set(y_gt)))) for i in range(len(set(y_gt))))/(len(set(y_gt)))
    precision_all = sum(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(len(set(y_gt)))) for i in range(len(set(y_gt))))/(len(set(y_gt)))
    fscore_all = sum(2*(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(len(set(y_gt)))))*(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(len(set(y_gt)))))/(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(len(set(y_gt))))+multi_cm[i,i]/sum(multi_cm[j,i] for j in range(len(set(y_gt))))) for i in range(len(set(y_gt))))/len(set(y_gt))
    
    result={}

    for i in np.unique(y_gt):

        i_gt = (y_gt==i).astype(int)
        i_pred = (y_pred==i).astype(int)

        cm = confusion_matrix(i_gt, i_pred)

        i_result = {}

        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
        # Precision for Positive = TP/(TP + FP)
        prec_pos = TP/(TP + FP)
        # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
        f1_pos = 2*TP/(TP*2 + FP+ FN)
        # TPR = TP/(TP+FN)
        TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(i_gt))))

        i_result = {'recall': TPR, 'precision': prec_pos, 'f1': f1_pos}

        result[str(int(i))] = i_result

    ave_result = {'accuracy':accuracy, 'recall_all':recall_all, 'precision_all':precision_all, 'fscore_all':fscore_all}
    result['average'] = ave_result

    return result, multi_cm




def wacc_from_CM(cm, n_classes):
    """
    weighted accuracy
    """

    if n_classes == 2:

        TN = cm[0,0]
        TP = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]
        ratio = float(FP+TN)/float(TP+FN)
        waccuracy = (TP*ratio+TN)/((TP+FN)*ratio+FP+TN)


    elif n_classes == 3:
        waccuracy = (cm[0,0]*(1.0/(cm[0,0]+cm[0,1]+cm[0,2])) + cm[1,1]*(1.0/(cm[1,0]+cm[1,1]+cm[1,2]) + cm[2,2]*(1.0/(cm[2,0]+cm[2,1]+cm[2,2]))))/3.0
        

    
    result = {'waccuracy':waccuracy}

    return result




def metrics_from_CM(multi_cm, n_classes):

    accuracy = sum(multi_cm[i,i] for i in range(n_classes))/sum(sum(multi_cm[i] for i in range(n_classes)))
    recall_all = sum(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes)) for i in range(n_classes))/n_classes
    precision_all = sum(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes)) for i in range(n_classes))/n_classes
    fscore_all = sum(2*(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes)))*(multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes)))/(multi_cm[i,i]/sum(multi_cm[i,j] for j in range(n_classes))+multi_cm[i,i]/sum(multi_cm[j,i] for j in range(n_classes))) for i in range(n_classes))/n_classes
    
    result = {'accuracy':accuracy, 'recall_all':recall_all, 'precision_all':precision_all, 'fscore_all':fscore_all}

    return result



