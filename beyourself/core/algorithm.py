from __future__ import division
import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree
from beyourself.settings import ABSOLUTE_TIME_FORMAT
from datetime import datetime, timedelta
from beyourself.core.util import *


def get_overlap(a, b):
    '''
    Given two pairs a and b,
    find the intersection between them
    could be either number or datetime objects
    '''

    tmp = min(a[1], b[1]) - max(a[0], b[0])
    
    if isinstance(tmp, timedelta):
        zero_value = timedelta(seconds=0)
    else:
        zero_value = 0
    
    return max(zero_value, tmp)


def _get_sum(segments):

    tmp = [(s[1] - s[0]) for s in segments]

    if isinstance(tmp[0], timedelta):
        out = timedelta(seconds=0)
        for diff in tmp:
            out += diff
        return out
    
    else:
        out = 0
        for diff in tmp:
            out += diff
        return out


def interval_intersect_interval(**kwargs):
    '''
    Efficient algorithm to find which intervals intersect

    Handles both unix timestamp or datetime object

    Return:
    -------

    prediction_gt: 
        array with same size as prediction,
        will be 1 if there's an overlapping label
        0 if not
    recall:
        recall percentage of labels
    overlap:
        how much overlap between label and prediction
    '''

    gt = kwargs['groundtruth']
    pred = kwargs['prediction']

    total_overlap = None
    missed = None
    false_alarm = None

    # calculate recall
    tree = IntervalTree()
    for segment in pred:
        tree.add(Interval(segment[0],segment[1]))

    TP = 0
    for segment in gt:
        overlap = tree.search(segment[0], segment[1])

        if len(overlap) != 0:
            TP += 1

    recall = TP/len(gt)

    # calculate precision
    tree = IntervalTree()
    for segment in gt:
        tree.add(Interval(segment[0],segment[1]))

    prediction_gt = []
    for segment in pred:
        overlap = tree.search(segment[0], segment[1])

        for label in overlap:
            if total_overlap == None:
                total_overlap = get_overlap(label, segment)
            else:
                total_overlap += get_overlap(label, segment)

        if len(overlap) != 0:
            prediction_gt.append(1)
        else:
            prediction_gt.append(0)

    total_groundtruth = _get_sum(gt)

    result = {}
    result['prediction_gt'] = prediction_gt
    result['recall'] = recall
    result['precision'] = np.mean(prediction_gt)

    result['overlap'] = total_overlap
    result['missed'] = total_groundtruth - total_overlap


    return result


def point_intersect_interval(points, df_interval):

    # store index of intervals as value of the interval
    tree = IntervalTree()
    for i in range(df_interval.shape[0]):
        tree[df_interval['start'][i]:df_interval['end'][i]] = i

    points_gt = np.zeros_like(points).astype(bool)

    interval_gt = [False] * df_interval.shape[0]


    for i in range(len(points)):

        intersection = tree.search(points[i])
        if len(intersection) == 0:
            points_gt[i] = False
        else:
            points_gt[i] = True
    
            for segment in intersection:
                interval_gt[segment.data] = True

    results = {}

    results['points_gt'] = points_gt
    results['interval_gt'] = interval_gt

    return results
