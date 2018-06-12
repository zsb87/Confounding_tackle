from ..core import batch_run, maybe_create_folder
from .. import settings
import os
import pandas as pd
from datetime import datetime
import logging
import numpy as np


logger = logging.getLogger(__name__)


def split_hour(raw_file, out_folder, parse_function):
    '''Remap samples into hour files

    Samples are guaranteed to be in the correct hour file
    Outlier will be removed (1970 or in the future), 
    determined by outlier_time function

    Also each sample will be parsed following the parse_function

    Parameters:

    raw_file: 
            absolute path of the original data file

    out_folder:
            output folder containing dates/hours

    parse_function:
            return a dict of filename map to data string

    '''

    logger.info("Mapping %s to hour file", raw_file)


    split_hour, header_list = parse_function(raw_file)

    for filename, data in sorted(split_hour.items()):

        hour_path = os.path.join(out_folder, filename)

        if not os.path.exists(hour_path):
            with open(hour_path, 'w') as f:
                f.write(','.join(header_list) + '\n')

        with open(hour_path,'a') as f:            
            f.write(data)


def get_reliability(df):
    '''
    Calculate reliability of a dataframe
    '''
    time_ms = df['Time'].as_matrix()
    time_sec = time_ms//1000

    time, count = np.unique(time_sec, return_counts=True)

    df_reliability = pd.DataFrame({ 'Time':1000*time,\
                                    'Count':count})

    return df_reliability
