import os
from .. import settings
import pandas as pd
from datetime import datetime, timedelta
import logging
from ..core.util import human_to_epoch
import numpy as np


logger = logging.getLogger(__name__)


def get_necklace(subj, start, end, reliability=0.2):
    folder = os.path.join(settings.CLEAN_FOLDER, subj + '/necklace')

    return _get_data(folder, start, end, reliability)


def get_necklace_timestr(subj, start_str, end_str, reliability=0.2):
    folder = os.path.join(settings.CLEAN_FOLDER, subj + '/necklace')

    return _get_data_timestr(folder, start_str, end_str, reliability)


def _get_data_timestr(folder, start_str, end_str, reliability):
    logger.debug("Querying from {} to {}".format(start_str, end_str))
    start = human_to_epoch(start_str)
    end = human_to_epoch(end_str)

    return _get_data(folder, start, end, reliability)


def _get_data(folder, start, end, reliability):
    '''
    Get data from the cleaned folder

    folder: path to the clean folder, following time-series structure
    start: unix time in ms
    end:   unix time in ms
    contiguous: whether need to return contiguous data or not
    reliability: lowest reliability score
    
    Returns:
    --------

    pandas dataframe

    '''

    logger.debug("Querying from {} to {}, lowest reliability {}".format(start, end, reliability))

    # reliability_path = os.path.join(folder, 'reliability.csv')
    # df = pd.read_csv(reliability_path, index_col=False)

    # query = df[(df['Time'] >= start) & (df['Time'] <= end)]

    # if query.empty:
    #     logger.debug("Empty query")
    #     return None

    # if (query['reliability'] < reliability).any():

    #     rel = query['reliability'].as_matrix()
    #     index = np.where(rel < reliability)
    #     logger.info(rel[index])
    #     return None

    start_dt = datetime.fromtimestamp(start // 1000)
    end_dt = datetime.fromtimestamp(end // 1000)

    concat_list = []

    current = start_dt.replace(minute=0, second=0, microsecond=0)
    while (current < end_dt):
        path = os.path.join(folder, 'data/{}_{:02d}.csv'.format(
            current.strftime(settings.DATEFORMAT), current.hour))

        if os.path.isfile(path):
            logger.debug(path)
            df = pd.read_csv(path, index_col=False)
            df = df[(df.Time >= start) & (df.Time <= end)]

            concat_list.append(df)

        current += timedelta(hours=1)

    if concat_list:
        total = pd.concat(concat_list)
        return total.reset_index(drop=True)
    else:
        return None
