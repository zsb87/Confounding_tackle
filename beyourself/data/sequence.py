import os
from .. import settings
from ..settings import *
import pandas as pd
from datetime import datetime, timedelta
import logging
from ..core.algorithm import *
from ..core.util import *
import numpy as np
from numpy.random import choice, randint
from .label import * 
from .sensordata import *
import logging


logger = logging.getLogger(__name__)


class SequentialTrainChewingData():

    def __init__(self, json_path, keylist, mode):

        self.ratios = []
        self.segments = []
        self.chewingpath = []
        self.bitespath = []
        self.keylist = keylist
        self.mode = mode

        with open(json_path) as f:
            data = json.load(f)

            for k, v in sorted(data.items()):
                self.ratios.append(v['ratio'])

                seg = {}
                seg['start'] = human_to_epoch(v['start'])
                seg['end'] = human_to_epoch(v['end'])
                seg['subj'] = v['subj']

                self.segments.append(seg)

                if 'chewing' in v:
                    self.chewingpath.append(v['chewing'])
                else:
                    self.chewingpath.append(None)

                if 'bites' in v:
                    self.bitespath.append(v['bites'])
                else:
                    self.bitespath.append(None)


        self.ratios = np.array(self.ratios)/np.sum(self.ratios)

        logger.debug("Chewing label paths:")
        logger.debug(self.chewingpath)
        logger.debug("Bites label paths:")
        logger.debug(self.bitespath)

    def next_batch(self):

        if self.mode == 'train':

            batch_x = np.zeros((BATCH_SIZE, N_CHUNK, WIN*N_SENSOR))
            batch_y = np.zeros((BATCH_SIZE, N_CHUNK))

            idx_segment = choice(len(self.ratios), 1, p=self.ratios)[0]

            logger.debug('Random pick segment: {}'.format(idx_segment))

            segment = self.segments[idx_segment]

            df = get_necklace(segment['subj'], segment['start'], segment['end'])
            # df['energy'] = df['aX'].pow(2) + df['aY'].pow(2) + df['aZ'].pow(2)

            # ================= generate chewing continuous label ================
            if self.chewingpath[idx_segment] == None:
                label_chewing = np.zeros((df.shape[0],))
            else:
                df_chewing = read_SYNC(os.path.join(settings.CLEAN_FOLDER, 
                        self.chewingpath[idx_segment]))

                # convert datetime column to unix timestamp column
                df_chewing['start'] = (df_chewing['start'].astype(np.int64)/1e6).astype(int)
                df_chewing['end'] = (df_chewing['end'].astype(np.int64)/1e6).astype(int)
    
                # merging
                label_chewing = point_intersect_interval(df['Time'].as_matrix(), df_chewing)


            df = df[self.keylist]

            for b in range(settings.BATCH_SIZE):
                start = randint(0, df.shape[0] - MAX_LENGTH)

                query = df.iloc[start:start + MAX_LENGTH]
                x = query[self.keylist].as_matrix()

                b_x = np.reshape(x, (N_CHUNK, WIN*N_SENSOR))
                batch_x[b,:,:] = b_x


                query_chewing = label_chewing[start:start + MAX_LENGTH]
                b_chew = query_chewing[(WIN//2)::WIN]
                batch_y[b,:] = b_chew

            return batch_x.astype(float), batch_y.astype(int)


class SequentialTestChewingData():

    def __init__(self, json_path, keylist):

        with open(json_path) as f:
            self.data = json.load(f)
            self.keylist = keylist

    def next_batch(self):

        df = get_necklace(self.data['subj'], human_to_epoch(self.data['start']),\
                                             human_to_epoch(self.data['end']))

        # df['energy'] = df['aX'].pow(2) + df['aY'].pow(2) + df['aZ'].pow(2)


        df_chewing = read_SYNC(os.path.join(settings.CLEAN_FOLDER, self.data['chewing']))
        
        df_chewing['start'] = (df_chewing['start'].astype(np.int64)/1e6).astype(int)
        df_chewing['end'] = (df_chewing['end'].astype(np.int64)/1e6).astype(int)


        label_chewing = point_intersect_interval(df['Time'].as_matrix(), df_chewing)

        
        # df_bites = read_SYNC(os.path.join(settings.CLEAN_FOLDER, 
        #             self.data['bites']))

        # df_bites['start'] = (df_bites['start'].astype(np.int64)/1e6).astype(int)
        # df_bites['end'] = (df_bites['end'].astype(np.int64)/1e6).astype(int)


        # df_bites['start'] = df_bites['start'] - 500
        # df_bites['end'] = df_bites['end'] + 500


        # label_bites = point_intersect_interval(df['Time'].as_matrix(), df_bites)
        
        N = df.shape[0]

        logger.debug(N)

        N_even = ((N - 1)//(BATCH_SIZE*WIN) + 1)*BATCH_SIZE*WIN

        logger.debug(N_even)

        batch_x = df[self.keylist].as_matrix()

        if N_even > N:
            batch_x = np.vstack((batch_x, np.zeros((N_even - N,batch_x.shape[1]))))
            label_chewing = np.append(label_chewing, np.zeros((N_even - N,)))
            # label_bites = np.append(label_bites, np.zeros((N_even - N,)))

        batch_x = np.reshape(batch_x, (BATCH_SIZE, -1, WIN*N_SENSOR))

        # majority voting
        label_chewing = np.reshape(label_chewing, (-1, WIN))
        label_sum_chewing = np.sum(label_chewing, axis = 1)
        batch_y_chewing = label_sum_chewing > WIN//2

        batch_y_chewing = np.reshape(batch_y_chewing, (BATCH_SIZE, -1))
        
    
        return batch_x.astype(float), batch_y_chewing.astype(int)



class SequentialTrainCBData():

    def __init__(self, json_path, keylist, mode):

        self.ratios = []
        self.segments = []
        self.chewingpath = []
        self.bitespath = []
        self.keylist = keylist
        self.mode = mode

        with open(json_path) as f:
            data = json.load(f)

            for k, v in sorted(data.items()):
                self.ratios.append(v['ratio'])

                seg = {}
                seg['start'] = human_to_epoch(v['start'])
                seg['end'] = human_to_epoch(v['end'])
                seg['subj'] = v['subj']

                self.segments.append(seg)

                if 'chewing' in v:
                    self.chewingpath.append(v['chewing'])
                else:
                    self.chewingpath.append(None)

                if 'bites' in v:
                    self.bitespath.append(v['bites'])
                else:
                    self.bitespath.append(None)


        self.ratios = np.array(self.ratios)/np.sum(self.ratios)

        logger.debug("Chewing label paths:")
        logger.debug(self.chewingpath)
        logger.debug("Bites label paths:")
        logger.debug(self.bitespath)

    def next_batch(self):

        if self.mode == 'train':

            batch_x = np.zeros((BATCH_SIZE, N_CHUNK, WIN*N_SENSOR))
            batch_y = np.zeros((BATCH_SIZE, N_CHUNK, N_CLASS_CHEWING_BITES))

            idx_segment = choice(len(self.ratios), 1, p=self.ratios)[0]

            logger.debug('Random pick segment: {}'.format(idx_segment))

            segment = self.segments[idx_segment]

            df = get_necklace(segment['subj'], segment['start'], segment['end'])
            df['energy'] = df['aX'].pow(2) + df['aY'].pow(2) + df['aZ'].pow(2)

            # ================= generate chewing continuous label ================
            if self.chewingpath[idx_segment] == None:
                label_chewing = np.zeros((df.shape[0],))
            else:
                df_chewing = read_SYNC(os.path.join(settings.CLEAN_FOLDER, 
                        self.chewingpath[idx_segment]))

                df_chewing['start'] = (df_chewing['start'].astype(np.int64)/1e6).astype(int)
                df_chewing['end'] = (df_chewing['end'].astype(np.int64)/1e6).astype(int)
    
                label_chewing = point_intersect_interval(df['Time'].as_matrix(), df_chewing)

            # =================== generate bites continuous label ========================
            # bites: expand the label by 1 sec

            if self.bitespath[idx_segment] == None:
                label_bites = np.zeros((df.shape[0],))
            else:
                df_bites = read_SYNC(os.path.join(settings.CLEAN_FOLDER, 
                        self.bitespath[idx_segment]))

                df_bites['start'] = (df_bites['start'].astype(np.int64)/1e6).astype(int)
                df_bites['end'] = (df_bites['end'].astype(np.int64)/1e6).astype(int)

                df_bites['start'] = df_bites['start'] - 500
                df_bites['end'] = df_bites['end'] + 500

                label_bites = point_intersect_interval(df['Time'].as_matrix(), df_bites)


            df = df[self.keylist]

            for b in range(settings.BATCH_SIZE):
                start = randint(0, df.shape[0] - MAX_LENGTH)

                query = df.iloc[start:start + MAX_LENGTH]
                x = query[self.keylist].as_matrix()

                b_x = np.reshape(x, (N_CHUNK, WIN*N_SENSOR))
                batch_x[b,:,:] = b_x


                query_chewing = label_chewing[start:start + MAX_LENGTH]
                b_chew = query_chewing[(WIN//2)::WIN]
                batch_y[b,:,0] = b_chew

                query_bites = label_bites[start:start + MAX_LENGTH]
                b_bite = query_bites[(WIN//2)::WIN]
                batch_y[b,:,1] = b_bite

            return batch_x.astype(float), batch_y.astype(int)


class SequentialTestCBData():

    def __init__(self, json_path, keylist):

        with open(json_path) as f:
            self.data = json.load(f)
            self.keylist = keylist

    def next_batch(self):

        df = get_necklace(self.data['subj'], human_to_epoch(self.data['start']),\
                                             human_to_epoch(self.data['end']))

        df['energy'] = df['aX'].pow(2) + df['aY'].pow(2) + df['aZ'].pow(2)


        df_chewing = read_SYNC(os.path.join(settings.CLEAN_FOLDER, self.data['chewing']))
        
        df_chewing['start'] = (df_chewing['start'].astype(np.int64)/1e6).astype(int)
        df_chewing['end'] = (df_chewing['end'].astype(np.int64)/1e6).astype(int)


        label_chewing = point_intersect_interval(df['Time'].as_matrix(), df_chewing)

        
        df_bites = read_SYNC(os.path.join(settings.CLEAN_FOLDER, 
                    self.data['bites']))

        df_bites['start'] = (df_bites['start'].astype(np.int64)/1e6).astype(int)
        df_bites['end'] = (df_bites['end'].astype(np.int64)/1e6).astype(int)


        df_bites['start'] = df_bites['start'] - 500
        df_bites['end'] = df_bites['end'] + 500


        label_bites = point_intersect_interval(df['Time'].as_matrix(), df_bites)
        
        N = df.shape[0]

        N_even = ((N - 1)//(BATCH_SIZE*WIN) + 1)*BATCH_SIZE*WIN

        batch_x = df[self.keylist].as_matrix()

        if N_even > N:
            batch_x = np.vstack((batch_x, np.zeros((N_even - N,batch_x.shape[1]))))
            label_chewing = np.append(label_chewing, np.zeros((N_even - N,)))
            label_bites = np.append(label_bites, np.zeros((N_even - N,)))

        batch_x = np.reshape(batch_x, (BATCH_SIZE, -1, WIN*N_SENSOR))

        # majority voting
        label_chewing = np.reshape(label_chewing, (-1, WIN))
        label_sum_chewing = np.sum(label_chewing, axis = 1)
        batch_y_chewing = label_sum_chewing > WIN//2

        batch_y_chewing = np.reshape(batch_y_chewing, (BATCH_SIZE, -1))
        

        # majority voting
        label_bites = np.reshape(label_bites, (-1, WIN))
        label_sum_bites = np.sum(label_bites, axis = 1)
        batch_y_bites = label_sum_bites > WIN//2
        

        batch_y_bites = np.reshape(batch_y_bites, (BATCH_SIZE, -1))

        batch_y = np.stack((batch_y_chewing, batch_y_bites),axis=-1)

        return batch_x.astype(float), batch_y.astype(int)