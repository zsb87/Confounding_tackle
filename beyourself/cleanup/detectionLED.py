import numpy as np
import cv2
from ..core.util import maybe_create_folder, epoch_to_relative_str
from datetime import timedelta, time, datetime
from .. import settings
import logging
import os


logger = logging.getLogger(__name__)


def get_LED_events(video_path, out_folder):
    '''
    Given absolute path to a video,
    find all LED events in that
    
    Parameters:

    video_path: path to the video

    out_folder: folder will contains frames of LED detection.
    '''

    logger.info(video_path)

    maybe_create_folder(out_folder)

    cap = cv2.VideoCapture(video_path)

    mask = np.zeros((360,640), np.uint8)
    mask[200:300,170:270] = 1

    CHANGE_WITHIN_LED = 0.75
    CHANGE_TO_LED = 0.6

    logging_time_count = 0


    # -1 is the successful state
    # columns: small change/intermediate/large change
    STATE_MACHINE = {0: (0,0,1),
                     1: (2,0,0),
                     2: (3,0,0),
                     3: (4,0,0),
                     4: (5,0,-1),
                     5: (6,0,-1),
                     6: (7,0,-1),
                     7: (0,0,-1),
                     -1: (0,0,0) 
                     }

    index_state_machine = 0

    prev = None

    f = open("correl.csv", "w")

    while(cap.isOpened()):
        ret, im = cap.read()

        if (type(im) == type(None)):
            break

        logging_time_count += 1
        if logging_time_count > 5*60*10:
            logging_time_count = 0
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            logger.info("Has processed until %s", epoch_to_relative_str(current_time))

        if ret is True:
            im_masked = im * mask[:,:,np.newaxis]

            hist = cv2.calcHist([im], [2], mask, [256],[0,256])

            if type(prev) == type(None):
                prev = np.copy(hist)

            change = cv2.compareHist(hist, prev, cv2.HISTCMP_CORREL)
            prev = np.copy(hist)

            f.write("{}\n".format(change))
           
            if (change > CHANGE_WITHIN_LED):
                jump = 0
            elif change > CHANGE_TO_LED:
                jump = 1
            else:
                jump = 2

            index_state_machine = STATE_MACHINE[index_state_machine][jump]


            if index_state_machine == -1:
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                logger.info("Found LED event at %s", current_time)
                outpath = os.path.join(out_folder, \
                    "{}.png".format(epoch_to_relative_str(current_time)))
                cv2.imwrite(outpath, im)

    cap.release()

    logger.info("DONE")