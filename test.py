import os
import pathlib
import sys
import time
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from modules.diarizer import Diarizer
from modules.utils import print_progress

diarizer = Diarizer(pathlib.Path("models/2020-04-05-1754_model.h5").absolute(), commit_strategy="gaussf", shift=1)

if __name__ == "__main__":
    start_time = time.time()

    segments = diarizer.diarize(sys.argv[1], progress_callback=print_progress)
    diar = open(sys.argv[1] + '.diar', 'w')
    rttm = open(sys.argv[1] + '.rttm', 'w')
    for item in segments:
        diar.write("{}\n".format(item))
        if item.value == 1:
            rttm.write("SPEAKER {file} 1 {turn_onset:.3f} {turn_dur:.3f} <NA> <NA> speaker <NA> <NA>\n".format(
                file=os.path.basename(sys.argv[1]).split('.')[0],
                turn_onset=item.onset,
                turn_dur=item.duration
            ))
    diar.close()
    rttm.close()

    print("--- Took {} s ---            ".format(time.time() - start_time))
