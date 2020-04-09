import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from modules.diarizer import Diarizer
from modules.utils import print_progress
import pathlib
import sys
import time
import datetime

diarizer = Diarizer(pathlib.Path("models/2020-04-05-1754_model.h5").absolute(), commit_strategy="sum", shift=1)

if __name__ == "__main__":
    start_time = time.time()

    segments = diarizer.diarize(sys.argv[1], progress_callback=print_progress)
    with open('export/output.txt', 'w') as f:
        for item in segments:
            f.write("%s\n" % item)

    print("--- Took {} s ---            ".format(time.time() - start_time))