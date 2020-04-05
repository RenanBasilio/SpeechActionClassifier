import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from modules.diarizer import Diarizer
from modules.utils import print_progress
import pathlib
import sys
import time
import datetime

diarizer = Diarizer(pathlib.Path("export/model_2020-2-19-2230.h5").absolute(), commit_strategy="maxc", shift=1)

if __name__ == "__main__":
    start_time = time.time()

    segments = diarizer.diarize(sys.argv[1], progress_callback=print_progress)
    with open('export/output.txt', 'w') as f:
        for item in segments:
            f.write("%s\n" % item)

    print("--- Took %s ---" % time.strftime("%-H hrs %-M min %-S sec", datetime.timedelta(seconds=(time.time() - start_time))))