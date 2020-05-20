#!/usr/bin/python
import os
import pathlib
import sys
import time
import datetime
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from modules.diarizer import Diarizer
from modules.utils import print_progress

classes = ['Idle', 'Speak']

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Diarize a video file.")
    argparser.add_argument('video', type=argparse.FileType('r'), help="input video file")
    argparser.add_argument('model', type=argparse.FileType('r'), help="prediction model to use")
    argparser.add_argument('-o', '--output', nargs='?', 
        help="output file path (default same as video path)", metavar="PATH")
    argparser.add_argument('--conf', nargs='?', 
        default=None, const='',
        help="output confidences file (default same as output path)", metavar="PATH")
    argparser.add_argument('--diar', nargs='?', 
        default=None, const='',
        help="output full diarization file (default same as output path)", metavar="PATH")
    args = argparser.parse_args()

    vidpath = pathlib.Path(args.video.name).resolve()
    outbase = vidpath.parent / vidpath.stem

    if args.output is None:
        args.output = open(vidpath.with_suffix('.rttm'), 'w')
    else:
        outbase = pathlib.Path(args.output).parent / pathlib.Path(args.output).stem
        args.output = open(args.output, 'w')

    if args.conf == '':
        args.conf = open(outbase.with_suffix('.conf.csv'), 'w')
    elif args.conf is not None:
        args.conf = open(pathlib.Path(args.conf), 'w')

    if args.diar == '':
        args.diar = open(outbase.with_suffix('.diar.csv'), 'w')
    elif args.diar is not None:
        args.diar = open(pathlib.Path(args.diar), 'w')

    diarizer = Diarizer(pathlib.Path(args.model.name).absolute(), commit_strategy="gaussf", shift=1, classes=classes)

    print("Diarizing {}...".format(vidpath))
    start_time = time.time()
    segments = diarizer.diarize(str(vidpath.absolute()), progress_callback=print_progress)

    if args.diar is not None:
        args.diar.write("prediction,frame_start,frame_dur,time_start,time_dur,mean_confidence,frame_confidences\n")
    if args.conf is not None:
        args.conf.write(','.join(classes) + '\n')

    for item in segments:
        if args.diar is not None:
            args.diar.write("{prediction},{start_frame},{dur_frame},{start_sec:.5f},{dur_sec:.5f},{mean_confidence:.5f},{confidences}\n".format(
                prediction=item.value,
                start_frame=item.frame_onset,
                dur_frame=item.frame_duration,
                start_sec=item.onset,
                dur_sec=item.duration,
                mean_confidence=item.confidence,
                confidences=','.join(['%.5f' % num for num in item.frame_confidences])
            ))
        if item.value == 'Speak':
            args.output.write("SPEAKER {file} 1 {turn_onset:.6f} {turn_dur:.3f} <NA> <NA> speaker {confidence:.3f} <NA>\n".format(
                file=vidpath.stem,
                turn_onset=item.onset,
                turn_dur=item.duration,
                confidence=item.confidence
            ))
        if args.conf is not None:
            for confidence in item.frame_confidences:
                row = ""
                for c in classes:
                    if item.value == c:
                        row += ('%.5f' % confidence)
                    else:
                        row += '0.00000'
                    row += ','
                args.conf.write(row + '\n')

    if args.diar is not None:
        args.diar.close()

    if args.conf is not None:
        args.conf.close()
    
    args.output.close()

    print("--- Took {} s ---            ".format(time.time() - start_time))
