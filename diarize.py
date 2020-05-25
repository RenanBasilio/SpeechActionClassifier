#!/usr/bin/python
import os
import pathlib
import sys
import time
import datetime
import argparse

from termcolor import cprint

from modules.utils import print_progress

classes = ['Idle', 'Speak']

def main(args):
    vidpath = pathlib.Path(args.video.name).resolve()
    outbase = vidpath.parent / vidpath.stem

    if args.output is not None:
        outbase = pathlib.Path(args.output).parent / pathlib.Path(args.output).stem

    args.output = outbase.with_suffix('.rttm')
    if (args.output.exists() and args.overwrite != True):
        cprint("ERROR: Output file {} already exists.".format(args.output), 'red', file=sys.stderr)
        exit(0)
    else:
        args.output = open(args.output, 'w')

    if args.conf is not None:
        if args.conf == '':
            args.conf = outbase.with_suffix('.conf.csv')
        else:
            args.conf = pathlib.Path(args.conf)

        if (outbase.with_suffix('.conf.csv').exists() and args.overwrite != True):
            cprint("ERROR: Output file {} already exists.".format(args.conf), 'red', file=sys.stderr)
            exit(0)
        else:
            args.conf = open(pathlib.Path(args.conf), 'w')

    if args.diar is not None:
        if args.diar == '':
            args.diar = outbase.with_suffix('.diar.csv')
        else:
            args.diar = pathlib.Path(args.diar)

        if (outbase.with_suffix('.diar.csv').exists() and args.overwrite != True):
            cprint("ERROR: Output file {} already exists.".format(args.diar), 'red', file=sys.stderr)
            exit(0)
        else:
            args.diar = open(pathlib.Path(args.diar), 'w')

    if args.silent:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        print("Diarizing {}...".format(vidpath))

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    from modules.diarizer import Diarizer

    diarizer = Diarizer(pathlib.Path(args.model.name).absolute(), commit_strategy=args.strat, shift=args.step, classes=classes)

    start_time = time.time()
    segments = diarizer.diarize(str(vidpath.absolute()), progress_callback=(print_progress if not args.silent else None))

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

    if not args.silent:
        print("--- Took {} s ---            ".format(time.time() - start_time))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Diarize a video file.")
    argparser.add_argument('video', type=argparse.FileType('r'), help="input video file")
    argparser.add_argument('model', type=argparse.FileType('r'), help="prediction model to use")

    argparser.add_argument('-s', '--silent', action='store_true',
        help="only log errors")
    argparser.add_argument('-o', '--output', nargs='?', 
        help="output file path (default same as video path)", metavar="PATH")

    out_config = argparser.add_argument_group('additional output arguments')
    out_config.add_argument('--conf', nargs='?', 
        default=None, const='',
        help="output confidences file (default same as output path)", metavar="PATH")
    out_config.add_argument('--diar', nargs='?', 
        default=None, const='',
        help="output full diarization file (default same as output path)", metavar="PATH")

    diar_config = argparser.add_argument_group('diarizer configuration arguments')
    diar_config.add_argument('--strat', nargs='?', choices=['mean', 'median', 'freq', 'gaussw', 'gaussf'],
        default='gaussf',
        help="diarization strategy")
    diar_config.add_argument('--step', nargs='?', type=int,
        default=1,
        help="sliding window size")

    argparser.add_argument('--overwrite',  action='store_true',
        help="overwrite output files if they already exist")

    args = argparser.parse_args()
    main(args)
