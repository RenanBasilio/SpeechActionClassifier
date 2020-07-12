#!/usr/bin/python
import pathlib
import sys
import shutil
import tempfile
import subprocess
import os
import argparse
import time

import pandas as pd
import cv2 as cv2
from termcolor import cprint

from modules.filters import draw_dot, draw_face_chip, draw_facial_reco
from modules.utils import print_progress

def main(args):
    vidpath = pathlib.Path(args.video.name).resolve()
    diarpath = pathlib.Path(args.diarization.name).resolve()

    if args.output is None:
        outbase = vidpath.parent / vidpath.stem
        args.output = outbase.with_suffix('.diar.mp4')
    else:
        args.output = pathlib.Path(args.output).resolve()

    if (args.output.exists() and args.overwrite != True):
        cprint("ERROR: Output file {} already exists.".format(args.output), 'red', file=sys.stderr)
        exit(0)
    else:
        fd, tmpfile = tempfile.mkstemp(".mp4")
        os.close(fd)

    capture = cv2.VideoCapture(str(vidpath.absolute()))

    if args.chip:
        outwidth = 150
        outheight = 150
    else:
        outwidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        outheight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tempout = cv2.VideoWriter(tmpfile, cv2.VideoWriter_fourcc(*'mp4v'), capture.get(cv2.CAP_PROP_FPS), (outwidth, outheight))
    framecount = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    start_time = time.time()

    args.diarization.readline()
    currframe = 0
    for line in args.diarization:
        # [ prediction, frame_start, frame_dur, time_start, time_dur, mean_confidence, frame_confidences... ]
        params = line.split(',')
        for i in range(int(params[2])):
            ret, frame = capture.read()
            if ret:
                color = (0,0,0)
                if params[0] == 'Idle':
                    color = (0,0,255)
                elif params[0] == 'Speak':
                    color = (0,255,0)

                if args.chip:
                    draw_face_chip(frame, color=color)
                elif args.overlay:
                    draw_facial_reco(frame, color=color)

                draw_dot(frame, color=color)
                tempout.write(frame)
                currframe += 1
                if not args.silent:
                    print_progress(currframe, framecount)
            else:
                raise("Reached end of video before end of data.")

    capture.release()
    tempout.release()

    if args.audio:
        try:
            cmd = [ 'ffmpeg', '-i', tmpfile, '-i', args.video, '-y', '-hide_banner', '-loglevel', 'error', '-nostats', 
                    '-map', '0:0', '-map', '1:1?', '-c:v', 'copy', '-c:a', 'copy', '-shortest', args.output ]
            returned_output = subprocess.call(cmd)
            os.remove(tmpfile)
        except:
            shutil.move(tmpfile, args.output)
    else:
        shutil.move(tmpfile, args.output)

    if not args.silent:
        print("--- Took {} s ---            ".format(time.time() - start_time))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate a visualization for a diarized video.")
    argparser.add_argument('video', type=argparse.FileType('r'), help="input video file")
    argparser.add_argument('diarization', type=argparse.FileType('r'), help="full diarization file")

    argparser.add_argument('-o', '--output', nargs='?', 
        help="output file path (default same as video path)", metavar="PATH")

    argparser.add_argument('-a','--audio',  action='store_true',
        help="include audio")

    vis_config = argparser.add_mutually_exclusive_group(required=False)
    vis_config.add_argument('--overlay',  action='store_true',
        help="overlay facial landmarking results")
    vis_config.add_argument('--chip',  action='store_true',
        help="output face chip only")

    argparser.add_argument('-s', '--silent', action='store_true',
        help="only log errors")
    argparser.add_argument('--overwrite',  action='store_true',
        help="overwrite output files if they already exist")

    args = argparser.parse_args()
    main(args)