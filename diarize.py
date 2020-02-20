import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from loader import load_frame_as_ndarray, print_video_frames, compute_facial_landmarks, get_label_from_index
from preprocess import print_progress
from termcolor import cprint
import numpy as np
import cv2 as cv2
import tensorflow as tf
import pathlib
import math
import subprocess
import sys

def draw_dot(frame, color):
    cx, cy = 310, 10
    y, x = np.ogrid[-5:5, -5:5]
    frame[cy-5:cy+5, cx-5:cx+5][x**2 + y**2 <= 25] = color
    return frame

def diarize(path, outpath):
    model_path = pathlib.Path("export/model_2020-2-19-2230.h5")
    model = tf.keras.models.load_model(model_path)

    temppath = pathlib.Path("export/diarized/") / (str(path.stem) + ".temp.mp4")
    capture = cv2.VideoCapture(str(path.absolute()))
    out = cv2.VideoWriter(str(temppath.absolute()), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (320, 240))

    ret = True
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    curr = 0
    errors = False

    while(capture.isOpened()):
        window = []
        prediction_window = []
        c = 0
        print_progress(curr, length)
        print(" {0:.1f}%                            ".format((curr / (length - 1)) * 100) ,end='')

        while c < 15:
            ret, frame = capture.read()
            curr += 1

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window.append(frame)
                frame = np.asarray(frame)
                c+=1

                try:
                    landmarks = compute_facial_landmarks(frame)
                    prediction_window.append(landmarks)
                except:
                    errors = True
                    pass
            else:
                capture.release()
                break
        
        if len(prediction_window) == 15 and errors == False:
            prediction_window = np.asarray(prediction_window)
            prediction_window = np.expand_dims(prediction_window, axis=3)
            #print_video_frames(prediction_window)
            prediction_window = np.expand_dims(prediction_window, axis=0)

            prediction = model.predict(prediction_window)

            label = get_label_from_index(pathlib.Path("data/classes.txt"), np.argmax(prediction[0]))
        else:
            label = None
            errors = False
        
        color = [0,0,0]
        if label == 'Idle':
            color = [255,0,0]
        elif label == 'Speak':
            color = [0,255,0]

        for frame in window:
            frame = draw_dot(frame, color)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            #cv2.imshow('video', frame)
            cv2.waitKey(1)

        #print_video_frames(window)

    capture.release()
    out.release()

    cmd = ['ffmpeg', '-i', temppath.absolute(), '-i', path.absolute(), '-y', '-hide_banner', '-loglevel', 'panic', '-nostats', 
                     '-map', '0:0', '-map', '1:1', '-c:v', 'copy', '-c:a', 'copy', '-shortest', outpath.absolute()]
    returned_output = subprocess.check_output(cmd)
    temppath.unlink()
    #cv2.destroyAllWindows()
    print("\rDone                                            ")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        outdir =  pathlib.Path("export/diarized/")
        outdir.mkdir(parents=True, exist_ok=True)

        path = pathlib.Path(sys.argv[1]).resolve()
        print("Processing file {}...".format(path.absolute()))

        outpath = outdir / (str(path.stem) + ".diarized.mp4")

        diarize(path, outpath)
    else:
        print("Diarize a video, producing a video with an indicator of whether the person is speaking or not.")
        print("The output will be placed in a folder named export/diarized in the execution directory.")
        print("")
        print("Usage: ")
        print("      diarize.py <video>")
        print("")
