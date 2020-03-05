import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from loader import compute_facial_landmarks
from termcolor import cprint
from imutils import face_utils
from utils import draw_facial_landmarks, print_progress
import numpy as np
import cv2 as cv2
import tensorflow as tf
import pathlib
import math
import dlib
import subprocess
import sys
import time
import os

classes=['Idle','Speak']

face_predictor = None
face_detector = None

def draw_facial_reco(image, color):
    global face_predictor
    global face_detector

    if face_predictor is None:
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

    faces = face_detector(image, 1)

    for (i, face) in enumerate(faces):
        #shape = face_predictor(image, face.rect)
        shape = face_predictor(image, face)
        shape_np = face_utils.shape_to_np(shape)

        draw_facial_landmarks(image, shape_np, color=(0, 255, 0))

    return image

def draw_face_chip(image, color):
    return compute_facial_landmarks(image)

def draw_dot(image, color):
    cx, cy = math.floor(image.shape[1] * 0.95), math.floor(image.shape[1] * 0.05)
    rad = math.floor(image.shape[0] * 0.025)
    y, x = np.ogrid[-rad:rad, -rad:rad]
    image[cy-rad:cy+rad, cx-rad:cx+rad][x**2 + y**2 <= rad**2] = color
    return image

def diarize(path, out_path, model_path, filters=[draw_dot]):
    model = tf.keras.models.load_model(model_path)

    outdir = pathlib.Path("export/diarized/")
    outdir.mkdir(parents=True, exist_ok=True)

    temppath = outdir / (str(path.stem) + ".temp.mp4")
    capture = cv2.VideoCapture(str(path.absolute()))

    out = None

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
            prediction_window = np.expand_dims(prediction_window, axis=0)

            prediction = model.predict(prediction_window)

            label = classes[np.argmax(prediction[0])]
        else:
            label = None
            errors = False
        
        color = [0,0,0]
        if label == 'Idle':
            color = [255,0,0]
        elif label == 'Speak':
            color = [0,255,0]

        for frame in window:

            for func in filters:
                frame = func(frame, color=color)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if out is None:
                out = cv2.VideoWriter(str(temppath.absolute()), cv2.VideoWriter_fourcc(*'mp4v'), capture.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
            
            out.write(frame)
            cv2.waitKey(1)

    capture.release()
    out.release()

    try:
        cmd = [ 'ffmpeg', '-i', str(temppath.absolute()), '-i', str(path.absolute()), '-y', '-hide_banner', '-loglevel', 'error', '-nostats', 
                '-map', '0:0', '-map', '1:1', '-c:v', 'copy', '-c:a', 'copy', '-shortest', str(out_path.absolute())]
        returned_output = subprocess.call(cmd)
        temppath.unlink()
    except:
        temppath.replace(out_path)

    print("\r"+(" "*(os.get_terminal_size().columns-1)), end='\r')
    print("Done.", end='')

if __name__ == '__main__':
    if len(sys.argv) > 2:
        outdir =  pathlib.Path("export/diarized/")
        outdir.mkdir(parents=True, exist_ok=True)

        model = pathlib.Path(sys.argv[1]).resolve()
        path = pathlib.Path(sys.argv[2]).resolve()
        print("Processing file {}...".format(path.absolute()))

        start_time = time.perf_counter()
        if any(opt in sys.argv for opt in [ "-d", "--draw-face"]):
            out_path = outdir / (str(path.stem) + ".ol.mp4")
            diarize(path, out_path, model, filters=[draw_facial_reco, draw_dot])

        elif any(opt in sys.argv for opt in [ "-c", "--chip-face"]):
            out_path = outdir / (str(path.stem) + ".chip.mp4")
            diarize(path, out_path, model, filters=[draw_face_chip])

        else:
            out_path = outdir / (str(path.stem) + ".diarized.mp4")
            diarize(path, out_path, model, filters=[draw_dot])
        total_time = time.perf_counter() - start_time

        print(" Took {} seconds.".format(total_time))

    else:
        print("Diarize a video, producing a new video with an indicator of whether the person is speaking or not.")
        print("The output will be placed in a folder named export/diarized in the execution directory.")
        print("")
        print("Usage: ")
        print("   diarize.py <model> <video> [<options>]")
        print("")
        print("Options: ")
        print("   -c, --chip-face   Output face chips only.")
        print("   -d, --draw-face   Draw facial landmarking layer onto output.")
