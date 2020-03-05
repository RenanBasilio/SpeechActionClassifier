import cv2 as cv2
from termcolor import colored
from math import floor

def print_progress(curr, total):
    bars = floor((curr/total)*20)*'■'
    dashes = colored((20 - floor((curr/total)*20))*'-', 'grey')
    print("\r[{}{}]".format(bars, dashes), end="")


def write_tee(file, message, verbose=True):
    file.write(message)
    if verbose:
        print(message, end='')

def draw_facial_landmarks(canvas, shape, color=(0,0,0)):
    cv2.polylines(canvas, [shape[0:16]], False, color, 0, cv2.LINE_8) # Boundaries
    cv2.polylines(canvas, [shape[17:21]], False, color, 0, cv2.LINE_8) # Left Eyebrow
    cv2.polylines(canvas, [shape[22:26]], False, color, 0, cv2.LINE_8) # Right Eyebrow
    cv2.polylines(canvas, [shape[27:30]], False, color, 0, cv2.LINE_8) # Nose Bridge
    cv2.polylines(canvas, [shape[31:35]], False, color, 0, cv2.LINE_8) # Nose
    cv2.polylines(canvas, [shape[36:41]], True, color, 0, cv2.LINE_8) # Left Eye
    cv2.polylines(canvas, [shape[42:47]], True, color, 0, cv2.LINE_8) # Right Eye
    cv2.polylines(canvas, [shape[48:59]], True, color, 0, cv2.LINE_8) # Mouth Outer
    cv2.polylines(canvas, [shape[60:67]], True, color, 0, cv2.LINE_8) # Mouth Inner
    return canvas