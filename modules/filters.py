import math
import dlib
import numpy as np
from imutils import face_utils
from modules.utils import draw_facial_landmarks
from modules.loader import compute_facial_landmarks

face_predictor = None
face_detector = None

# Computes facial landmarks from 'image' and overlays the extracted faces onto it.
# Returns the image with the overlay. 
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

# Computes facial landmarks from 'image'.
# Returns an image drawn from the computed landmarks.
def draw_face_chip(image, color):
    return compute_facial_landmarks(image)

# Draws a colored dot at the top right corner of 'image' with 'color'.
# Returns the modified image.
def draw_dot(image, color):
    cx, cy = math.floor(image.shape[1] * 0.95), math.floor(image.shape[1] * 0.05)
    rad = math.floor(image.shape[0] * 0.025)
    y, x = np.ogrid[-rad:rad, -rad:rad]
    image[cy-rad:cy+rad, cx-rad:cx+rad][x**2 + y**2 <= rad**2] = color
    return image
