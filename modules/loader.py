import pathlib

import dlib
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from termcolor import cprint, colored
from imutils import face_utils
from math import floor, ceil
from matplotlib.colors import LinearSegmentedColormap, DivergingNorm

from modules.utils import draw_facial_landmarks

errors = ['Partial', '!! BAD', 'Inaudible', 'Maybe']

# Get class names from classes.txt
# data_dir = pathlib.Path("data/")

face_detector = None
face_predictor = None

color_array = plt.get_cmap('RdYlBu')(range(256))
color_array[0:128,-1] = np.linspace(1.0,0.0,128)
color_array[128:256,-1] = np.linspace(0.0,1.0,128)

map_object = LinearSegmentedColormap.from_list(name='seismic_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)

class Entry(object):
    def __init__(self, filename, classname):
        self.filename = filename
        self.classname = classname

def get_classes(path):
    class_names = []
    with open(path) as classes_file:
        class_names = classes_file.read().splitlines()
    return class_names

def get_splits(splits_file):
    splits = {}
    with open(splits_file) as splits_file:
        for line in splits_file.read().splitlines():
            name, members = line.split(maxsplit=1)
            members = members.split(" ")
            splits[name] = members
    return splits

def get_files_list(path, verbose=False):
    path = pathlib.Path(path).resolve()
    
    splits_file = path / "splits.txt"
    if not splits_file.exists():
        raise FileNotFoundError("No splits.txt in dataset root.")
    classes_file = path / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError("No classes.txt in dataset root.")

    splits = get_splits(splits_file)
    classes = get_classes(classes_file)

    files = {}

    for split in splits.keys():

        loaded = 0

        files[split] = []
        for fileset in splits[split]:

            loaded_tags = {}

            data_path = path / "unsorted" / fileset
            manual_classes = data_path / "_classes.csv"
            segments_path = data_path / "segments"

            classes_ds = pd.read_csv(manual_classes, dtype={'id': np.int32})

            for index, row in enumerate(classes_ds.itertuples()):
                if row.tag in classes \
                    and (pd.isnull(row.notes) or (not any([x in row.notes for x in errors]) and not "Speaker Change" in row.notes)) \
                    and (pd.notnull(row.validated) and row.validated):
                    file_path = segments_path / "{:06d}.mp4".format(row.id)
                    if file_path.exists():
                        files[split].append(Entry(file_path, row.tag))
                        loaded += 1
                        if row.tag not in loaded_tags.keys():
                            loaded_tags[row.tag] = 1
                        else:
                            loaded_tags[row.tag] += 1

            if verbose:
                tag_count = 0
                tag_count_str = ""
                for key in loaded_tags.keys():
                    if tag_count_str != "":
                        tag_count_str += ", "
                    tag_count_str += "{}: {}".format(key, loaded_tags[key])
                    tag_count += loaded_tags[key]
                print("Loaded {} entries from set {} ( {} )".format(tag_count, fileset, tag_count_str))

        if verbose:
            cprint("Loaded {} entries for split {}.".format(loaded, split), "cyan")

    return files

# Gets the label of a file from the path
def get_label(path):
    return path.parts[-2]

def get_label_from_index(classes_file, idx):
    return get_classes(classes_file)[idx]

# Load a frame from an opencv capture and returns it as a numpy ndarray
def load_frame_as_ndarray(cap, colormode):
    ret, frame = cap.read()
    if ret:
        if colormode == 'rgb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif colormode == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif colormode == 'landmarks':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = compute_facial_landmarks(frame)
        elif colormode == 'raw':
            return frame
        else:
            cprint("\r\nERROR: Invalid colormode specified.", 'red')
            return None
        frame = np.asarray(frame)
        frame = frame.astype('float32')
        frame *= 1.0/255.0
        return frame
    else:
        return None

def compute_facial_landmarks(image):
    global face_predictor
    global face_detector

    if face_predictor is None:
        #face_detector = dlib.cnn_face_detection_model_v1("resources/mmod_human_face_detector.dat")
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

    faces = face_detector(image, 1)
    blank_image = np.full(image.shape, 255, np.uint8)
    face_chip = None
    for (i, face) in enumerate(faces):
        face_chip = np.full((150, 150, 3), 255, np.uint8)
        #shape = face_predictor(image, face.rect)
        shape = face_predictor(image, face)
        shape_np = face_utils.shape_to_np(shape)

        #for (x, y) in shape_np:
            #cv2.circle(blank_image, (x,y), 0, (0, 0, 0))

        draw_facial_landmarks(blank_image, shape_np)

        face_chip = dlib.get_face_chip(blank_image, shape, 150, 0.33)
        face_chip = cv2.cvtColor(face_chip, cv2.COLOR_RGB2GRAY)
        
    return face_chip

# Código adaptado a partir de em https://github.com/ferreirafabio/video2tfrecord
def compute_dense_optical_flow(prev_image, current_image):
    old_shape = current_image.shape
    assert current_image.shape == old_shape

    if len(current_image.shape) > 2:
        prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
    else:
        prev_image_gray = prev_image
        current_image_gray = current_image

    hsv = np.zeros((prev_image.shape[0], prev_image.shape[1], 3), dtype=prev_image.dtype)
    hsv[..., 1] = 255
    flow = None
    flow = cv2.calcOpticalFlowFarneback(prev=prev_image_gray,
                                        next=current_image_gray, flow=flow,
                                        pyr_scale=0.8, levels=15, winsize=5,
                                        iterations=10, poly_n=5, poly_sigma=0,
                                        flags=10)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    of = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    return of

# Loads the video file in the provided path as an array of frames
def load_video_as_ndarray(path, color_mode='rgb', mirror=True, optical_flow=False, warnings=True):
    path = pathlib.Path(path)

    if path.is_file():
        #print("Loading file {}...".format(path))
        cap = cv2.VideoCapture(str(path.absolute()))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames > 15 and warnings != False:
            cprint("WARNING: Video file {} contains more than 15 frames (was: {}). Extra frames will be ignored.".format(path, n_frames), 'yellow')
            if warnings is 'except':
                raise Exception("Invalid video data.") 
        elif n_frames < 15 and warnings:
            cprint("WARNING: Video file {} contains less than 15 frames (was: {}). Last frame will be duplicated.".format(path, n_frames), 'yellow')
            if warnings is 'except':
                raise Exception("Invalid video data.") 

        frames = []
        last_frame = None
        for i in range(15):
            frame = load_frame_as_ndarray(cap, color_mode)
            if frame is not None:
                if optical_flow:
                    if last_frame is not None:
                        flow = compute_dense_optical_flow(last_frame, frame)
                    else:
                        flow = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                    last_frame = frame

                if len(frame.shape) < 3:
                    frame = np.expand_dims(frame, axis=2)

                if optical_flow:
                    frame = np.concatenate((frame, np.expand_dims(flow, axis=2)), axis=2)

                if mirror:
                    frame = np.fliplr(frame)

                frames.append(frame)
            else:
                frames.append(frames[i-1])

        frames = np.asarray(frames)

        return frames
    else:
        raise FileNotFoundError("File not found: {}".format(path))