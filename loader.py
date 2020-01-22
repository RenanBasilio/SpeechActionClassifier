import cv2 as cv2
import numpy as np
import pathlib

from termcolor import cprint

# Get class names from classes.txt
data_dir = pathlib.Path("data/")
CLASS_NAMES = []
with open(data_dir/"classes.txt") as classes_file:
    CLASS_NAMES = classes_file.read().splitlines()

# Gets the label of a file from the path
def get_label(path):
    return path.parts[-2]


def get_label_from_index(idx):
    return CLASS_NAMES[idx]

# Load a frame from an opencv capture and returns it as a numpy ndarray
def load_frame_as_ndarray(cap):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frame = frame.astype('float32')
        frame *= 1.0/255.0
        return frame
    else:
        return None


# Loads the video file in the provided path as an array of frames
def load_video_as_ndarray(path):
    if path.is_file():
        #print("Loading file {}...".format(path))
        cap = cv2.VideoCapture(str(path.absolute()))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames > 15:
            cprint("WARNING: Video file {} contains more than 15 frames (was: {}). Extra frames will be ignored.".format(path, n_frames), 'yellow')
        elif n_frames < 15:
            cprint("WARNING: Video file {} contains less than 15 frames (was: {}). Last frame will be duplicated.".format(path, n_frames), 'yellow')

        frames = []
        for i in range(15):
            frame = load_frame_as_ndarray(cap)
            if frame is not None:
                frames.append(frame)
            else:
                frames.append(frames[i-1])

        frames = np.asarray(frames)
        return frames
    else:
        cprint("ERROR: File does not exist '{}'".format(path), 'red')
        return None