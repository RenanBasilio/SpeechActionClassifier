import cv2 as cv2
import numpy as np
import pathlib
import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
from enum import Enum

from termcolor import cprint
from imutils import face_utils
from math import floor, ceil
from matplotlib.colors import LinearSegmentedColormap

# Get class names from classes.txt
data_dir = pathlib.Path("data/")
cache_dir = data_dir / "__cache__/"
cache_dir.mkdir(parents=True, exist_ok=True)

face_detector = None
face_predictor = None

color_array = plt.get_cmap('hot')(range(256))
color_array[0:32,-1] = np.linspace(0.0,1.0,32)
color_array[223:255,-1] = np.linspace(1.0,0.0,32)
map_object = LinearSegmentedColormap.from_list(name='hot_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)

def get_classes(path):
    class_names = []
    with open(path) as classes_file:
        class_names = classes_file.read().splitlines()
    return class_names

# Gets the label of a file from the path
def get_label(path):
    return path.parts[-2]

def get_label_from_index(idx):
    return get_classes(data_dir / "classes.txt")[idx]

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
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

    rects = face_detector(image, 0)
    blank_image = np.full(image.shape, 255, np.uint8)
    face_chip = None
    for (i, rect) in enumerate(rects):
        shape = face_predictor(image, rect)
        shape_np = face_utils.shape_to_np(shape)

        #for (x, y) in shape_np:
            #cv2.circle(blank_image, (x,y), 0, (0, 0, 0))

        cv2.polylines(blank_image, [shape_np[0:16]], False, (0,0,0), 0, cv2.LINE_8) # Boundaries
        cv2.polylines(blank_image, [shape_np[17:21]], False, (0,0,0), 0, cv2.LINE_8) # Left Eyebrow
        cv2.polylines(blank_image, [shape_np[22:26]], False, (0,0,0), 0, cv2.LINE_8) # Right Eyebrow
        cv2.polylines(blank_image, [shape_np[27:30]], False, (0,0,0), 0, cv2.LINE_8) # Nose Bridge
        cv2.polylines(blank_image, [shape_np[31:35]], False, (0,0,0), 0, cv2.LINE_8) # Nose
        cv2.polylines(blank_image, [shape_np[36:41]], True, (0,0,0), 0, cv2.LINE_8) # Left Eye
        cv2.polylines(blank_image, [shape_np[42:47]], True, (0,0,0), 0, cv2.LINE_8) # Right Eye
        cv2.polylines(blank_image, [shape_np[48:59]], True, (0,0,0), 0, cv2.LINE_8) # Mouth Outer
        cv2.polylines(blank_image, [shape_np[60:67]], True, (0,0,0), 0, cv2.LINE_8) # Mouth Inner

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
    return cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

# Loads the video file in the provided path as an array of frames
def load_video_as_ndarray(path, colormode='rgb', optical_flow=False, warnings=True, enable_cache=True):
    path = pathlib.Path(path)
    cache_file_path = cache_dir / colormode / str(optical_flow) / path.relative_to(data_dir).with_suffix('.npy')

    if enable_cache and cache_file_path.is_file():
        return np.load(cache_file_path)

    if path.is_file():
        #print("Loading file {}...".format(path))
        cap = cv2.VideoCapture(str(path.absolute()))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames > 15 and warnings:
            cprint("\r\nWARNING: Video file {} contains more than 15 frames (was: {}). Extra frames will be ignored.".format(path, n_frames), 'yellow')
        elif n_frames < 15 and warnings:
            cprint("\r\nWARNING: Video file {} contains less than 15 frames (was: {}). Last frame will be duplicated.".format(path, n_frames), 'yellow')

        frames = []
        last_frame = None
        for i in range(15):
            frame = load_frame_as_ndarray(cap, colormode)
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

                frames.append(frame)
            else:
                frames.append(frames[i-1])

        frames = np.asarray(frames)
        if enable_cache:
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file_path, frames)

        return frames
    else:
        cprint("ERROR: File does not exist '{}'".format(path), 'red')
        return None

def print_video_frames(video, step=2):
    f, axarr = plt.subplots(1, floor((len(video)) / step), sharey=True)
    f.set_figwidth(4 * floor((len(video)) / step))
    f.set_figheight(4)
    for i in range(0,  floor(len(video) / step)):
        fig = axarr[i]
        fig.text(0.5, -0.2, 'Frame {}/{}'.format(floor(i * step) + 1, len(video)), size=12, ha="center", transform=fig.transAxes)
        if len(video[i].shape) > 2:
            if video[i].shape[2] >= 3:
                fig.imshow(video[floor(i * step)])
            elif video[i].shape[2] == 1:
                fig.imshow(video[floor(i * step)].squeeze(), cmap='gray')
            else:
                imgs = np.dsplit(video[floor(i * step)], 2)
                fig.imshow(imgs[0].squeeze(), cmap='gray')
                fig.imshow(imgs[1].squeeze(), cmap='hot_alpha')
        else:
            fig.imshow(video[floor(i * step)])
    f.show()

# Classe que carrega dados de maneira assíncrona para o preditor
# Necessário para reduzir o custo de memória do treinamento do modelo
# Código adaptado a partir de https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly (https://github.com/afshinea/keras-data-generator)
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=4, dim=(15, 240, 320), color_mode='rgb', optical_flow=False, classes=[], shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.color_mode = color_mode
        self.optical_flow = optical_flow

        if color_mode == 'rgb':
            self.n_channels = 3
        elif color_mode == 'gray':
            self.n_channels = 1
        if optical_flow:
            self.n_channels += 1

        self.classes = classes
        self.n_classes = len(classes)
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            if self.n_channels >= 3:
                X[i,] = load_video_as_ndarray(ID, colormode=self.color_mode, optical_flow=self.optical_flow, warnings=False)
            elif self.n_channels >= 1:
                X[i,] = load_video_as_ndarray(ID, colormode=self.color_mode, optical_flow=self.optical_flow, warnings=False)
            
            y[i] = self.classes.index(get_label(ID))

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return (X, y)

class FacialDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=4, dim=(15, 240, 320), n_channels=3, classes=[], shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.classes = classes
        self.n_classes = len(classes)
        self.shuffle = shuffle
        self.on_epoch_end()