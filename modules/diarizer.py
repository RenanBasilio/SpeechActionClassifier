import tensorflow as tf
import cv2 as cv2
import numpy as np
import pathlib
from modules.loader import compute_facial_landmarks

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

gauss_weights = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

# This commit strategy takes the sum of the elements for each column in the input array
# and chooses the column with the highest total, or -1 if both are equal.
def strat_sum(array):
    if len(array) > 0:
        sum_v = np.sum(array, axis=0)
        if sum_v[0][0] == sum_v[0][1]:
            return -1
        else:
            return np.argmax(sum_v[0])

# This commit strategy takes the frequency with which each class in the input array would
# be selected and chooses the most frequent class, or -1 if both are equal.
def strat_freq(array):
    if len(array) > 0:
        argc = np.zeros(len(array[0][0]))
        for e in array:
            argc[np.argmax(e)] += 1
        if argc[0] == argc[1]:
            return -1
        else:
            return np.argmax(argc)

# This commit strategy takes the frequency with which each class in the input array would
# be selected, weighted using a gaussian distribution, and chooses the most frequent class, 
# or -1 if both are equal.
def strat_gaussf(array):
    pass

class Diarizer():
    def __init__(self, model, commit_strategy="sum", shift=1):
        self.__predictor = tf.keras.models.load_model(model)
        self.__shift = shift

        if commit_strategy is "sum":
            self.__commit_strategy = strat_sum
        elif commit_strategy is "freq":
            self.__commit_strategy = strat_freq
        elif commit_strategy is "gaussf":
            self.__commit_strategy = strat_gaussf
        elif callable(commit_strategy):
            self.__commit_strategy = commit_strategy
        else:
            raise TypeError("commit_strategy can only be either a string in ['sum', 'freq', 'gaussf'] or a callable method taking a numpy 2d array and returning the integer index of the class.")

    def diarize(self, video, progress_callback=None):
        capture = cv2.VideoCapture(video)

        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        curr = 0

        done = False
        loaded = 0
        window = []
        predictions = []
        segments = []
        while not done:
            # If not enough frames loaded for prediction, load more
            while len(window) < 15:

                # Handle progress callback
                if progress_callback is not None:
                    progress_callback(curr, length)

                # Load frames
                curr += 1
                ret, frame = capture.read()
                if ret:
                    frame = self.__preprocess_frame(frame)
                    if frame is not None:
                        window.append(frame)
                        predictions.append([])
                    else:
                        # If a frame failed to load or preprocess, commit the ones before it if possible
                        segments = self.__commit_results(segments, predictions)
                        segments.append(-1)
                        predictions.clear()
                        window.clear()
                else:
                    done = True
                    break

            # Once enough frames loaded, predict
            if len(window) == 15:
                prediction = self.__predictor.predict(np.expand_dims(np.asarray(window), axis=0))
                for frame in predictions:
                    frame.append(prediction)

            # Commit the first 'shift' entries and shift the arrays
            segments = self.__commit_results(segments, predictions[:self.__shift])
            window = window[self.__shift:]
            predictions = predictions[self.__shift:]

            # If done, commit all remaining entries
            if done:
                segments = self.__commit_results(segments, predictions)

        return segments

    def __commit_results(self, transcription, prediction_window):
        # For each frame in the prediction window
        for result_array in prediction_window:
            transcription.append(self.__commit_strategy(result_array))

        return transcription

    def __preprocess_frame(self, frame):
        #frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frame = compute_facial_landmarks(frame)
        if frame is not None:
            frame = np.expand_dims(frame, axis=2)
        return frame
            