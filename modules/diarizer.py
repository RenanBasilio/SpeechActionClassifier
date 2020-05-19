import cv2 as cv2
import numpy as np
import pathlib
import math
import functools
from modules.loader import compute_facial_landmarks
from tensorflow.keras.models import load_model

gauss_weights = np.array([
    0.000133830225764885,
    0.00111769186648184,
    0.00673409947523024,
    0.0292702560291608, 
    0.0917831739641082, 
    0.207629558115201, 
    0.338847935756788,
    0.398942280401433, 
    0.338847935756788, 
    0.207629558115201, 
    0.0917831739641082, 
    0.0292702560291608, 
    0.00673409947523025, 
    0.00111769186648185, 
    0.000133830225764885
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

# This commit strategy assigns a weight to the probability of each class in the input array,
# and chooses the class with the highest weighted probability
def strat_weighted(array, weights):
    if len(array) > 0:
        weighted_sums = gauss_weights[:len(array)].dot(array)
        if weighted_sums[0] == weighted_sums[1]:
            return -1
        else:
            return np.argmax(weighted_sums)

# This commit strategy assigns a weight to each predicted class in the input array, and
# chooses the most frequent class by the weighted sum of the predictions.
def strat_weightedfreq(array, weights):
    if len(array) > 0:
        argc = np.zeros(len(array[0][0]))
        for index, e in enumerate(array):
            argc[np.argmax(e)] += weights[index]
        if argc[0] == argc[1]:
            return -1
        else:
            return np.argmax(argc)

# This function computes a set of count gaussian weights centered on x=0 and with standard 
# deviation equal to sigma (default 2). The returned weights are distributed linearly within 
# 2 standard deviations.
def init_gauss_weights(count, sigma=2):
    gauss_weights = []
    for i in np.linspace(-(2*sigma), (2*sigma), count):
        weight = math.pow(math.e, -(math.pow(i, 2)/2(math.pow(sigma), 2))) / (sigma*math.sqrt(2 * math.pi))
        gauss_weights.append(weight)
    return np.asarray(gauss_weights)

class Diarizer():
    def __init__(self, model, commit_strategy="sum", shift=1):
        assert isinstance(shift, int) and shift > 0

        self.__predictor = load_model(model)
        self.__shift = shift

        if commit_strategy is "sum":
            self.__commit_strategy = strat_sum
        elif commit_strategy is "freq":
            self.__commit_strategy = strat_freq
        elif commit_strategy is "gaussf":
            self.__commit_strategy = functools.partial(strat_weightedfreq, weights=init_gauss_weights(15 / shift))
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
            