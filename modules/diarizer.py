import cv2 as cv2
import numpy as np
import pathlib
import math
import functools
from modules.loader import compute_facial_landmarks
from tensorflow.keras.models import load_model

class Segment():
    def __init__(self, value=None, frame_onset=None, frame_rate=30):
        self.value = value
        self.frame_onset = frame_onset
        self.frame_duration = 0
        self.frame_rate = frame_rate
        self.__frame_confidences = []
        self.__confidence = None

    @property
    def frame_confidences(self):
        return self.__frame_confidences

    @property
    def confidence(self):
        if self.__confidence is None:
            self.__confidence = np.mean(self.__frame_confidences)
        return self.__confidence

    @property
    def onset(self):
        return self.frame_onset / self.frame_rate

    @property
    def duration(self):
        return self.frame_duration / self.frame_rate

    def extend(self, confidences):
        self.__confidence = None
        if isinstance(confidences, list):
            self.frame_duration += len(confidences)
            self.frame_confidences.extend(confidences)
        else:
            self.frame_duration += 1
            self.frame_confidences.append(confidences)
    
    def __str__(self):
        return "CLASS: {} | ONSET: {:.3f} s | DURATION: {:.3f} s | CONFIDENCE: {:.3f}".format(self.value, self.onset, self.duration, self.confidence)

class Diarizer():
    def __init__(self, model, commit_strategy="mean", shift=1, classes=[]):
        assert isinstance(shift, int) and shift > 0

        self.__classes = classes
        self.__predictor = load_model(model, compile=False)
        self.__shift = shift

        if commit_strategy == "mean":
            self.__commit_strategy = strat_mean
        elif commit_strategy == "median":
            self.__commit_strategy = strat_median
        elif commit_strategy == "freq":
            self.__commit_strategy = strat_freq
        elif commit_strategy == "gaussw":
            gauss_weights = init_gauss_weights(math.floor(15 / shift), (-2, 2)) 
            self.__commit_strategy = functools.partial(strat_weighted, weights=gauss_weights)
        elif commit_strategy == "gaussf":
            gauss_weights = init_gauss_weights(math.floor(15 / shift), (-2, 2)) 
            self.__commit_strategy = functools.partial(strat_weightedfreq, weights=gauss_weights)
        elif callable(commit_strategy):
            self.__commit_strategy = commit_strategy
        else:
            raise ValueError("commit_strategy can only be either a string in ['mean', 'median', 'freq', 'gaussw', 'gaussf'] or a callable method taking a numpy 2d array and returning a numpy array of confidences for each class.")

    def diarize(self, video, progress_callback=None):
        capture = cv2.VideoCapture(video)

        vid_props = {
            "FRAME_COUNT": capture.get(cv2.CAP_PROP_FRAME_COUNT),
            "FRAME_RATE": capture.get(cv2.CAP_PROP_FPS)
        }
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
                    progress_callback(curr, vid_props['FRAME_COUNT'])

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
                        predictions.append(np.zeros((1, 2)))
                        segments = self.__commit_results(segments, predictions, vid_props)
                        predictions.clear()
                        window.clear()
                else:
                    done = True
                    break

            # Once enough frames loaded, predict
            if len(window) == 15:
                prediction = self.__predictor.predict(np.expand_dims(np.asarray(window), axis=0))
                for frame in predictions:
                    frame.extend(prediction)

            # Commit the first 'shift' entries and shift the arrays
            segments = self.__commit_results(segments, predictions[:self.__shift], vid_props)
            window = window[self.__shift:]
            predictions = predictions[self.__shift:]

            # If done, commit all remaining entries
            if done:
                segments = self.__commit_results(segments, predictions, vid_props)

        return segments

    def __commit_results(self, transcription, prediction_window, vid_props):
        # For each frame in the prediction window
        for result_array in prediction_window:
            if len(result_array) > 0:
                # Compute predictions for each class based on the commit strategy
                confidences = self.__commit_strategy(result_array)
                if confidences[0] == confidences[1]:
                    result = None
                    confidence = 0
                else:
                    result = self.__classes[np.argmax(confidences)]
                    confidence = confidences[np.argmax(confidences)]
            else:
                result = None
                confidence = 0

            if len(transcription) == 0 or transcription[-1].value != result:
                # If first segment or not matching the previous value, start a new segment
                transcription.append(Segment(
                    result, 
                    frame_onset=(0 if len(transcription) == 0 else (transcription[-1].frame_onset + transcription[-1].frame_duration)),
                    frame_rate=vid_props['FRAME_RATE'],
                ))
            transcription[-1].extend(confidence)

        return transcription

    def __preprocess_frame(self, frame):
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frame = compute_facial_landmarks(frame)
        if frame is not None:
            frame = np.expand_dims(frame, axis=2)
        return frame

# This commit strategy takes the sum of the elements for each column in the input array
# and chooses the column with the highest total, or -1 if both are equal.
def strat_mean(array):
    if len(array) > 0:
        mean_v = np.mean(array, axis=0)
        return mean_v

def strat_median(array):
    if len(array) > 0:
        med_v = array[math.floor(len(array)/2)]
        return med_v

# This commit strategy takes the frequency with which each class in the input array would
# be selected and chooses the most frequent class, or -1 if both are equal.
def strat_freq(array):
    if len(array) > 0:
        argc = np.zeros(len(array[0]))
        for e in array:
            argc[np.argmax(e)] += 1
        argc = np.asarray(argc) / len(array)
        return argc

# This commit strategy assigns a weight to the probability of each class in the input array,
# and chooses the class with the highest weighted probability
def strat_weighted(array, weights):
    if len(array) > 0:
        weighted_sums = weights[:len(array)].dot(np.asarray(array))
        return weighted_sums

# This commit strategy assigns a weight to each predicted class in the input array, and
# chooses the most frequent class by the weighted sum of the predictions.
def strat_weightedfreq(array, weights):
    if len(array) > 0:
        argc = np.zeros(len(array[0]))
        for index, e in enumerate(array):
            if not np.all(e == e[0]):
                argc[np.argmax(e)] += weights[index]
        return argc

# This function computes a set of gaussian weights centered on x=0 and with standard deviation sigma=1. 
# The returned weights are distributed linearly within the given span and scaled to add up to 1.
def init_gauss_weights(count, span=(-2, 2)):
    gauss_weights = []
    X, step = np.linspace(span[0], span[1], count, retstep=True)
    for x in X:
        weight = math.pow(math.e, -(math.pow(x, 2)/2)) * step / math.sqrt(2 * math.pi)
        gauss_weights.append(weight)
    gauss_weights = np.asarray(gauss_weights)
    gauss_weights *= 1 / np.sum(gauss_weights)
    return gauss_weights