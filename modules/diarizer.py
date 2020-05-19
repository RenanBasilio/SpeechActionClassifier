import cv2 as cv2
import numpy as np
import pathlib
import math
import functools
from modules.loader import compute_facial_landmarks
from tensorflow.keras.models import load_model

class Segment():
    def __init__(self, value=None, onset=None, duration=None, confidence=None):
        self.value = value
        self.onset = onset
        self.duration = duration
        self.confidence = confidence
    
    def __str__(self):
        return "CLASS: {} | ONSET: {:.3f} s | DURATION: {:.3f} s | CONFIDENCE: {:.3f}".format(self.value, self.onset, self.duration, self.confidence)

class Diarizer():
    def __init__(self, model, commit_strategy="mean", shift=1):
        assert isinstance(shift, int) and shift > 0

        self.__predictor = load_model(model)
        self.__shift = shift

        if commit_strategy is "mean":
            self.__commit_strategy = strat_mean
        elif commit_strategy is "freq":
            self.__commit_strategy = strat_freq
        elif commit_strategy is "gaussf":
            gauss_weights = init_gauss_weights(math.floor(15 / shift)) 
            self.__commit_strategy = functools.partial(strat_weightedfreq, weights=gauss_weights)
        elif callable(commit_strategy):
            self.__commit_strategy = commit_strategy
        else:
            raise TypeError("commit_strategy can only be either a string in ['mean', 'freq', 'gaussf'] or a callable method taking a numpy 2d array and returning the integer index of the class.")

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
                        predictions.append([np.zeros((1, 2))])
                        segments = self.__commit_results(segments, predictions)
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

        segments = self.__finalize(segments, {'FPS':capture.get(cv2.CAP_PROP_FPS)})
        return segments

    def __commit_results(self, transcription, prediction_window):
        # For each frame in the prediction window
        for result_array in prediction_window:
            if len(result_array) > 0:
                # Compute predictions for each class based on the commit strategy
                confidences = self.__commit_strategy(result_array)
                if confidences[0] == confidences[1]:
                    result = -1
                    confidence = 0
                else:
                    result = np.argmax(confidences)
                    confidence = confidences[result]
            else:
                result = -1
                confidence = 0

            if len(transcription) > 0 and transcription[-1].value == result:
                # If result same as last segment type, extend it by one frame
                transcription[-1].duration += 1
                transcription[-1].confidence.append(confidence)
            else:
                # Otherwise start a new segment
                transcription.append(Segment(
                    result, 
                    onset=(0 if len(transcription) == 0 else (transcription[-1].onset + transcription[-1].duration)),
                    duration=1,
                    confidence=[confidence]
                ))
        return transcription

    def __preprocess_frame(self, frame):
        #frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frame = compute_facial_landmarks(frame)
        if frame is not None:
            frame = np.expand_dims(frame, axis=2)
        return frame

    def __finalize(self, transcription, props):
        for segment in transcription:
            segment.onset = segment.onset / props['FPS']
            segment.duration = segment.duration / props['FPS']
            segment.confidence = np.mean(segment.confidence)
        return transcription

# This commit strategy takes the sum of the elements for each column in the input array
# and chooses the column with the highest total, or -1 if both are equal.
def strat_mean(array):
    if len(array) > 0:
        mean_v = np.mean(array, axis=0)
        return mean_v[0]

# This commit strategy takes the frequency with which each class in the input array would
# be selected and chooses the most frequent class, or -1 if both are equal.
def strat_freq(array):
    if len(array) > 0:
        argc = np.zeros(len(array[0][0]))
        for e in array:
            argc[np.argmax(e)] += 1
        return argc

# This commit strategy assigns a weight to the probability of each class in the input array,
# and chooses the class with the highest weighted probability
def strat_weighted(array, weights):
    if len(array) > 0:
        weighted_sums = weights[:len(array)].dot(array)
        return weighted_sums

# This commit strategy assigns a weight to each predicted class in the input array, and
# chooses the most frequent class by the weighted sum of the predictions.
def strat_weightedfreq(array, weights):
    if len(array) > 0:
        argc = np.zeros(len(array[0][0]))
        for index, e in enumerate(array):
            if e[0][0] == e[0][1]:
                argc[0] = 0
                argc[1] = 0
            else:
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