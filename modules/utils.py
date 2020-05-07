import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from termcolor import colored
from math import floor

def print_progress(curr, total, end="\r"):
    bars = floor((curr/total)*20)*'■'
    dashes = colored((20 - floor((curr/total)*20))*'-', 'grey')
    progress = "[{}{}] {:.1f}%".format(bars, dashes, (curr / (total - 1)) * 100)
    print(progress, end=end)


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


def print_video_frames(video, step=2):
    f, axarr = plt.subplots(1, floor((len(video)) / step), sharey=True)
    f.set_figwidth(4 * floor((len(video)) / step))
    f.set_figheight(4)
    for i in range(0,  floor(len(video) / step)):
        fig = axarr[i]
        fig.text(0.5, -0.2, 'Frame {}/{}'.format(floor(i * step) + 1, len(video)), size=12, ha="center", transform=fig.transAxes)
        if len(video[i].shape) > 2:
            if video[i].shape[2] == 4:
                imgs = np.dsplit(video[floor(i * step)], np.array([3, 6]))
                fig.imshow(imgs[0])
                fig.imshow(imgs[1].squeeze(), cmap='seismic_alpha', vmin=-5000, vmax=5000)
            elif video[i].shape[2] == 3:
                fig.imshow(video[floor(i * step)])
            elif video[i].shape[2] == 1:
                fig.imshow(video[floor(i * step)].squeeze(), cmap='gray')
            else:
                imgs = np.dsplit(video[floor(i * step)], 2)
                fig.imshow(imgs[0].squeeze(), cmap='gray')
                fig.imshow(imgs[1].squeeze(), cmap='seismic_alpha', vmin=-5000, vmax=5000)
        else:
            fig.imshow(video[floor(i * step)])
    f.show()


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_roc_curve(fpr, tpr, roc_auc):
    figure = plt.figure(figsize=(8, 8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return figure