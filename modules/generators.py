from modules.loader import load_video_as_ndarray
import numpy as np
import tensorflow as tf
import multiprocessing
from functools import partial

# Classe que carrega dados de maneira assíncrona para o preditor
# Necessário para reduzir o custo de memória do treinamento do modelo
# Código adaptado a partir de https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly (https://github.com/afshinea/keras-data-generator)
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_Entries, max_processes=6, batch_size=4, color_mode='rgb', optical_flow=False, classes=[], shuffle=True):
        self.batch_size = batch_size
        self.entries = []
        for i in list_Entries:
            self.entries.append((i, True))
            self.entries.append((i, False))
        self.color_mode = color_mode
        self.optical_flow = optical_flow

        test_video = load_video_as_ndarray(self.entries[0][0].filename, color_mode=self.color_mode, optical_flow=self.optical_flow, warnings=False)
        self.dim = test_video.shape

        self.classes = classes
        self.n_classes = len(classes)
        self.shuffle = shuffle
        self.procpool = multiprocessing.Pool(processes=max_processes)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.entries))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, entries_temp):
        X = None
        y = np.empty((self.batch_size), dtype=int)

        jobs = []
        for i, entry in enumerate(entries_temp):
            jobs.append((entry[0].filename, self.color_mode, entry[1], self.optical_flow, False, True))
            y[i] = self.classes.index(entry[0].classname)

        X = self.procpool.starmap(load_video_as_ndarray, jobs)
        X = np.array(X)

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.entries) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        entries_temp = [self.entries[k] for k in indexes]

        X, y = self.__data_generation(entries_temp)

        return (X, y)