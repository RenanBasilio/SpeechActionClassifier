from modules.loader import load_video_as_ndarray, get_label
import numpy as np
import tensorflow as tf

# Classe que carrega dados de maneira assíncrona para o preditor
# Necessário para reduzir o custo de memória do treinamento do modelo
# Código adaptado a partir de https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly (https://github.com/afshinea/keras-data-generator)
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=4, color_mode='rgb', optical_flow=False, classes=[], shuffle=True):
        self.batch_size = batch_size
        self.IDs = []
        for i in list_IDs:
            self.IDs.append((i, True))
            self.IDs.append((i, False))
        self.color_mode = color_mode
        self.optical_flow = optical_flow

        test_video = load_video_as_ndarray(self.IDs[0][0], color_mode=self.color_mode, optical_flow=self.optical_flow, warnings=False)
        self.dim = test_video.shape

        self.classes = classes
        self.n_classes = len(classes)
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, IDs_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(IDs_temp):
            X[i,] = load_video_as_ndarray(ID[0], mirror=ID[1], color_mode=self.color_mode, optical_flow=self.optical_flow, warnings=False)
            
            y[i] = self.classes.index(get_label(ID[0]))

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        IDs_temp = [self.IDs[k] for k in indexes]

        X, y = self.__data_generation(IDs_temp)

        return (X, y)
