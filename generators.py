from loader import load_video_as_ndarray, get_label
import numpy as np
import tensorflow as tf

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