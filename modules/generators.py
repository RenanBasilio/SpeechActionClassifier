from modules.loader import load_video_as_ndarray
import numpy as np
import multiprocessing
import pathlib
import hashlib
import pickle
from termcolor import cprint
from tensorflow.keras.utils import Sequence, to_categorical

cache_dir = pathlib.Path("__datacache__/")
cache_dir.mkdir(parents=True, exist_ok=True)

# Classe que carrega dados de maneira assíncrona para o preditor
# Necessário para reduzir o custo de memória do treinamento do modelo
# Código adaptado a partir de https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly (https://github.com/afshinea/keras-data-generator)
class VideoDataGenerator(Sequence):
    def __init__(self, list_Entries, max_processes=6, batch_size=4, color_mode='rgb', optical_flow=False, classes=[], shuffle=True, enable_cache=True):
        self.version = 2
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
        self.enable_cache = enable_cache
        self.procpool = multiprocessing.Pool(processes=max_processes)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.entries))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, entries_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        jobs = ([],[])
        for i, entry in enumerate(entries_temp):
            data = None
            if self.enable_cache:
                data = self.__read_cache(self.__make_cache_path(entry[0].filename.absolute(), entry[1]), str(entry[0].filename.absolute()))

            if data is None:
                jobs[0].append(i)
                jobs[1].append((entry[0].filename, self.color_mode, entry[1], self.optical_flow, False))
            else:
                X[i,] = data

            y[i] = self.classes.index(entry[0].classname)

        results = self.procpool.starmap(load_video_as_ndarray, jobs[1])
        
        j = 0
        for i in jobs[0]:
            X[i,] = results[j]
            if self.enable_cache:
                self.__write_cache(self.__make_cache_path(entries_temp[i][0].filename.absolute(), entries_temp[i][1]), str(entries_temp[i][0].filename.absolute()), results[j])
            j += 1

        return X, to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.entries) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        entries_temp = [self.entries[k] for k in indexes]

        X, y = self.__data_generation(entries_temp)

        return (X, y)

    def __make_cache_path(self, path, mirror):
        cache_file_name = hashlib.sha1((str(path) + self.color_mode + ("opticalflow" if self.optical_flow else "") + ("mirror" if mirror else "")).encode()).hexdigest()
        return (cache_dir / cache_file_name).with_suffix(".cache")

    def __read_cache(self, path, uid):
        if path.is_file():
            with open(path, mode='rb') as cache_file:
                file_data = pickle.load(cache_file)
                if file_data['version'] != self.version:
                    cprint("WARNING: Cached version not created by this API version. Regenerating cache...")
                elif file_data['uid'] != uid:
                    cprint("WARNING: UID in cached data does not match current UID.\nGOT: '{}' EXPECTED: '{}'".format(file_data['uid'], uid), 'yellow')
                else:
                    return file_data['data']
        return None

    def __write_cache(self, path, uid, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        file_data = {
            "uid": uid,
            "version": self.version,
            "data" : data
        }
        with open(path, mode='wb') as cache_file:
            pickle.dump(file_data, cache_file)