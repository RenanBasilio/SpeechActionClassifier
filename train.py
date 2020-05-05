from __future__ import absolute_import, division, print_function, unicode_literals

from modules.loader import get_classes, get_files_list

if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv3D, Dropout, SpatialDropout3D, MaxPooling3D, AveragePooling3D

    import numpy as np
    import pathlib
    import matplotlib.pyplot as plt
    import datetime
    import pandas as pd
    import shutil
    import io
    import termcolor

    from modules.generators import VideoDataGenerator
    from modules.utils import plot_confusion_matrix

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Tensorflow Version: ", tf.__version__)
    print("Num GPUs Available: ", len(gpus))

    #%% Set training parameters
    data_dir = pathlib.Path("./data")
    classes = get_classes(data_dir / "classes.txt")

    # Create an identifier for the model. Currently using time of training.
    export_base = pathlib.Path("export/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d-%H%M")))
    export_base.mkdir(parents=True, exist_ok=True)

    shutil.copy("train.py", (export_base / "train.py"))
    shutil.copy(data_dir / "splits.txt", (export_base / "splits.txt"))

    params = {
        'color_mode': 'landmarks',
        'optical_flow': False,
        'batch_size': 64,
        'shuffle': True,
        'classes': classes,
        'max_processes': 8
    }

    partition = get_files_list(data_dir)

    training_generator = VideoDataGenerator(partition['Train'], **params)
    validation_generator = VideoDataGenerator(partition['Test'], **params)

    #%% Build Keras model
    model = Sequential([
        Conv3D(16, (1, 3, 3), activation='relu', input_shape=(training_generator.dim)),
        MaxPooling3D((1,2,2)),
        Conv3D(24, (3, 3, 3), activation='relu'),
        Conv3D(48, (5, 7, 7), activation='relu'),
        MaxPooling3D((1,2,2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        AveragePooling3D((2,1,1)),
        MaxPooling3D((1,2,2)),
        SpatialDropout3D(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='sigmoid'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    #%%
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def log_confusion_matrix(epoch, logs):
        cm = np.zeros((2,2))

        # Use the model to predict the values from the validation dataset.
        for i in validation_generator:
            test_pred_raw = model.predict(i[0])
            cm += tf.math.confusion_matrix(np.argmax(i[1], axis=1), np.argmax(test_pred_raw, axis=1), num_classes=2)

        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm.numpy(), class_names=classes)
        cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    # %%
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(export_base.absolute()), histogram_freq=1)

    file_writer_cm = tf.summary.create_file_writer(str( (export_base).absolute() / "validation" ), filename_suffix=".cm")
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)

    checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=(str(export_base.absolute()) + "/model.best-loss.h5"), 
        monitor='val_loss', 
        save_best_only=True)

    checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=(str(export_base.absolute()) + "/model.best-accuracy.h5"), 
        monitor='val_accuracy', 
        save_best_only=True)

    history = model.fit(training_generator, validation_data=validation_generator, epochs=999, callbacks=[tensorboard_callback, cm_callback, checkpoint_loss, checkpoint_acc, earlystop])

    # %%
    pd.DataFrame.from_dict(history.history).to_csv((export_base / "history.csv"), index=False)
    model.save(export_base / "model.final.h5")

    print("Model data exported to {}".format(export_base))