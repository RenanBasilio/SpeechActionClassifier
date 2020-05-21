from __future__ import absolute_import, division, print_function, unicode_literals

from modules.loader import get_classes, get_files_list

if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow_addons as tfa
    from sklearn import metrics

    import numpy as np
    import pathlib
    import matplotlib.pyplot as plt
    import datetime
    import pandas as pd
    import shutil
    import io
    import termcolor

    from modules.generators import VideoDataGenerator
    from modules.utils import plot_confusion_matrix, plot_roc_curve

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Tensorflow Version: ", tf.__version__)
    print("Num GPUs Available: ", len(gpus))

    import config.model as modelcfg
    if pathlib.Path("envconfig.py").is_file():
        print("Using environment confuguration from envconfig.py")
        import config.envconfig as env
    else:
        print("Using default environment configuration.")
        import config.envconfig_default as env

    data_dir = pathlib.Path(env.dataset_directory)

    # Create an identifier for the model. Currently using time of training.
    export_base = pathlib.Path(env.export_directory) / "{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
    export_base.mkdir(parents=True, exist_ok=True)

    shutil.copy("train.py", (export_base / "train.py"))
    shutil.copy(data_dir / "splits.txt", (export_base / "splits.txt"))

    print("Loading entries...")
    partition = get_files_list(data_dir, verbose=True)

    training_generator = VideoDataGenerator(partition['Train'], 
        classes=modelcfg.classes, 
        optical_flow=modelcfg.optical_flow,
        color_mode=modelcfg.colormode, 
        **env.train_params
    )
        
    validation_generator = VideoDataGenerator(partition['Test'], 
        classes=modelcfg.classes, 
        color_mode=modelcfg.colormode, 
        optical_flow=modelcfg.optical_flow,
        **env.val_params
    )
    
    model = modelcfg.get_model((training_generator.dim))

    shutil.copy("model.py", (export_base / "model.py"))
    shutil.copy(data_dir / "splits.txt", (export_base / "splits.txt"))

    model.compile(
        optimizer='nadam', 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy',
            tf.metrics.AUC(),
            tfa.metrics.F1Score(num_classes=len(modelcfg.classes))]
    )

    model.summary()

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

    def log_image_metrics(epoch, logs):
        if 'val_loss' in logs.keys():
            cm = np.zeros((2,2))
            test_true = []
            test_score = []

            # Use the model to predict the values from the validation dataset.
            for i in validation_generator:
                test_true.extend(np.argmax(i[1], axis=1))
                test_score.extend(model.predict(i[0]))
                
            # Confusion Matrix
            cm = tf.math.confusion_matrix(test_true, np.argmax(test_score, axis=1), num_classes=2)

            figure = plot_confusion_matrix(cm.numpy(), class_names=modelcfg.classes)
            cm_image = plot_to_image(figure)

            with file_writer_cm.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)

            # ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(test_true, np.array(test_score)[:,1], pos_label=1)

            figure = plot_roc_curve(fpr, tpr, logs['val_auc'])
            roc_image = plot_to_image(figure)

            with file_writer_roc.as_default():
                tf.summary.image("ROC Curve", roc_image, step=epoch)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(export_base.absolute()), histogram_freq=1)

    file_writer_cm = tf.summary.create_file_writer(str( (export_base).absolute() / "validation" ), filename_suffix=".cm")
    file_writer_roc = tf.summary.create_file_writer(str( (export_base).absolute() / "validation" ), filename_suffix=".roc")

    metrics_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_image_metrics)

    checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=(str(export_base.absolute()) + "/model.best-loss.h5"), 
        monitor='val_loss', 
        save_best_only=True)

    checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=(str(export_base.absolute()) + "/model.best-accuracy.h5"), 
        monitor='val_accuracy', 
        save_best_only=True)

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)

    history = model.fit(
        training_generator, 
        validation_data=validation_generator, 
        epochs=999, 
        callbacks=[tensorboard_callback, metrics_callback, checkpoint_loss, checkpoint_acc, earlystop]
    )

    pd.DataFrame.from_dict(history.history).to_csv((export_base / "history.csv"), index=False)
    model.save(export_base / "model.final.h5")

    print("Model data exported to {}".format(export_base))