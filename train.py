from __future__ import absolute_import, division, print_function, unicode_literals

if __name__ == '__main__':
    import tensorflow as tf
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
    from modules.loader import get_classes, get_files_list

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Tensorflow Version: ", tf.__version__)
    print("Num GPUs Available: ", len(gpus))

    import config.model as modelcfg
    if pathlib.Path("config/envconfig.py").is_file():
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

    shutil.copy("./config/model.py", (export_base / "model.py"))
    shutil.copy(data_dir / "splits.txt", (export_base / "splits.txt"))

    model.compile(
        optimizer='nadam', 
        loss=tf.losses.CategoricalCrossentropy(), 
        metrics=[
            tf.metrics.CategoricalCrossentropy(name="basics/loss"),
            tf.metrics.CategoricalAccuracy(name="basics/accuracy"),
            tf.metrics.Precision(name="confusion/precision"),
            tf.metrics.Recall(name="confusion/recall"),
            tf.metrics.Precision(class_id=0, name="confusion/precision/idle"),
            tf.metrics.Precision(class_id=1, name="confusion/precision/speak"),
            tf.metrics.Recall(class_id=0, name="confusion/recall/idle"),
            tf.metrics.Recall(class_id=1, name="confusion/recall/speak"),
            tf.metrics.AUC(name="basics/auc")
        ]
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

    def log_metrics(epoch, logs):
        if 'val_loss' in logs.keys():
            cm = np.zeros((2,2))
            test_true = []
            test_score = []

            # Use the model to predict the values from the validation dataset.
            for i in validation_generator:
                test_true.extend(np.argmax(i[1], axis=1))
                test_score.extend(model.predict(i[0]))
                
            with train_writer.as_default():
                # F1 Score
                f1_score = 2 * ((logs['confusion/precision'] * logs['confusion/recall']) / (logs['confusion/precision'] + logs['confusion/recall']))
                tf.summary.scalar("epoch_confusion/f1_score", f1_score, step=epoch)

            with val_writer.as_default():
                # F1 Score
                f1_score = 2 * ((logs['val_confusion/precision'] * logs['val_confusion/recall']) / (logs['val_confusion/precision'] + logs['val_confusion/recall']))
                tf.summary.scalar("epoch_confusion/f1_score", f1_score, step=epoch)

                # Confusion Matrix
                cm = tf.math.confusion_matrix(test_true, np.argmax(test_score, axis=1), num_classes=2)
                figure = plot_confusion_matrix(cm.numpy(), class_names=modelcfg.classes)
                cm_image = plot_to_image(figure)
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)

                # ROC curve
                fpr, tpr, thresholds = metrics.roc_curve(test_true, np.array(test_score)[:,1], pos_label=1)
                figure = plot_roc_curve(fpr, tpr, logs['val_basics/auc'])
                roc_image = plot_to_image(figure)
                tf.summary.image("ROC Curve", roc_image, step=epoch)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(export_base.absolute()), histogram_freq=1)

    train_writer = tf.summary.create_file_writer(str( (export_base).absolute() / "train" ), filename_suffix=".custom.v2")
    val_writer = tf.summary.create_file_writer(str( (export_base).absolute() / "validation" ), filename_suffix=".custom.v2")

    metrics_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metrics)

    checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=(str(export_base.absolute()) + "/model.best-loss.h5"), 
        monitor='val_loss', 
        save_best_only=True)

    checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=(str(export_base.absolute()) + "/model.best-accuracy.h5"), 
        monitor='val_basics/accuracy', 
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