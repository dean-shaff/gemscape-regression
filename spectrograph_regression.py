import logging

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as tf_preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import preprocessing
import settings


def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.expand_dims(tf.abs(spectrogram), -1)
    return spectrogram


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]

    X = range(width)
    Y = range(height)

    ax.pcolormesh(X, Y, log_spec)

def make_model(ds):

    for (spec, target) in ds.take(1):
        input_shape = spec.shape
        output_shape = target.shape[0]

    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    # num_labels = len(commands)

    norm_layer = tf_preprocessing.Normalization()
    norm_layer.adapt(ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        tf_preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(output_shape),
    ])

    model.summary()

    return model


def main():

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("eyed3").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    ds = preprocessing.build_dataset_from_dirs(settings.data_dir)
    spec_ds = ds.map(lambda waveform, target: (get_spectrogram(waveform), target))

    # batch_size = 2
    # spec_ds = spec_ds.batch(batch_size)

    # spec_features = features.map(func)
    # spec_features = features.map(get_spectrogram)

    model = make_model(spec_ds)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    batch_size = 2
    spec_ds = spec_ds.batch(batch_size)

    history = model.fit(
        spec_ds,
        # validation_data=val_ds,
        epochs=10
        # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    # fig, axes = plt.subplots(1, 2, figsize=(10, 12))
    # for (spec, target) in spec_ds.take(1):
    #     pass

    # axes[0].set_title(" ".join([str(val) for val in target.numpy()]))
    # axes[0].plot(waveform.numpy())
    # axes[0].set_yticks(np.arange(-1.2, 1.2, 0.2))
    #
    # plot_spectrogram(spec.numpy(), axes[1])
    #
    # plt.show()




if __name__ == "__main__":
    main()
