import logging

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as tf_preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import preprocessing
import settings


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
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape)
    ])

    model.summary()

    return model

def reshape_waveform(waveform, samples_per_slice: int = 64):
    """
    Reshape waveform into (nslices, samples_per_slice) size array.
    If waveform's shape isn't an integer multiple of samples_per_slice,
    then add enough zeros to end such that it is.

    Args:
        waveform:
        samples_per_slice
    Returns:
        tensor
    """
    shape = tf.shape(waveform)
    nslices = tf.cast(tf.math.ceil(shape[0] / samples_per_slice), tf.int32)
    remaining = samples_per_slice*nslices - shape[0]
    waveform = tf.concat([waveform, tf.zeros(remaining, dtype=waveform.dtype)], 0)
    waveform = tf.reshape(waveform, (nslices, samples_per_slice))
    return waveform


def main():

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("eyed3").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    ds = preprocessing.build_dataset_from_dirs(
        settings.data_dir,
        desired_time=0.5,
        max_files=2
    )
    rnn_ds = ds.map(lambda waveform, target: (reshape_waveform(waveform), target))

    model = make_model(rnn_ds)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    batch_size = 2
    rnn_ds = rnn_ds.batch(batch_size)

    history = model.fit(
        rnn_ds,
        epochs=100
    )

if __name__ == "__main__":
    main()
