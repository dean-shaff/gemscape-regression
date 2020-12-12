import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as tf_preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import preprocessing
import settings
import create_parser


def make_model(ds):
    """
    Make LSTM model

    Args:
        ds: tensorflow dataset
    """
    for (spec, target) in ds.take(1):
        input_shape = spec.shape
        output_shape = target.shape[0]

    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    # num_labels = len(commands)

    norm_layer = tf_preprocessing.Normalization()
    norm_layer.adapt(ds.map(lambda x, _: x))
    # axis=1 is normalizing on samples
    # axis=2 is normalizing on time
    model = models.Sequential([
        tf.keras.layers.LayerNormalization(axis=1, center=True, scale=True, input_shape=input_shape),
        tf.keras.layers.LSTM(64, input_shape=input_shape),
        # tf.keras.layers.LSTM(64),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128), input_shape=input_shape),
        tf.keras.layers.Dense(32),
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
    # waveform = tf.reshape(waveform, (nslices, samples_per_slice))
    waveform = tf.transpose(tf.reshape(waveform, (samples_per_slice, nslices)))
    return waveform


def main():
    parsed = create_parser.create_parser().parse_args()
    level = logging.INFO
    if parsed.verbose:
        level = logging.DEBUG

    logging.basicConfig(level=level)
    logging.getLogger("eyed3").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    desired_time = 1.0
    samples_per_slice = 32
    max_files = 10
    epochs = 10

    checkpoints_dir = os.path.join(settings.checkpoints_dir, f"t-{desired_time}_sps-{samples_per_slice}")
    checkpoints_path = os.path.join(checkpoints_dir, "cp-{epoch:04d}.ckpt")


    train_ds, test_ds = preprocessing.build_dataset_from_dirs(
        settings.data_dir,
        desired_time=desired_time,
        max_files=max_files
    )

    def mapper(waveform, target):
        return (reshape_waveform(waveform, samples_per_slice=samples_per_slice), target)

    train_ds = train_ds.map(mapper)
    test_ds = test_ds.map(mapper)

    for (spec, target) in test_ds.take(1):
        print(spec.shape)


    loss = tf.keras.losses.MeanSquaredError()

    model = make_model(train_ds)
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0001),
        # metrics=[tf.keras.metrics.RootMeanSquaredError()],
        loss=loss
    )

    batch_size = 2
    train_ds_batched = train_ds.batch(batch_size)

    for (spec, target) in test_ds.take(1):
        spec = spec[np.newaxis, :]
        pred = model.predict(spec)
        print("before training:")
        with np.printoptions(precision=3):
            print(f"    target={target}")
            print(f"    pred={pred[0]}")
            print(f"    loss={loss(target, pred[0]).numpy():.3f}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoints_path,
        verbose=0,
        save_weights_only=True,
        save_freq=max_files)

    history = model.fit(
        train_ds_batched,
        epochs=epochs,
        callbacks=[cp_callback],
        validation_data=test_ds.batch(1)
    )

    for (spec, target) in test_ds.take(1):
        spec = spec[np.newaxis, :]
        pred = model.predict(spec)
        print("after training")
        with np.printoptions(precision=3):
            print(f"    target={target}")
            print(f"    pred={pred[0]}")
            print(f"    loss={loss(target, pred[0]).numpy():.3f}")






if __name__ == "__main__":
    main()
