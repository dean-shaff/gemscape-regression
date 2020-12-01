import os
import logging

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as tf_preprocessing
import numpy as np

import preprocessing
import settings


module_logger = logging.getLogger(__name__)


sequence_length = 1000
train_fraction = 0.8


def make_model(ds):

    for (feature, target) in ds.take(1):
        print(f"feature.shape={feature.shape}")
        print(f"target.shape={target.shape}")
        input_shape = feature.shape[0]
        output_shape = target.shape[0]

    print(f"Input Shape={input_shape}")
    print(f"Output Shape={output_shape}")

    norm_layer = tf_preprocessing.Normalization()
    norm_layer.adapt(ds.map(lambda x, _: x))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        norm_layer,
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(output_shape)
    ])

    model.summary()

    return model

def main():

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("eyed3").setLevel(logging.ERROR)

    ds = preprocessing.build_dataset_from_dirs(settings.data_dir, desired_time=0.5)
    # ds = ds.map(lambda a, b: (tf.expand_dims(a, -1), b))
    print(ds)
    model = make_model(ds)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    batch_size = 2
    ds = ds.batch(batch_size)

    model.fit(ds, epochs=100)


if __name__ == "__main__":
    main()
