import os
import logging

import tensorflow as tf
import numpy as np

import preprocessing
import settings



module_logger = logging.getLogger(__name__)


sequence_length = 1000
train_fraction = 0.8


def main():

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("eyed3").setLevel(logging.ERROR)

    samples, labels = preprocessing.build_dataset_from_dirs(
        settings.mp3_dir, settings.wav_dir, sequence_length)

    print(samples.shape, labels.shape)

    total_nseq = samples.shape[0]
    train_portion = int(train_fraction*total_nseq)

    train_samples, train_labels = samples[:train_portion], labels[:train_portion]
    test_samples, test_labels = samples[train_portion:], labels[train_portion:]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(sequence_length, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(8)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )


    model.fit(train_samples, train_labels, epochs=100)



if __name__ == "__main__":
    main()
