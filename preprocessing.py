import itertools
import typing
import logging
import shlex
import subprocess
import os

import eyed3
import tensorflow as tf
import pysndfile
import numpy as np

ArrayType = np.array
AUTOTUNE = tf.data.experimental.AUTOTUNE

module_logger = logging.getLogger(__name__)

blue_dot_sessions_metadata = [
    "Mood",
    "Density",
    "Gravity",
    "Energy",
    "Ensemble",
    "Melody",
    "Tension",
    "Rhythm"
]


def convert_to_wav(file_path: str, output_file_path: str = None) -> str:
    """
    returns new file_path
    """
    if output_file_path is None:
        output_file_path = os.path.splitext(input_file_path)[0] + ".wav"

    cmd_str = "ffmpeg -y -i \"{}\" \"{}\"".format(file_path, output_file_path)
    module_logger.debug("convert_to_wav: issuing command {}".format(cmd_str))
    cmd = shlex.split(cmd_str)
    subprocess.run(cmd)


def get_blue_dot_sessions_metadata(file_path: str):
    """
    Get Blue Dot Sessions characteristics from .mp3 file, including BPM information, if available.
    This function will not fail in meta data is not found, rather it just returns None.

    Args:
        file_path (str): File path to .mp3 file

    Returns:
        dict: keys are `gemscapes.metadata.blue_dot_sessions_metadata`
            entries, and values are integer values corresponding to each
            metadata characteristic. Additional key/value pair for BPM.
    """
    id_locator = "ID:"
    audiofile = eyed3.load(file_path)
    module_logger.debug(f"get_blue_dot_sessions_metadata: audiofile={audiofile}")
    if audiofile.tag is None:
        return None

    result = {}
    bpm = audiofile.tag.bpm
    if bpm is not None:
        bpm = int(bpm)

    result["bpm"] = bpm

    for comment in itertools.chain(audiofile.tag.comments, audiofile.tag.user_text_frames):
        if id_locator in comment.text:
            start_idx = comment.text.find(id_locator)
            end_idx = comment.text.find(".", start_idx)
            characteristics = comment.text[start_idx + len(id_locator):end_idx]
            result.update({
                blue_dot_sessions_metadata[idx]: int(characteristics[idx])
                for idx in range(len(characteristics))
            })
            return result
    return None


def load_metadata(file_path: str) -> ArrayType:
    # first, get target data
    metadata = get_blue_dot_sessions_metadata(file_path)
    # now create a numpy array with metadata values; this is the target data
    metadata_arr = np.asarray([metadata[name] for name in blue_dot_sessions_metadata], dtype=np.float32)
    return metadata_arr


def load_samples(file_path: str, desired_channels: int = 1, desired_samples: int = -1) -> ArrayType:
    # load the entire mp3 file into memory using pysndfile
    sndfd = pysndfile.PySndfile(file_path, 'r')
    samples = sndfd.read_frames(sndfd.frames())
    # just use first channel, for now
    samples_channel = np.squeeze(samples[:, :desired_channels]).astype(np.float32)
    if desired_samples == -1:
        return samples_channel
    else:
        if samples_channel.shape[0] < desired_samples:
            padding = desired_samples - samples_channel.shape[0]
            return np.concatenate([samples_channel, np.zeros(padding, dtype=np.float32)])
        else:
            return samples_channel[:desired_samples]

    # ab = tf.io.read_file(file_path)
    # waveform, _ = tf.audio.decode_wav(ab, **kwargs)
    #
    # return tf.squeeze(waveform)


def load_tagged_vector(mp3_file_path: str, wav_file_path: str, **kwargs) -> typing.Tuple[ArrayType, ArrayType]:
    """
    get a tagged vector from a mp3 file

    Returns:
        tuple:
    """
    return load_samples(wav_file_path, **kwargs), load_metadata(mp3_file_path)


# def reshape_samples_labels(samples: ArrayType, labels: ArrayType, sequence_length: int) -> typing.Tuple[ArrayType, ArrayType]:
#     nseq = int(samples.shape[0] / sequence_length)
#     samples = samples[:nseq*sequence_length].reshape((nseq, sequence_length))
#     labels = np.tile(labels, nseq).reshape((nseq, -1))
#     return samples, labels


def get_waveform_and_label(mp3_file_path: str, **kwargs):

    wav_file_path = f"{os.path.splitext(mp3_file_path)[0]}.wav"
    if not os.path.exists(wav_file_path):
        convert_to_wav(mp3_file_path, wav_file_path)

    return load_tagged_vector(mp3_file_path, wav_file_path, **kwargs)



def build_dataset_from_dirs(data_dir: str, desired_time: float = 30.0):
    """
    """


    sample_rate = 44100
    desired_samples = int(desired_time * sample_rate)

    mp3_file_paths = [os.path.join(data_dir, name)
                      for name in os.listdir(data_dir)
                      if name.endswith(".mp3")]
    kwargs = dict(desired_samples=desired_samples, desired_channels=1)

    # samples_labels = list(zip(*[get_waveform_and_label(file_path, **kwargs)
    #                             for file_path in mp3_file_paths[:2]]))
    features = []
    targets = []
    for file_path in mp3_file_paths:
        feature, target = get_waveform_and_label(file_path, **kwargs)
        features.append(feature)
        targets.append(target)

    features = np.array(features)
    targets = np.array(targets)
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    return ds

    # ds = tf.data.Dataset.from_generator(
    #     generator,
    #     (tf.float32, tf.float32)
    # )
    # features = np.concatenate(features)
    # targets = np.concatenate(targets)
    # print(features.shape, targets.shape)
    # print(ds)
    # features_ds = tf.data.Dataset.from_tensor_slices(samples_labels[0])
    # targets_ds = tf.data.Dataset.from_tensor_slices(samples_labels[1])
    # return features_ds, targets_ds
