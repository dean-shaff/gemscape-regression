import itertools
import typing
import logging
import shlex
import subprocess
import os

import tqdm
import eyed3
import tensorflow as tf
import pysndfile
import numpy as np

ArrayType = np.array

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


def build_dataset_from_dirs(
    data_dir: str,
    desired_time: float = 30.0,
    max_files: int = None,
    test_split: float = 0.8
):
    """
    Args:
        max_files: max number of files to process
    """
    if test_split > 1.0 or test_split < 0.0:
        raise ValueError("test_split needs to be between 0.0 and 1.0")

    sample_rate = 44100
    desired_samples = int(desired_time * sample_rate)

    mp3_file_paths = [os.path.join(data_dir, name)
                      for name in os.listdir(data_dir)
                      if name.endswith(".mp3")]

    if max_files is None:
        max_files = len(mp3_file_paths)

    kwargs = dict(desired_samples=desired_samples, desired_channels=1)

    features = []
    targets = []
    max_files = min(max_files, len(mp3_file_paths))

    iterator = range(max_files)
    if module_logger.getEffectiveLevel() > logging.DEBUG:
        iterator = tqdm.tqdm(range(max_files))

    for idx in iterator:
        file_path = mp3_file_paths[idx]
        try:
            feature, target = get_waveform_and_label(file_path, **kwargs)
        except (IndexError, ValueError, TypeError) as err:
            module_logger.error(f"build_dataset_from_dirs: couldn't process {file_path}")
        features.append(feature)
        targets.append(target)

    features = np.array(features)
    targets = np.array(targets)
    len_features = len(features)
    test_split_int = int(test_split * len_features)
    module_logger.debug(f"build_dataset_from_dirs: test_split_int={test_split_int}")
    train_ds = tf.data.Dataset.from_tensor_slices((features[:test_split_int], targets[:test_split_int]))
    test_ds = tf.data.Dataset.from_tensor_slices((features[test_split_int:], targets[test_split_int:]))
    return train_ds, test_ds
