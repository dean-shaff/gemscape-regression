# download_data.py
import json
import typing
import sys
import os
import functools
from multiprocessing.pool import ThreadPool

import boto3
import tqdm

import settings


bucket = "bluedotsessions"
prefix = "ingestion/"


def get_mp3s(s3) -> typing.List[str]:

    def mp3_keys_from_resp(resp):
        return [v["Key"] for v in resp["Contents"] if v["Key"].endswith(".mp3")]

    resp = s3.list_objects(Bucket=bucket, Prefix=prefix)
    mp3_keys = mp3_keys_from_resp(resp)

    while resp["IsTruncated"]:
        marker = resp["Contents"][-1]["Key"]
        resp = s3.list_objects(Bucket=bucket, Prefix=prefix, Marker=marker)
        mp3_keys.extend(mp3_keys_from_resp(resp))

    return mp3_keys


def download_mp3s(s3, mp3_keys: typing.List[str]):
    """
    Download mp3 files to data directory
    """
    for key in tqdm.tqdm(mp3_keys):
        # pretend key is actually a path
        file_name = os.path.basename(key)
        file_path = os.path.join(settings.data_dir, file_name)
        with open(file_path, "wb") as fd:
            s3.download_fileobj(bucket, key, fd)


def main(mp3_keys_file_path: str = None):
    s3 = boto3.client('s3')

    mp3_keys = None
    if mp3_keys_file_path is None:
        mp3_keys_file_path = "mp3_keys.json"
        mp3_keys = get_mp3s(s3)
        with open(mp3_keys_file_path, "w") as fd:
            json.dump(mp3_keys, fd, indent=2)
    else:
        with open(mp3_keys_file_path, "r") as fd:
            mp3_keys = json.load(fd)

    # pool = ThreadPool(2)
    # pool.map(functools.partial(download_mp3s, s3), mp3_keys)
    download_mp3s(s3, mp3_keys)


if __name__ == "__main__":
    mp3_keys_file_path = None
    if len(sys.argv) > 1:
        mp3_keys_file_path = sys.argv[1]
    main(mp3_keys_file_path)
