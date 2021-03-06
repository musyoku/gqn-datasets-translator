"""
A lot of the following code is a rewrite of:	
https://github.com/deepmind/gqn-datasets/data_reader.py	
https://github.com/l3robot/gqn_datasets_translator	
"""
import argparse
import collections
import multiprocessing
import os
import h5py
import sys
import time
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from more_itertools import chunked

DatasetInfo = collections.namedtuple(
    "DatasetInfo",
    ["basepath", "train_size", "test_size", "frame_size", "sequence_size"])

map_dataset_info = dict(
    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath="rooms_free_camera_with_object_rotations",
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),
    rooms_ring_camera=DatasetInfo(
        basepath="rooms_ring_camera",
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),
    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath="rooms_free_camera_no_object_rotations",
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),
    shepard_metzler_5_parts=DatasetInfo(
        basepath="shepard_metzler_5_parts",
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),
    shepard_metzler_7_parts=DatasetInfo(
        basepath="shepard_metzler_7_parts",
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15))


def convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def preprocess_frames(dataset_info, example):
    frames = tf.concat(example["frames"], axis=0)
    frames = tf.map_fn(
        convert_frame_data,
        tf.reshape(frames, [-1]),
        dtype=tf.float32,
        back_prop=False)
    dataset_image_dimensions = tuple([dataset_info.frame_size] * 2 + [3])
    frames = tf.reshape(
        frames, (-1, dataset_info.sequence_size) + dataset_image_dimensions)
    if dataset_info.frame_size != 64:
        frames = tf.reshape(frames, (-1, ) + dataset_image_dimensions)
        new_frame_dimensions = (64, ) * 2 + (3, )
        frames = tf.image.resize_bilinear(
            frames, new_frame_dimensions[:2], align_corners=True)
        frames = tf.reshape(
            frames, (-1, dataset_info.sequence_size) + new_frame_dimensions)
    return frames


def preprocess_cameras(dataset_info, example):
    raw_pose_params = example["cameras"]
    raw_pose_params = tf.reshape(raw_pose_params,
                                 [-1, dataset_info.sequence_size, 5])
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    cameras = tf.concat(
        [pos, tf.cos(yaw),
         tf.sin(yaw),
         tf.cos(pitch),
         tf.sin(pitch)], axis=2)
    return cameras


def get_dataset_filenames(dataset_info, mode, root):
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)
    files = sorted(os.listdir(base))
    return [os.path.join(base, file) for file in files]


def convert_raw_to_numpy(dataset_info, raw_data):
    feature_map = {
        "frames":
        tf.FixedLenFeature(shape=dataset_info.sequence_size, dtype=tf.string),
        "cameras":
        tf.FixedLenFeature(
            shape=[dataset_info.sequence_size * 5], dtype=tf.float32)
    }
    example = tf.parse_single_example(raw_data, feature_map)
    frames = preprocess_frames(dataset_info, example).numpy()
    cameras = preprocess_cameras(dataset_info, example).numpy()
    return frames, cameras


def show_frame(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.pause(1e-8)


def extract(output_directory, tfrecord_filename, dataset_info):
    try:
        os.mkdir(output_directory)
    except:
        pass

    frames_array = []
    viewpoints_array = []
    frames_npy = None
    viewpoints_npy = None
    engine = tf.python_io.tf_record_iterator(tfrecord_filename)
    start_time = time.time()
    for raw_data in engine:
        frames, viewpoints = convert_raw_to_numpy(dataset_info, raw_data)

        # [0, 1] -> [0, 255]
        frames = np.uint8(frames * 255)

        frames_array.append(frames)
        viewpoints_array.append(viewpoints)

    frames_npy = np.vstack(frames_array)
    viewpoints_npy = np.vstack(viewpoints_array)

    filename = os.path.basename(tfrecord_filename).replace(".tfrecord", ".h5")
    with h5py.File(os.path.join(output_directory, filename), "w") as f:
        f.create_dataset("images", data=frames_npy)
        f.create_dataset("viewpoints", data=viewpoints_npy)

    print(tfrecord_filename, frames_npy.shape, viewpoints_npy.shape, "done in",
          time.time() - start_time, "sec")


def process(arguments):
    (tfrecord_filename, working_directory, mode, dataset_info) = arguments
    extract(
        os.path.join(working_directory, mode), tfrecord_filename, dataset_info)


def run(dataset_info, mode):
    base = os.path.join(args.dataset_directory, mode)
    files = sorted(os.listdir(base))
    tmp_filename_array = [os.path.join(base, file) for file in files]

    filename_array = []
    for tfrecord_filename in tmp_filename_array:
        filename = os.path.basename(tfrecord_filename).replace(
            ".tfrecord", ".h5")
        if os.path.isfile(
                os.path.join(args.working_directory, mode, filename)):
            print(filename, "skipped")
        else:
            filename_array.append(tfrecord_filename)

    arguments = []
    for tfrecord_filename in filename_array:
        arguments.append(
            [tfrecord_filename, args.working_directory, mode, dataset_info])
        if len(arguments) == args.num_threads:
            p = Pool(args.num_threads)
            p.map(process, arguments)
            p.close()
            arguments = []

    if len(arguments) > 0:
        p = Pool(len(arguments))
        p.map(process, arguments)
        p.close()


def main():
    tf.enable_eager_execution()
    try:
        os.mkdir(args.working_directory)
    except:
        pass

    dataset_name = args.dataset_name
    dataset_info = map_dataset_info[dataset_name]

    run(dataset_info, "train")
    run(dataset_info, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", "-thread", type=int, default=10)
    parser.add_argument("--working-directory", "-out", type=str, default="tmp")
    parser.add_argument(
        "--dataset-name", type=str, default="shepard_metzler_7_parts")
    parser.add_argument(
        "--dataset-directory",
        "-dataset",
        type=str,
        default="shepard_metzler_7_parts_tfrecord")
    args = parser.parse_args()
    main()
