"""
A lot of the following code is a rewrite of:	
https://github.com/deepmind/gqn-datasets/data_reader.py	
https://github.com/l3robot/gqn_datasets_translator	
"""

import argparse
import time
import sys
import os
import collections
import torch
import gzip
import numpy as np
import cupy as cp

import tensorflow as tf

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
    if (64 != dataset_info.frame_size):
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
    if mode == "train":
        num_files = dataset_info.train_size
    else:
        num_files = dataset_info.test_size

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
    frames = preprocess_frames(dataset_info, example)
    cameras = preprocess_cameras(dataset_info, example)
    with tf.train.SingularMonitoredSession() as sess:
        frames = sess.run(frames)
        cameras = sess.run(cameras)

    return frames, cameras


def show_frame(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.pause(1e-8)


def compute_mean_and_variance(np_frames,
                              total_observations,
                              dataset_mean=None,
                              dataset_var=None):
    subset_size = np_frames.shape[0]
    new_total_size = total_observations + subset_size
    co1 = total_observations / new_total_size
    co2 = subset_size / new_total_size

    frames = cp.asarray(np_frames)

    subset_mean = cp.mean(frames, axis=(0, 1))
    subset_var = cp.var(frames, axis=(0, 1))

    new_dataset_mean = subset_mean if dataset_mean is None else co1 * dataset_mean + co2 * subset_mean
    new_dataset_var = subset_var if dataset_var is None else co1 * (
        dataset_var + dataset_mean**2) + co2 * (
            subset_var + subset_mean**2) - new_dataset_mean**2

    # avoid negative value
    new_dataset_var[new_dataset_var < 0] = 0

    return new_dataset_var, new_dataset_mean, cp.sqrt(new_dataset_var)


def extract(output_directory, filenames, dataset_info):
    try:
        os.mkdir(output_directory)
    except:
        pass
    try:
        os.mkdir(os.path.join(output_directory, "images"))
    except:
        pass
    try:
        os.mkdir(os.path.join(output_directory, "viewpoints"))
    except:
        pass

    num_observed = 0
    current_file_number = 1
    frames_array = []
    viewpoints_array = []
    dataset_mean, dataset_var = None, None

    for filename in filenames:
        engine = tf.python_io.tf_record_iterator(filename)
        for raw_data in engine:
            frames, viewpoints = convert_raw_to_numpy(dataset_info, raw_data)

            if args.with_visualization:
                show_frame(frames[0, 0])

            # [0, 1] -> [-1, 1]
            frames = (frames - 0.5) * 2.0

            frames_array.append(frames)
            viewpoints_array.append(viewpoints)
            num_observed += 1

            sys.stdout.write("\r")
            sys.stdout.write("extracting {} of {} ...".format(
                len(frames_array), args.num_observations_per_file))

            if (len(frames_array) == args.num_observations_per_file):
                frames_array = np.vstack(frames_array)
                viewpoints_array = np.vstack(viewpoints_array)

                dataset_mean, dataset_var, dataset_std = compute_mean_and_variance(
                    frames_array, num_observed, dataset_mean, dataset_var)

                cp.save(
                    os.path.join(output_directory, "mean.npy"), dataset_mean)
                cp.save(os.path.join(output_directory, "std.npy"), dataset_std)

                filename = "{:03d}-of-{}.npy".format(
                    current_file_number, args.num_observations_per_file)
                np.save(
                    os.path.join(output_directory, "images", filename),
                    frames_array)

                filename = "{:03d}-of-{}.npy".format(
                    current_file_number, args.num_observations_per_file)
                np.save(
                    os.path.join(output_directory, "viewpoints", filename),
                    viewpoints_array)

                frames_array = []
                viewpoints_array = []

                sys.stdout.write("\r")
                print("\033[2K{} of {} completed.".format(
                    num_observed, args.total_observations))

                if num_observed >= args.total_observations:
                    return


def main():
    assert args.total_observations > args.num_observations_per_file

    dataset_name = args.dataset_name
    dataset_info = map_dataset_info[dataset_name]

    try:
        os.mkdir(args.output_directory)
    except:
        pass

    ## train
    filenames = get_dataset_filenames(dataset_info, "train",
                                      args.source_dataset_directory)
    extract(
        os.path.join(args.output_directory, "train"), filenames, dataset_info)

    ## test
    filenames = get_dataset_filenames(dataset_info, "test",
                                      args.source_dataset_directory)
    extract(
        os.path.join(args.output_directory, "test"), filenames, dataset_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-visualization",
        "-visualize",
        action="store_true",
        default=False)
    parser.add_argument(
        "--total-observations", "-total", type=int, default=2000000)
    parser.add_argument(
        "--num-observations-per-file", "-per-file", type=int, default=2000)
    parser.add_argument("--output-directory", type=str, default="dataset")
    parser.add_argument(
        "--dataset-name", type=str, default="shepard_metzler_7_parts")
    parser.add_argument(
        "--source-dataset-directory", "-source", type=str, default=".")
    args = parser.parse_args()
    main()
