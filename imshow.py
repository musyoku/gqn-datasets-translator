import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    base_path = os.path.join(args.dataset_directory, "train", "images")
    filename_array = os.listdir(base_path)

    filename = filename_array[1]
    frames = np.load(os.path.join(base_path, filename))
    segments = np.split(frames, 100)
    scenes = segments[25]
    width = len(scenes) * scenes.shape[3]
    height = scenes.shape[1] * scenes.shape[2]
    image = np.zeros((width, height, 3), dtype=scenes.dtype)
    columns = []
    for frame in scenes:
        frame = frame.reshape((-1, 64, 3))
        columns.append(frame)
    image = np.stack(columns, axis=1).reshape((height, width, 3))
    plt.imshow(image)
    plt.pause(100)


    for k, filename in enumerate(filename_array[1:]):
        frames = np.load(os.path.join(base_path, filename))
        segments = np.split(frames, 100)
        for m, scenes in enumerate(segments):
            width = len(scenes) * scenes.shape[3]
            height = scenes.shape[1] * scenes.shape[2]
            image = np.zeros((width, height, 3), dtype=scenes.dtype)
            columns = []
            for frame in scenes:
                frame = frame.reshape((-1, 64, 3))
                columns.append(frame)
            image = np.stack(columns, axis=1).reshape((height, width, 3))
            plt.imshow(image)
            plt.pause(0.1)
            print(k, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory",
        "-dataset",
        type=str,
        default="shepard_metzler_7_npy")
    args = parser.parse_args()
    main()
