import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    base_path = os.path.join(args.dataset_directory, "train")
    image_filename_array = os.listdir(os.path.join(base_path, "images"))
    viewpoint_filename_array = os.listdir(os.path.join(base_path, "viewpoints"))

    for (image_filename, viewpoint_filename) in zip(image_filename_array, viewpoint_filename_array):
        frames = np.load(os.path.join(base_path, "images", image_filename))
        viewpoints = np.load(os.path.join(base_path, "viewpoints", viewpoint_filename))
        print(frames.shape)
        print(viewpoints.shape)
        print(viewpoints[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory",
        "-dataset",
        type=str,
        default="shepard_metzler_7_npy")
    args = parser.parse_args()
    main()
