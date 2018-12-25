import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    base_path = os.path.join(args.dataset_directory, "train", "images")
    filename_array = os.listdir(base_path)
    for filename in filename_array:
        frames = np.load(os.path.join(base_path, filename))
        for scene in frames:
            plt.imshow(scene[0])
            plt.pause(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory",
        "-dataset",
        type=str,
        default="shepard_metzler_7_npy")
    args = parser.parse_args()
    main()
