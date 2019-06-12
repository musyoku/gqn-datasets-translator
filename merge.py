import os
import argparse
import h5py
import numpy as np


def merge(source_directory, target_directory, mode, num_scenes_per_file):
    try:
        os.mkdir(os.path.join(args.output_directory, mode))
    except:
        pass

    # Aggregate numbers of scenes.
    num_scenes = 0
    base_path = os.path.join(source_directory, mode)
    filename_array = os.listdir(base_path)
    for index, filename in enumerate(filename_array):
        filepath = os.path.join(base_path, filename)
        with h5py.File(filepath, "r") as f:
            images = f["images"][()]
            num_scenes += images.shape[0]

    print("#scene", num_scenes)
    assert num_scenes % num_scenes_per_file == 0

    output_file_number = 1

    images = None
    viewpoints = None
    for index, filename in enumerate(filename_array):
        with h5py.File(os.path.join(base_path, filename), "r") as f:
            _images = f["images"][()]
            _viewpoints = f["viewpoints"][()]
            if images is None:
                images = _images
            else:
                images = np.concatenate((images, _images), axis=0)
            if viewpoints is None:
                viewpoints = _viewpoints
            else:
                viewpoints = np.concatenate((viewpoints, _viewpoints), axis=0)

            while len(images) > num_scenes_per_file:
                images_to_save, images_rest = images[:
                                                     num_scenes_per_file], images[
                                                         num_scenes_per_file:]
                viewpoints_to_save, viewpoints_rest = viewpoints[:num_scenes_per_file], viewpoints[
                    num_scenes_per_file:]

                with h5py.File(
                        os.path.join(target_directory, mode,
                                     "{:04d}.h5".format(output_file_number)),
                        "w") as f:
                    f.create_dataset("images", data=images_to_save)
                    f.create_dataset("viewpoints", data=viewpoints_to_save)

                images = images_rest
                viewpoints = viewpoints_rest
                output_file_number += 1

            print(index, "/", len(filename_array), "completed")


def main():
    try:
        os.mkdir(args.output_directory)
    except:
        pass

    merge(args.source_directory, args.output_directory, "train",
          args.num_scenes_per_file)
    merge(args.source_directory, args.output_directory, "test",
          args.num_scenes_per_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes-per-file", "-ns", type=int, default=1000)
    parser.add_argument("--output-directory", "-out", type=str, required=True)
    parser.add_argument(
        "--source-directory", "-source", type=str, default="tmp")
    args = parser.parse_args()
    main()
