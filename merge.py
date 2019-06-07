import os
import argparse
import numpy as np


def merge(source_directory, target_directory, mode, num_scenes_per_file):
    try:
        os.mkdir(os.path.join(args.output_directory, mode))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.output_directory, mode, "images"))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.output_directory, mode, "viewpoints"))
    except:
        pass

    # Aggregate numbers of scenes.
    num_scenes = 0
    base_path = os.path.join(source_directory, mode, "images")
    filename_array = os.listdir(base_path)
    for index, filename in enumerate(filename_array):
        filepath = os.path.join(base_path, filename)
        with open(filepath, "rb") as f:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            num_scenes += shape[0]

    print("#scene", num_scenes)
    assert num_scenes % num_scenes_per_file == 0

    output_file_number = 1

    frames_base_path = os.path.join(source_directory, mode, "images")
    viewpoints_base_path = os.path.join(source_directory, mode, "viewpoints")
    frames = None
    viewpoints = None
    for index, filename in enumerate(filename_array):
        _frames = np.load(os.path.join(frames_base_path, filename))
        _viewpoints = np.load(os.path.join(viewpoints_base_path, filename))
        if frames is None:
            frames = _frames
        else:
            frames = np.concatenate((frames, _frames), axis=0)
        if viewpoints is None:
            viewpoints = _viewpoints
        else:
            viewpoints = np.concatenate((viewpoints, _viewpoints), axis=0)

        while len(frames) > num_scenes_per_file:
            frames_to_save, frames_rest = frames[:num_scenes_per_file], frames[
                num_scenes_per_file:]
            viewpoints_to_save, viewpoints_rest = viewpoints[:num_scenes_per_file], viewpoints[
                num_scenes_per_file:]
            np.save(
                os.path.join(target_directory, mode, "images",
                             "{:05d}.npy".format(output_file_number)),
                frames_to_save)
            np.save(
                os.path.join(target_directory, mode, "viewpoints",
                             "{:05d}.npy".format(output_file_number)),
                viewpoints_to_save)
            frames = frames_rest
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
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="shepard_metzler_7_npy")
    parser.add_argument(
        "--source-directory", "-source", type=str, default="tmp")
    args = parser.parse_args()
    main()
