# Installation

See [installation instructions](https://cloud.google.com/storage/docs/gsutil_install).

```
pip3 install tensorflow more_itertools h5py
```

# Usage

```
gsutil rsync -r gs://gqn-dataset/shepard_metzler_7_parts /path/to/downlowd/directory/shepard_metzler_7_parts_tfrecord
python3 convert.py --dataset-name shepard_metzler_7_parts --working-directory /path/to/tmp --dataset-directory /path/to/downlowd/directory/shepard_metzler_7_parts_tfrecord
python3 merge.py --source-directory /path/to/tmp --output-directory /path/to/output/directory/shepard_metzler_7
```