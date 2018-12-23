# Installation

See [installation instructions](https://cloud.google.com/storage/docs/gsutil_install).

```
pip3 install tensorflow more_itertools
```

# Usage

```
gsutil -m cp -R gs://gqn-dataset/shepard_metzler_7_parts .
python3 convert.py --dataset-name shepard_metzler_7_parts --working-directory tmp
```