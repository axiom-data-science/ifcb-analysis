# ifcb-analysis

Python based IFCB feature extraction and classification toolkit,
adapted from [Dr Heidi Sosik](https://github.com/hsosik)'s
[ifcb-analysis](https://github.com/hsosik/ifcb-analysis)
(MATLAB). Original python port by [Dr Jesse Lopez](https://github.com/yosoyjay).

Performs feature extraction on IFCB bins
and optionally classifies organisms using a
CNN model.

## Running locally

### Set up environment

```
conda env create -yf environment.yml
conda activate ifcb-analysis
./fix-tensorrt-libs.sh
```

### Run

See help for all available options.

```
python ./src/python/ifcb_analysis/process.py --help
```

Basic execution (extract features and classify all recursively found bins)

```
python ./src/python/ifcb_analysis/process.py \
  --no-extract-images \
  --start-date 2016-04-01 --end-date 2024-05-01 \
  /path/to/raw/data \
  /path/to/output/dir \
  /path/to/model/model.h5 \
  model-id \
  /path/to/model/class_list.json
```

Run feature extraction only on bins within specified dates

```
python ./src/python/ifcb_analysis/process.py \
  --no-classify-images \
  --start-date 2016-04-01 --end-date 2024-05-01 \
  /path/to/raw/data \
  /path/to/output/dir \
  /path/to/model/model.h5 \
  model-id \
  /path/to/model/class_list.json
```

Run classification only on bins within specified dates, in a
flat output directory (not organized into `YYYY/DYYYYMMDD` directories).
Requires that feature and blob files exist from a previous extraction phase.

```
python ./src/python/ifcb_analysis/process.py \
  --no-date-dirs \
  --no-extract-images \
  --start-date 2016-04-01 --end-date 2024-05-01 \
  /path/to/raw/data \
  /path/to/output/dir \
  /path/to/model/model.h5 \
  model-id \
  /path/to/model/class_list.json
```

Force regeneration of existing feature, blob, and class files:

```
python ./src/python/ifcb_analysis/process.py \
  --force \
  --start-date 2016-04-01 --end-date 2024-05-01 \
  /path/to/raw/data \
  /path/to/output/dir \
  /path/to/model/model.h5 \
  model-id \
  /path/to/model/class_list.json
```

A dask cluster may be used for feature extraction by setting
environment variable `DASK_CLUSTER="tcp://some.dask.server:8786"`.
This requires classification to be run in a separate phase using
`--no-classify-images`.

### Run tests

To run tests, first install testing dependencies.

```
conda activate ifcb-analysis
pip install -r test-requirements.txt
```

You'll also need the phytoClassUCSC model.

```
git clone https://huggingface.co/patcdaniel/phytoClassUCSC src/python/tests/data/phytoClassUCSC
```

With that in place, run:

```sh
pytest
```

NOTE: tests may fail with `TensorFlow: Dst tensor is not initialized`
if your GPU is busy with other tasks.
To specify which (non-busy) GPU core(s) to use,
set `CUDA_VISIBLE_DEVICES` when running `pytest`:

```
CUDA_VISIBLE_DEVICES=2,3 pytest
```

## Docker

To run classification in Docker, nvidia drivers and container-toolkit
must be installed on the Docker *host*.
*All* other dependencies are managed/installed in the container using pip.
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Build

```
docker build -t ifcb-analysis .
```

Run (see non-Docker usage examples for more info)

```
docker run --rm --name ifcb-analysis-extraction \
  -e DASK_CLUSTER="tcp://some.dask.server:8786" \
  -v /path/to/raw/ifcb/bins:/data/ifcb/raw:ro \
  -v /path/to/output/data:/data/ifcb/analysis \
  -v /path/to/model:/data/ifcb/model \
  ifcb-analysis \
  --no-classify-images \
  --dask-priority 200
  --start-date 2016-08-01 \
  --end-date 2024-04-23 \
  /data/ifcb/raw \
  /data/ifcb/analysis \
  /data/ifcb/model/phytoClassUCSC.h5 \
  phytoClassUCSC \
  /data/ifcb/model/class_list.json
```
