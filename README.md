# ifcb-analysis

## Running locally

### Set up environment

```
conda create -yf environment.yml
conda activate ifcb-analysis
./fix-tensorrt-libs.sh
```

### Run tests

To run tests you'll need a CNN model to use. `ifcb-analysis` is configured to work with Keras models, which, for testing purposes, you can copy into `src/python/tests/data`.

With that in place, run:

```sh
pytest
```
