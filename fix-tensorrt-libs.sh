#!/bin/bash
# Set up symlinks to allow tensorflow to find tensorrt library files
# https://github.com/tensorflow/tensorflow/issues/61986

echo "Getting linked tensorrt version"
TENSORRT_VERSION=$(python3 -c "import tensorflow.compiler as tf_cc; print('.'.join(map(str, tf_cc.tf2tensorrt._pywrap_py_utils.get_linked_tensorrt_version())))" 2> /dev/null)
if [ -z "$TENSORRT_VERSION" ]; then
  echo "Linked tensorrt version not detected" >&2
  exit 1
fi
echo $TENSORRT_VERSION

echo "Getting tensorrt lib dir (where tensorflow is looking)"
TENSORRT_FILE="$(python3 -c "import tensorrt; print(tensorrt.__file__)" 2>/dev/null)"
if [ -z "$TENSORRT_FILE" ]; then
  echo "tensorrt dir not found (is tensorrt installed?)" >&2
  exit 1
fi
TENSORRT_DIR="$(dirname "$TENSORRT_FILE")"
echo $TENSORRT_DIR

echo "Getting tensorrt_libs dir (where .so files actually are)"
TENSORRT_LIBS_FILE="$(python3 -c "import tensorrt_libs; print(tensorrt_libs.__file__)" 2>/dev/null)"
if [ -z "$TENSORRT_LIBS_FILE" ]; then
  echo "tensorrt_libs dir not found (is tensorrt installed?)" >&2
  exit 1
fi
TENSORRT_LIBS_DIR="$(dirname "$TENSORRT_LIBS_FILE")"
echo $TENSORRT_LIBS_DIR

echo "Creating links"
ln -srf "${TENSORRT_LIBS_DIR}/libnvinfer.so.8" "${TENSORRT_DIR}/libnvinfer.so.${TENSORRT_VERSION}"
ln -srf "${TENSORRT_LIBS_DIR}/libnvinfer_plugin.so.8" "${TENSORRT_DIR}/libnvinfer_plugin.so.${TENSORRT_VERSION}"

echo "tensorrt lib dir (${TENSORRT_DIR}) contents:"
ls -l "${TENSORRT_DIR}"
