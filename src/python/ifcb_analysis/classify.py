import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

import h5py as h5
import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass
class KerasModelConfig:
    model_path: Union[Path, str]
    class_path: Union[Path, str]
    model_id: str = 'unknown'
    _model:  tf.keras.Model = None
    class_names: dict = field(init=False)
    img_dims: Tuple[int, int] = (299, 299)
    norm: int = 255

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.class_path = Path(self.class_path)
        self.class_names = self._read_class_names(self.class_path)

    @property
    def model(self):
        if self._model is None:
            self._model = tf.keras.models.load_model(self.model_path)
        return self._model
        

    def _read_class_names(self, path: Path) -> dict:
        if path.suffix == '.txt':
            # assume comma seperated values in txt file
            with open(path, 'r') as f:
                classes = f.readlines()[0].split(',')
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                classes = [x for x in json.load(f) if x.lower() != 'unclassified']
        return {ix: name for ix, name in enumerate(classes)}


def predict(model_config: KerasModelConfig, image_stack: np.ndarray, batch_size=64) -> pd.DataFrame:
    # Classify images and save as csv
    predictions = model_config.model.predict(image_stack, batch_size)
    predictions_df = pd.DataFrame(
        predictions,
        columns=model_config.class_names.values()
    )

    return predictions_df
