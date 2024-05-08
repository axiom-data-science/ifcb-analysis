import filecmp
import os
from pathlib import Path

import ifcb
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from click.testing import CliRunner
from ifcb_analysis import classify, compute_features
from ifcb_analysis.process import cli
from PIL import Image


class TestFeatures:
    basedir = Path(__file__).parent / 'data'
    output_dir = basedir / 'output'
    reference_dir = basedir / 'reference'
    adc_file = basedir / 'D20141117T234033_IFCB102.adc'
    hdf_file = basedir / 'D20141117T234033_IFCB102.hdf'
    roi_file = basedir / 'D20141117T234033_IFCB102.roi'
    model_path = basedir / 'phytoClassUCSC' / 'phytoClassUCSC.h5'
    classes_path = basedir / 'phytoClassUCSC' / 'class_list.json'
    blobs_filename = 'D20141117T234033_IFCB102_blobs_v2.zip'
    classes_filename = 'D20141117T234033_IFCB102_class.h5'
    features_filename = 'D20141117T234033_IFCB102_fea_v2.csv'
    output_blobs = output_dir / blobs_filename
    output_features = output_dir / features_filename
    output_classes = output_dir / classes_filename
    reference_blobs = reference_dir / blobs_filename
    reference_features = reference_dir / features_filename
    reference_classes = reference_dir / classes_filename


    def _pack_df(self, features, roi):
        cols, values = zip(*features)
        cols = ('roi_number',) + cols
        values = (roi,) + values
        values = [(value,) for value in values]
        return pd.DataFrame(
            {c: v for c, v in zip(cols, values)},
            columns=cols
        )


    def _cli_opts(self, *opts):
        return list(opts) + [
            '--no-date-dirs',
            str(self.basedir),
            str(self.output_dir),
            str(self.model_path),
            'test',
            str(self.classes_path),
        ]


    def run_cli(self, *opts):
        runner = CliRunner()
        result = runner.invoke(cli, self._cli_opts(*opts))
        print(result.output)
        assert result.exit_code == 0


    def _clean_output_files(self):
        for f in [self.output_features, self.output_blobs, self.output_classes]:
            f.unlink(missing_ok=True)


    @pytest.fixture(autouse=True)
    def test_cleanup(self):
        # clean up output files before and after running each test
        self._clean_output_files()
        yield
        self._clean_output_files()
        

    def test_process_image(self):
        # Give ADC file
        bin = ifcb.open_raw(self.adc_file)
        PID = 'D20141117T234033_IFCB102'
        N_ROIS = 1346
        N_FEATURES = 240
        IMG_SHAPE = (72, 80)
        ROI = 2

        # Check pid/lid
        assert str(bin.lid) == PID
        assert str(bin.pid) == PID

        # Check number of samples is correct
        assert len(bin.images.keys()) == N_ROIS

        # Check that image is correct
        assert bin.images[2].shape == IMG_SHAPE
        assert bin.images[2].dtype == np.uint8

        # Check features and blob
        blob_img, features = compute_features(bin.images[ROI])
        assert np.sum(blob_img) == 389
        assert blob_img.dtype == bool
        assert blob_img.shape == IMG_SHAPE

        df = self._pack_df(features, ROI)
        assert len(features) == N_FEATURES
        # Adds ROI number to features, so n + 1 features
        assert df.shape == (1, N_FEATURES+1)

    def test_classify(self):
        bin = ifcb.open_raw(self.adc_file)
        ROI = 2

        model_config = classify.KerasModelConfig(self.model_path, self.classes_path, 'test')
        img = (Image
            .fromarray(bin.images[ROI])
            .convert('RGB')
            .resize(model_config.img_dims, Image.BILINEAR)
        )
        # expecting (1, 299, 299, 3)
        input_array = tf.keras.preprocessing.image.img_to_array(img)
        # predict will not normalize the image, this test model used 255.
        img = input_array[np.newaxis, :] / 255

        predictions_df = classify.predict(model_config, img)
        assert predictions_df.iloc[0].argmax() == 29

    def test_script(self):
        self.run_cli()
        assert filecmp.cmp(self.reference_features, self.output_features)
        assert self.reference_classes.stat().st_size == self.output_classes.stat().st_size
        assert self.reference_blobs.stat().st_size == self.output_blobs.stat().st_size

    def test_script_no_classify(self):
        self.run_cli('--no-classify-images')
        assert filecmp.cmp(self.reference_features, self.output_features)
        assert not self.output_classes.exists()
        assert self.reference_blobs.stat().st_size == self.output_blobs.stat().st_size

    def test_script_no_force(self):
        self.output_features.touch()
        self.output_blobs.touch()
        self.output_classes.touch()
        self.run_cli()
        assert not filecmp.cmp(self.reference_features, self.output_features)
        assert not self.reference_classes.stat().st_size == self.output_classes.stat().st_size
        assert not self.reference_blobs.stat().st_size == self.output_blobs.stat().st_size

    def test_script_force(self):
        self.output_features.touch()
        self.output_classes.touch()
        self.output_blobs.touch()
        self.run_cli('--force')
        assert filecmp.cmp(self.reference_features, self.output_features)
        assert self.reference_classes.stat().st_size == self.output_classes.stat().st_size
        assert self.reference_blobs.stat().st_size == self.output_blobs.stat().st_size
