#!python
"""Load bin, extract features, create blob image, classify samples, and write to disk."""
from __future__ import annotations

import click
import gc
import h5py as h5
import ifcb
import logging
import numpy as np
import os
import pandas as pd
import warnings

from contextlib import nullcontext
from datetime import datetime, timedelta
from ifcb.data.imageio import format_image
from pathlib import Path
from src.python.ifcb_analysis import classify, compute_features
from zipfile import ZipFile
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)


def process_bin(
    file: Path,
    outdir: Path,
    model_config: classify.KerasModelConfig,
    extract_images: bool = True,
    classify_images: bool = True,
    force: bool = False
):
    if not outdir.exists():
        # Due to race conditions when a job is concurrently processing bins
        # from the same, date mkdir will occasionally fail if the directory
        # was created by a different process in the time between when it checks
        # for the directory's existence and when it actually goes to create the
        # directory. If this happens, just ignore it.
        try:
            outdir.mkdir(parents=True)
        except FileExistsError:
            pass

    bin = ifcb.open_raw(file)

    blobs_fname = outdir / f'{bin.lid}_blobs_v2.zip'
    features_fname = outdir / f'{bin.lid}_fea_v2.csv'
    class_fname = outdir / f'{bin.lid}_class.h5'

    # determine which files need to be (re-)generated
    mode_change_messages = []
    if not force:
        all_files_exist = True
        if extract_images:
            extract_files_exist = blobs_fname.exists() and features_fname.exists()
            if extract_files_exist:
                #no need to extract if files exist and we're not forcing
                extract_images = False
                mode_change_messages.append(f'All extraction files exist for {bin.pid}, skipping extraction')
            all_files_exist = all_files_exist and extract_files_exist
        if classify_images:
            all_files_exist = all_files_exist and class_fname.exists()
            if class_fname.exists():
                mode_change_messages.append(f'All classification files exist for {bin.pid}, skipping classification')
        if all_files_exist:
            logging.debug(f'Output files for {bin.pid} already exist, skipping. To override, set --force flag.')
            return

    logging.info(f'Processing {bin.pid}, saving results to {outdir}')

    if mode_change_messages:
        for msg in mode_change_messages:
            logging.info(msg)

    if not extract_images:
        features_df = pd.read_csv(features_fname)
    else:
        features_df = None

    roi_number = None
    num_rois = len(bin.images.keys())

    if classify_images:
        image_stack = np.zeros((num_rois, model_config.img_dims[0], model_config.img_dims[1], 3))

    # loop through all images in the bin
    with (ZipFile(blobs_fname, 'w') if extract_images else nullcontext()) as blob_zip:
        num_images = len(bin.images)
        for ix, roi_number in enumerate(bin.images):
            if ix % 100 == 0:
                logging.info(f'Processing ROI {roi_number} ({ix + 1}/{num_images})')
            try:
                # Select image
                image = bin.images[roi_number]

                if extract_images:
                    # Compute features
                    blob_img, features = compute_features(image)

                    # Write blob image to zip as bytes.
                    # Include ROI number in filename. e.g. D20141117T234033_IFCB102_2.png
                    image_bytes = blob2bytes(blob_img)
                    blob_zip.writestr(f'{bin.pid.with_target(roi_number)}.png', image_bytes)

                    # Add features row to dataframe
                    # - Copied pyifcb
                    row_df = features2df(features, roi_number)
                    if features_df is None:
                        features_df = row_df
                    else:
                        features_df = pd.concat([features_df, row_df])

                if classify_images:
                    # Resize image, normalized, and add to stack
                    pil_img = (Image
                        .fromarray(image)
                        .convert('RGB')
                        .resize(model_config.img_dims, Image.BILINEAR)
                    )
                    img = np.array(pil_img) / model_config.norm
                    image_stack[ix, :] = img
            except Exception as e:
                logging.error(f'Failed to extract {file} for ROI {roi_number}')
                if os.path.exists(blobs_fname):
                    os.remove(blobs_fname)
                raise e

    # Save features dataframe
    # - Empty features indicates no samples, so remaining steps are skipped
    if features_df is not None:
        #only save features to disk if we're extracting, otherwise we loaded
        #these features from this existing file above
        if extract_images:
            logging.info(f'Saving features to {features_fname}')
            features_df.to_csv(features_fname, index=False, float_format='%.6f')

        if classify_images:
            logging.info(f'Classifying images and saving to {class_fname}')
            predictions_df = classify.predict(model_config, image_stack)

            # Since classify.predict (which calls Model.predict) is run in a for loop, memory consumption
            # will build up and result in an OOM error, so we excplicitly clear it out after each model run.
            gc.collect()

            # Save predictions to h5
            predictions2h5(model_config, class_fname, predictions_df, bin.lid, features_df)

        else:
            logging.info('Classification turned off, skipping')

    else:
        logging.info(f'No features found in {file}. Skipping classification.')

def blob2bytes(blob_img: np.ndarray) -> bytes:
    """Format blob as image to be written in zip file."""
    image_buf = format_image(blob_img)
    return image_buf.getvalue()


def features2df(features: list, roi_number: int) -> pd.DataFrame:
    """Convert features to dataframe (copy pasta from featureio.py)."""
    cols, values = zip(*features)
    cols = ('roi_number',) + cols
    values = (roi_number,) + values
    values = [(value,) for value in values]
    return pd.DataFrame({c: v for c, v in zip(cols, values)},
                        columns=cols)


def predictions2h5(model_config: classify.KerasModelConfig, outfile: Path, predictions_df: pd.DataFrame, bin_lid: str, features: pd.DataFrame):
    """Save classified predictions to h5 file."""

    with h5.File(outfile, 'w') as f:
        meta = f.create_dataset('metadata', data=h5.Empty('f'))
        meta.attrs['version'] = 'v3'
        meta.attrs['model_id'] = model_config.model_id
        # DYYYYMMDDTHHMMSS_IFCBXXX
        meta.attrs['timestamp'] = bin_lid.split('_')[0][1:]
        meta.attrs['bin_id'] = bin_lid
#       I think these are just indices
#        f.create_dataset('output_classes', data=results['output_classes'], compression='gzip', dtype='float16')
        f.create_dataset('output_scores', data=predictions_df.values, compression='gzip', dtype='float16')
#        f.create_dataset('class_labels', data=np.string_(results['class_labels']), compression='gzip', dtype=h5.string_dtype())
        f.create_dataset('class_labels', data=predictions_df.columns, compression='gzip', dtype=h5.string_dtype())
        f.create_dataset('roi_numbers', data=features['roi_number'], compression='gzip', dtype='uint16')


def available_bins(ifcb_data_dir: Path, pids: List[str], start_date: datetime, end_date: datetime) -> List[Path]:
    """Given path to data return list of bins available, optionally using start and end date to discover"""
    if pids:
        bins = []
        for pid in pids:
            adc = f'{pid}.adc'
            date_dir_adc = ifcb_data_dir / date_dir(pid) / adc
            root_dir_adc = ifcb_data_dir / adc
            if root_dir_adc.exists():
                bins.append(root_dir_adc)
            elif date_dir_adc.exists():
                bins.append(date_dir_adc)
            else:
                logging.warn(f'No matches found for specified pid {pid}')
    elif start_date and end_date:
        ndays = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(ndays)]

        bins = []
        for date in dates:
            day_dir = ifcb_data_dir / f'{date.year}/D{date.year}{date.month:02d}{date.day:02d}'
            if not day_dir.exists():
                continue

            daily_bins = list(day_dir.glob('*.adc'))
            daily_bins.sort()
            logging.debug(f'Adding {daily_bins} to available bins')
            bins += daily_bins
    else:
        # just scan recursively for *.adc if start and end date weren't provided
        bins = list(ifcb_data_dir.rglob('*.adc'))
        bins.sort()

    return bins


def date_from_pid(pid: str) -> datetime:
    return datetime.strptime(pid[0:10], 'D%Y%m%dT')


def date_dir(pid: str) -> str:
    # DYYYYmmddTHHMMSS_<ifcb-name>.adc -> YYYY/DYYYYmmdd
    # D20230717T012101_IFCB104.adc -> 2023/D20230717
    return f'{pid[1:5]}/{pid[:9]}'


def output_path(features_output_dir: Path, bin: Path) -> Path:
    """Given output dir and bin name, create path for output file following canonical naming."""
    # DYYYYmmddTHHMMSS_<ifcb-name>.adc
    return features_output_dir / date_dir(bin.name)


def process(
    ifcb_data_dir: Path,
    features_output_dir: Path,
    model_path: Path,
    model_id: str,
    class_path: Path,
    dask_priority: int,
    pids: List[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    date_dirs: bool = True,
    extract_images: bool = True,
    classify_images: bool = True,
    force: bool = False,
    use_dask: bool = True
):
    """Process bins."""
    logging.info(f'Processing IFCB data in {ifcb_data_dir}')
    logging.info(f'Writing output to {features_output_dir}')
    if pids:
        if len(pids) <= 10:
            logging.info(f'pids: {pids}')
        else:
            logging.info(f'{len(pids)} pids, first {pids[0]}, last {pids[-1]}')
    if start_date or end_date:
        logging.info(f'Start date f{start_date}, end date f{end_date}')

    model_config = classify.KerasModelConfig(model_path=model_path, class_path=class_path, model_id=model_id)
    bins = available_bins(ifcb_data_dir=ifcb_data_dir, pids=pids, start_date=start_date, end_date=end_date)

    if use_dask and classify_images:
        logging.info('Classification is not supported by the Dask cluster. Running serially instead.')
        use_dask = False

    if use_dask:
        import dask
        from dask.distributed import Client, fire_and_forget

        with Client(os.environ['DASK_CLUSTER']) as client:

            if date_dirs:
                outdirs = [output_path(features_output_dir, bin) for bin in bins]
            else:
                outdirs = features_output_dir * len(bins)

            args = [
                bins,
                outdirs,
                [model_config] * len(bins),
                [extract_images] * len(bins),
                [classify_images] * len(bins),
                [force] * len(bins)
            ]
            futures = client.map(
                process_bin,
                *args,
                priority=dask_priority
            )
            fire_and_forget(futures)

    else:
        num_bins = len(bins)
        for ix, bin in enumerate(bins):
            if num_bins <= 500 or ix % 100 == 0:
                percent_complete = (ix/float(num_bins)) * 100
                logging.info(f'Progress: {percent_complete:.1f}% {ix}/{num_bins} ({bin})')

            if date_dirs:
                outdir = output_path(features_output_dir, bin)
            else:
                outdir = features_output_dir

            try:
                process_bin(bin, outdir, model_config, extract_images, classify_images, force)
            except Exception as e:
                logging.error(f'Error processing {bin}: {e}')


@click.command()
@click.option('--extract-images/--no-extract-images', default=True)
@click.option('--classify-images/--no-classify-images', default=True)
@click.option('--force/--no-force', default=False)
@click.option('--log-level', default='INFO')
@click.option('--log-file', type=click.Path(writable=True, dir_okay=False))
@click.option('--use-dask/--no-use-dask', default=os.getenv('DASK_CLUSTER') is not None)
@click.option('--dask-priority', type=click.INT, default=0)
@click.option('--pids', '--pid', type=click.STRING)
@click.option('--pids-file', '--pid-file', type=click.Path(dir_okay=False))
@click.option('--start-date', type=click.DateTime(formats=['%Y-%m-%d']))
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']))
@click.option('--date-dirs/--no-date-dirs', default=True)
@click.argument('ifcb_data_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('features_output_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('model_id', type=click.STRING)
@click.argument('class_path', type=click.Path(exists=True, dir_okay=False))
def cli(
    extract_images: bool,
    classify_images: bool,
    force: bool,
    log_level: str,
    log_file: Path,
    use_dask: bool,
    dask_priority: int,
    pids: str,
    pids_file: str,
    start_date: datetime,
    end_date: datetime,
    date_dirs: bool,
    ifcb_data_dir: Path,
    features_output_dir: Path,
    model_path: Path,
    model_id: str,
    class_path: Path,
):
    """Process bins between start and end dates."""

    # set up logging (stdout, plug log file if path provided)
    log_handlers = [logging.StreamHandler()]
    if (log_file):
        log_handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(handlers=log_handlers, level=log_level,
                        format='%(message)s')

    if pids:
        pids = [pid.strip() for pid in pids.split(',')]
        pids.sort()
    elif pids_file:
        with open(pids_file) as f:
            pids = [line.strip() for line in f if line.strip()]
        pids.sort()

    process(
        ifcb_data_dir=ifcb_data_dir,
        features_output_dir=features_output_dir,
        model_path=model_path,
        model_id=model_id,
        class_path=class_path,
        dask_priority=dask_priority,
        pids=pids,
        start_date=start_date,
        end_date=end_date,
        date_dirs=date_dirs,
        extract_images=extract_images,
        classify_images=classify_images,
        force=force,
        use_dask=use_dask,
    )

if __name__ == '__main__':
    cli()
