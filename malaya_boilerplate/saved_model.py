import logging
import tensorflow as tf
import tarfile
import os
from glob import glob
from .frozen_graph import get_device
from .utils import _get_home

logger = logging.getLogger(__name__)

try:
    if not tf.config.get_soft_device_placement():
        logger.warning('soft device placement is False, TF Saved Model will throw an error if you do not the device provided.')
except:
    pass


def load_saved_model(package, saved_model_filename, model, **kwargs):

    directory = os.path.split(saved_model_filename)[0]
    directory_model = os.path.join(directory, model)
    pb_files = glob(os.path.join(directory_model, '**', 'saved_model.pb'))

    if not len(pb_files):
        tar = tarfile.open(saved_model_filename, 'r:')
        tar.extractall(path=directory_model)
        tar.close()
        pb_files = glob(os.path.join(directory_model, '**', 'saved_model.pb'))

    path = os.path.split(pb_files[0])[0]
    device = get_device(**kwargs)

    logger.info(f'running {path} using device {device}')

    with tf.device(device):
        imported = tf.saved_model.load(path)
    return imported
