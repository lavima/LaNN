import os
import logging
import struct
import numpy as np

from kagglehub import dataset_download

logger = logging.getLogger(__name__)

def log_load(name):
    logger.info(f'Loading {name}')

def load_mnist():
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols, 1)

            images = images.astype(np.float32) / 255.0

            return images

    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    log_load('MNIST')

    path = dataset_download("hojjatk/mnist-dataset")

    return (
        (
            load_images(os.path.join(path, 'train-images.idx3-ubyte')), 
            load_labels(os.path.join(path, 'train-labels.idx1-ubyte')) 
        ),
        (
            load_images(os.path.join(path, 't10k-images.idx3-ubyte')), 
            load_labels(os.path.join(path, 't10k-labels.idx1-ubyte')) 
        )
    )


