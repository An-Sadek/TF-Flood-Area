import os
from pathlib import Path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import utils


class FloodAreaDataset(utils.PyDataset):

    def __init__(self, img_paths, mask_paths, batch_size, img_size: list|tuple):
        assert len(img_paths) == len(mask_paths)
    
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size

        self.num_batches = math.ceil(len(self.img_paths) / batch_size)

    def __len__(self):
        return self.num_batches

if __name__ == "__main__":
    metadata_path = r"..\Flood Area\metadata.csv"

    assert os.path.exists(metadata_path)

    csv = pd.read_csv(metadata_path)
    print(csv.head())
    print()

    # Them duong dan vao moi phan tu
    parent_dir = str(Path(metadata_path).parent)
    print("Parent dir:", parent_dir)

    img_paths = parent_dir + "\\Image\\" + csv.iloc[:, 0]
    mask_paths = parent_dir + "\\Mask\\" + csv.iloc[:, 1]

    print(img_paths[:5])
    print(mask_paths[:5])
    print()

    # Kiem tra so luong dau vao
    print("Image len:", len(img_paths))
    print("Label len:", len(mask_paths))
    print()

    # Chia du lieu
    X_train, X_test, y_train, y_test = train_test_split(
        img_paths, 
        mask_paths,
        random_state=42, 
        train_size=0.8
    )

    print("X train len:", len(X_train))
    print("X test len:", len(X_test))
    print("y train len:", len(y_train))
    print("y train len:", len(y_test))
    print()

    #dataset = FloodAreaDataset()

