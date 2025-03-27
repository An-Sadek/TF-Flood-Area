import os
from pathlib import Path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
from PIL import ImageFilter

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
        assert len(img_size) == 2
        assert img_size[0] == img_size[1]
    
        self.img_paths = list(img_paths)
        self.mask_paths = list(mask_paths)
        self.batch_size = batch_size
        self.img_size = img_size

        self.batches = math.ceil(len(self.img_paths) / batch_size)

    def __len__(self):
        return self.batches
    
    def __getitem__(self, idx):
        # Lay idx theo batch size
        start_idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        
        # Lay duong dan cua batch hien tai
        # De phong khi batch cuoi nho
        curr_img_paths = self.img_paths[start_idx: end_idx]
        curr_mask_paths = self.mask_paths[start_idx: end_idx]
        assert len(curr_img_paths) == len(curr_mask_paths)
        curr_batch_size = len(curr_img_paths)

        # Batch anh
        img_batch = np.zeros(shape=(
            curr_batch_size,
            self.img_size[0],
            self.img_size[1],
            1 # De anh xam
        ), dtype=np.float32)

        for i in range(len(curr_img_paths)):
            img = Image.open(curr_img_paths[i]).convert("L")

            # Tien xu ly anh
            img = img.resize(self.img_size)
            img = img.filter(ImageFilter.BoxBlur(1)) # Blur 3x3

            img = np.array(img, dtype=np.float32)
            img = img / 255
            img = np.expand_dims(img, axis=2)
            img_batch[i] = img

        # Batch mask
        mask_batch = np.zeros(shape=(
            curr_batch_size,
            self.img_size[0],
            self.img_size[1],
            1 # Mask anh nhi phan
        ), dtype=bool)

        for i in range(len(curr_mask_paths)):
            mask = Image.open(curr_mask_paths[i]).convert("L")
            mask = mask.resize(self.img_size)
            mask = np.array(mask, dtype=np.float32)
            mask = np.where(mask >= 127, 1, 0)
            mask = np.expand_dims(mask, axis=2)
            mask_batch[i] = mask

        return (
            tf.constant(img_batch, dtype=tf.float32), 
            tf.constant(mask_batch, dtype=tf.float32)
        )

    

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
    ## Train, test
    X_train, X_test, y_train, y_test = train_test_split(
        img_paths, 
        mask_paths,
        random_state=42, 
        train_size=0.9
    )

    ## Train, val, test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        random_state=42, 
        train_size=0.8
    )

    print("X train len:", len(X_train))
    print("y train len:", len(y_train))
    print("X val len:", len(X_val))
    print("y val len:", len(y_val))
    print("X test len:", len(X_test))
    print("y test len:", len(y_test))
    print()

    dataset = FloodAreaDataset(
        img_paths=X_train, 
        mask_paths=y_train,
        batch_size=16, 
        img_size=(56, 56)
    )
    img_batch, mask_batch = dataset[0]

    # Kiem tra anh
    img_single = img_batch[0, :, :, 0]
    print("Mean of image 1:", np.mean(img_single))
    img = np.array(img_single*255, dtype=np.uint8)
    print("Mean of image 2:", np.mean(img))
    pil_img = Image.fromarray(img)
    pil_img.save("test/img.jpg")
    print()

    # Kiem tra mask
    mask_single = mask_batch[0, :, :, 0]
    print("Mean of mask 1:", np.mean(mask_single))
    mask = np.array(mask_single*255, dtype=np.uint8)
    print("Mean of mask 2:", np.mean(mask))
    pil_mask = Image.fromarray(mask)
    pil_mask.save("test/mask.jpg")
    print("Unique value: ", np.unique(mask_single))
    print()

    # Kiem tra tung batch
    train_dataset = FloodAreaDataset(
        X_train,
        y_train,
        16,
        (56, 56)
    )
    print(len(train_dataset))

    val_dataset = FloodAreaDataset(
        X_val,
        y_val,
        16,
        (56, 56)
    )
    print(len(val_dataset))

    test_dataset = FloodAreaDataset(
        X_test,
        y_test,
        16,
        (56, 56)
    )
    print(len(test_dataset))

    

