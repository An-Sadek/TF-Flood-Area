import os
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from dataset.FloodArea import FloodAreaDataset
from model.UNet import UNET

# Hyperparameter
BATCH_SIZE = 8
IMG_SZ = (576, 576)

# Chuẩn bị dữ liệu
## Đọc csv
metadata_path = r"..\Flood Area\metadata.csv"
assert os.path.exists(metadata_path)
csv = pd.read_csv(metadata_path)

## Đổi đường dẫn phù hợp
parent_dir = str(Path(metadata_path).parent)
img_paths = parent_dir + "\\Image\\" + csv.iloc[:, 0]
mask_paths = parent_dir + "\\Mask\\" + csv.iloc[:, 1]

## Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    img_paths, 
    mask_paths,
    random_state=42, 
    train_size=0.9
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    random_state=42, 
    train_size=0.8
)

train_dataset = FloodAreaDataset(
    img_paths=X_train, 
    mask_paths=y_train,
    batch_size=BATCH_SIZE, 
    img_size=IMG_SZ
)

val_dataset = FloodAreaDataset(
    img_paths=X_val, 
    mask_paths=y_val,
    batch_size=BATCH_SIZE, 
    img_size=IMG_SZ
)

train_batch = tf.data.Dataset.from_generator(
    lambda: train_dataset,
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SZ[0], IMG_SZ[1], 1), dtype=tf.float32),  # For images
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SZ[0], IMG_SZ[1], 1), dtype=tf.float32)   # For masks
    )
)
train_batch = train_batch.prefetch(tf.data.AUTOTUNE)

val_batch = tf.data.Dataset.from_generator(
    lambda: val_dataset,
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SZ[0], IMG_SZ[1], 1), dtype=tf.float32),  # For images
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SZ[0], IMG_SZ[1], 1), dtype=tf.float32)   # For masks
    )
)
val_batch = val_batch.prefetch(tf.data.AUTOTUNE)

# Setting model
