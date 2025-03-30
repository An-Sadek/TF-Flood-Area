import os
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from dataset.FloodArea import FloodAreaDataset
from model.UNet import UNET

from sklearn.metrics import roc_curve

from keras import losses, metrics, optimizers

# Hyperparameter
EPOCH = 1
BATCH_SIZE = 8
IMG_SZ = (576, 576)

# Set up model
unet_layer = UNET([64, 128, 256, 512], 64)
inputs = Input(shape=(
    IMG_SZ[0],
    IMG_SZ[1],
    1
), batch_size=BATCH_SIZE)

outputs = unet_layer(inputs)
output_w = outputs.shape[1]
output_h = outputs.shape[2]

model = Model(inputs=inputs, outputs=outputs)

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

### Train batches
train_dataset = FloodAreaDataset(
    img_paths=X_train, 
    mask_paths=y_train,
    batch_size=BATCH_SIZE, 
    img_size=IMG_SZ,
    mask_size=(output_w, output_h)
)
print("Len of train dataset:", len(train_dataset))
print()

train_batch = tf.data.Dataset.from_generator(
    lambda: train_dataset,
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SZ[0], IMG_SZ[1], 1), dtype=tf.float32),  # For images
        tf.TensorSpec(shape=(BATCH_SIZE, output_w, output_h, 2), dtype=tf.float32)   # For masks
    )
)
train_batch = train_batch.prefetch(tf.data.AUTOTUNE)


### Val batch
val_dataset = FloodAreaDataset(
    img_paths=X_val, 
    mask_paths=y_val,
    batch_size=BATCH_SIZE, 
    img_size=IMG_SZ,
    mask_size=(output_w, output_w)
)
print("Len of val dataset:", len(val_dataset))
print()

val_batch = tf.data.Dataset.from_generator(
    lambda: val_dataset,
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SZ[0], IMG_SZ[1], 1), dtype=tf.float32),  # For images
        tf.TensorSpec(shape=(BATCH_SIZE, output_w, output_h, 1), dtype=tf.float32)   # For masks
    )
)
val_batch = val_batch.prefetch(tf.data.AUTOTUNE)

# Training prepare
loss_fn = losses.BinaryCrossentropy(from_logits=False)
metrics_list = [
    metrics.Accuracy(),
    metrics.Precision(),
    metrics.Recall(),
    metrics.AUC(),
]

optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.9)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics_list
)
"""
history = model.fit(train_batch, validation_data=val_batch, epochs=EPOCH)
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)

print(train_dataset[0][0].shape)
print(train_dataset[0][1].shape)

print(val_dataset[0][0].shape)
print(val_dataset[0][1].shape)
"""