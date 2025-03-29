import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras 
from keras import layers, regularizers, initializers


class ConvBlock(layers.Layer):
    def __init__(self, filter):
        super(ConvBlock, self).__init__()

        self.dropout = layers.Dropout(0.5)

        self.conv1 = layers.Conv2D(
            filters=filter, 
            kernel_size=(3, 3), 
            strides=1, 
            padding="valid", 
            activation="relu",
            kernel_initializer = initializers.HeUniform(),
            kernel_regularizer = regularizers.L2(0.01),
            use_bias=False
        )

        self.conv2 = layers.Conv2D(
            filters=filter, 
            kernel_size=(3, 3), 
            strides=1, 
            padding="valid", 
            activation="relu",
            kernel_initializer = initializers.HeUniform(),
            kernel_regularizer = regularizers.L2(0.01),
            use_bias=False
        )

        self.batch_norm = layers.BatchNormalization()
        

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        return x

def copy_crop(down: tf.Tensor, x: tf.Tensor):
    h = x.shape[1]
    w = x.shape[2]
    down = down[:, :w, :h, :]

    return tf.concat([down, x], 3)

class UNET(layers.Layer):

    def __init__(self, filters, feature_fraction=1):
        super(UNET, self).__init__()

        assert len(filters) == 4
        assert feature_fraction in [2**i for i in range(0, 7)]
        filters = [value // feature_fraction for value in filters]
        # [64, 128, 256, 512]

        # Encoder
        self.down1 = ConvBlock(filters[0])
        self.down2 = ConvBlock(filters[1])
        self.down3 = ConvBlock(filters[2])
        self.down4 = ConvBlock(filters[3])

        self.pool = layers.MaxPool2D((2, 2))

        # Bottleneck
        self.bottleneck = ConvBlock(filters[3])

        # Decoder
        self.up_sampling = layers.UpSampling2D((2,2))

        self.up1 = ConvBlock(filters[3])
        self.up2 = ConvBlock(filters[2])
        self.up3 = ConvBlock(filters[1])
        self.up4 = ConvBlock(filters[0])

        # Result
        self.final = layers.Conv2D(2, (3, 3), (1, 1), "same", kernel_regularizer=regularizers.L2())

    def call(self, x):
        # Encoder
        down1 = self.down1(x)
        x = self.pool(down1)

        down2 = self.down2(x)
        x = self.pool(down2)

        down3 = self.down3(x)
        x = self.pool(down3)

        down4 = self.down4(x)
        x = self.pool(down4)
        

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up_sampling(x)
        x = copy_crop(down4, x)
        x = self.up1(x)

        x = self.up_sampling(x)
        x = copy_crop(down3, x)
        x = self.up2(x)

        x = self.up_sampling(x)
        x = copy_crop(down2, x)
        x = self.up3(x)

        x = self.up_sampling(x)
        x = copy_crop(down1, x)
        x = self.up4(x)

        return self.final(x)
    
if __name__ == "__main__":
    # ConvBlock
    conv_data = tf.random.uniform(shape=(16, 56, 56, 1))
    conv_input = layers.Input(shape=(56, 56, 1), batch_size=16)
    conv_layer = ConvBlock(64)
    conv_output = conv_layer(conv_input)
    conv_model = keras.Model(inputs=conv_input, outputs=conv_output)
    conv_result = conv_layer(conv_data)
    print("Conv output shape:", conv_result.shape)

    # Copy crop
    data1 = tf.random.uniform((1, 64, 64, 512))
    data2 = tf.random.uniform((1, 56, 56, 512))
    result = copy_crop(data1, data2)
    print("Copy and crop shape: ", result.shape)

    # UNET
    unet_data = tf.random.uniform(shape=(3, 572, 572, 1))
    unet_input = layers.Input((572, 572, 1), 3)
    unet_layer = UNET([64, 128, 256, 512])
    unet_result = unet_layer(unet_data)
    print("UNET output shape:", unet_result.shape)
