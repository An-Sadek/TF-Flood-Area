import tensorflow as tf
from tensorflow import keras 
from keras import layers


class UNET(layers.Layer):

    def __init__(self):
        super(UNET).__init__()

    def call(self, x):
        return