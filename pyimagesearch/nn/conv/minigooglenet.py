from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras import backend as K

class MiniGoogleNet:
    @staticmethod
    def conv_module(x,k, kx, ky, strides, chanDim, padding='same'):
        x = Conv2D(k, (kx, ky), strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)

        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):

        conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim=chanDim)
        conv_3x3 = MiniGoogleNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim=chanDim)

        x = concatenate([conv_1x1, conv_3x3], axis= chanDim)

        return x

    @staticmethod
    def downsample_module(x, k, chanDim):
        # define the CONV module and POOL, then concatenate
        # across the channel dimensions
        conv_3x3 = MiniGoogleNet.conv_module(x, k, 3, 3, (2, 2), chanDim=chanDim, padding='valid')
        pool = MaxPooling2D((3, 3),strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=input_shape)
        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
        x = MiniGoogleNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogleNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogleNet.downsample_module(x, 80, chanDim)

        # four Inception modules followed by a downsample module
        x = MiniGoogleNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogleNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogleNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogleNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogleNet.downsample_module(x, 96, chanDim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)


        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs, x, name='googlenet')
        return model
