# imports
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

class DeeperGoogleNet:
    @staticmethod
    def conv_module(x, k, kx, ky, strides, chanDim, padding='same', reg=0.0005, name=None):
        (convName, bnName, actName) = (None, None, None)

        if not name==None:
            convName = name + '_conv'
            bnName = name + '_bn'
            actName = name + '_act'

        x = Conv2D(k, (kx, ky), strides=strides, padding=padding, name=convName,
                   kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation('relu', name=actName)(x)

        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce,
                         num5x5, num1x1Proj, chanDim, stage, reg=0.0005):
        first = DeeperGoogleNet.conv_module(x, num1x1, 1, 1, (1, 1), chanDim=chanDim, reg=reg,
                                            name=stage + '_first')

        second = DeeperGoogleNet.conv_module(x, num3x3Reduce, 1, 1, (1, 1), chanDim=chanDim,
                                             reg=reg, name=stage +'_second1')
        second = DeeperGoogleNet.conv_module(second, num3x3, 3, 3, (1, 1), chanDim=chanDim,
                                             reg=reg, name=stage + '_second2')

        third = DeeperGoogleNet.conv_module(x, num5x5Reduce, 1, 1, (1, 1), chanDim=chanDim,
                                            reg=reg, name=stage + '_third1')
        third = DeeperGoogleNet.conv_module(third, num5x5, 5, 5, (1, 1), chanDim=chanDim,
                                            reg=reg, name=stage + '_third2')

        fourth = MaxPooling2D((3, 3), (1, 1), padding='same', name=stage + '_pool')(x)
        fourth = DeeperGoogleNet.conv_module(fourth, num1x1Proj, 1, 1, (1, 1), chanDim=chanDim,
                                             reg=reg, name=stage + '_fourtg')
        X = concatenate([first, second, third, fourth], axis=chanDim, name=stage+'_mixed')

        return X
    @staticmethod
    def build(height, width, depth, classes, reg=0.0005):

        input_shape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=input_shape)
        x = DeeperGoogleNet.conv_module(inputs, 64, 5, 5, (1, 1), chanDim=chanDim,
                                        reg=reg, name='block1')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

        x = DeeperGoogleNet.conv_module(x, 64, 1, 1, (1, 1), chanDim=chanDim,
                                        reg=reg,name='block2')
        x = DeeperGoogleNet.conv_module(x, 192, 3, 3, (1, 1), chanDim=chanDim,
                                        reg=reg, name='block3')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool2')(x)

        x = DeeperGoogleNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim,
                                             stage='3a', reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim=chanDim,
                                             stage='3b', reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool3')(x)

        x = DeeperGoogleNet.inception_module(x, 192, 92, 208, 16, 48, 64, chanDim, stage='4a', reg=reg)
        x = DeeperGoogleNet.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, stage='4b', reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, stage='4c', reg=reg)
        x = DeeperGoogleNet.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, stage='4d', reg=reg)
        x = DeeperGoogleNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, stage='4e', reg=reg)
        x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool4')(x)

        x = AveragePooling2D((4, 4), name='pool5')(x)
        x = Dropout(0.4, name='do')(x)

        x = Flatten(name='Flatten')(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name='Label')(x)
        x = Activation('softmax', name='softmax')(x)

        model = Model(inputs, x, name='googlenet')

        return model


