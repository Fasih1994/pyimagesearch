# imports
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import add
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

class ResNet:
    @staticmethod
    def residual_module(data, k, strides, chanDim, reg=0.0001,
                        red=False, bnEps=2e-5, bnMom=0.9):
        shortcut = data

        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(k*0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(k*0.25), (3, 3), strides=strides, padding='same',
                       use_bias=False, kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(k, (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act3)

        # if dimensions needed to reduce
        if red:
            shortcut = Conv2D(k, (1, 1), strides=strides, use_bias=False,
                              kernel_regularizer=l2(reg))(act1)
        x = add([conv3, shortcut])

        return x

    @staticmethod
    def build(height, width, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset='cifar'):
        if K.image_data_format()=='channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        else:
            inputShape = (height, width, depth)
            chanDim = -1
        # set the inputs and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        if dataset=='cifar':
            # apply a single conv layer
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same',
                       kernel_regularizer=l2(reg))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the strides and apply residual module to reduce the spatial size of
            # input volume
            strides = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i+1], strides=strides, chanDim=chanDim,
                                       red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over number of layers in stages
            for j in range(0, stages[i]-1):
                # apply a resnet module
                x = ResNet.residual_module(x, filters[i+1], (1, 1), chanDim=chanDim,
                                           bnEps=bnEps, bnMom=bnMom)

        # apply BN ==> ACT ==> POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)


        # softmax
        x = Flatten()(x)
        x = Dense(classes,kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        # create model
        model = Model(inputs, x, name='resnet')

        return model
