from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()
        input_shape = (height, width, depth)
        chanDims = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chanDims = 1

        # block #1 CONV => RELU => POOL Layer set
        model.add(Conv2D(96, (11, 11), input_shape=input_shape, strides=(4, 4),
                         padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDims))
        model.add(MaxPooling2D((3, 3), (2, 2)))
        model.add(Dropout(0.25))

        # Block #2 CONV => RELU => POOL Layer set
        model.add(Conv2D(256, (5, 5), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDims))
        model.add(MaxPooling2D((3, 3), (2, 2)))
        model.add(Dropout(0.25))

        # Block # 3 CONV => RELU => CONV => RELU => CONV => RELU => POOL Layer Set
        model.add(Conv2D(384, (3, 3), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDims))

        model.add(Conv2D(384, (3, 3), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDims))

        model.add(Conv2D(256, (3, 3), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDims))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),))
        model.add(Dropout(0.25))

        # Block #4 FLATTEN => Dense => RELU => Dense => RELU
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # block # 5 second set of FC layer
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax Classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))

        return model