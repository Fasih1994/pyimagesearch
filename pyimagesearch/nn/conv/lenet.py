from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        input_shape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
