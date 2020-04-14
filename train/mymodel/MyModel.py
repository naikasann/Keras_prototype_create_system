import keras
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential
from keras.models import Model

class MyModel:
    def __init__(self, yaml):
        self.yml = yaml
        
    def create_model()

    def nin_model(self):
        model = Sequential()

        # MLPconv layer1
        model.add(Conv2D(96, (11, 11), strides = (4, 4), padding = "same", input_shape = self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(96, (1, 1), strides = (4, 4), padding = "same"))
        model.add(Activation('relu'))

        # pooling layer1
        model.add(MaxPooling2D((3, 3), strides = (2, 2)))

        # MLPconv layer2
        model.add(Conv2D(256, (5, 5), strides = (2, 2), padding = "same"))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (1, 1), strides = (2, 2), padding = "same"))
        model.add(Activation('relu'))

        # pooling layer2
        model.add(MaxPooling2D((3, 3), strides = (2, 2), padding="same"))

        # MLPconv layer3
        model.add(Conv2D(384, (3, 3), strides = (1, 1), padding = "same"))
        model.add(BatchNormalization())
        model.add(Conv2D(384, (1, 1), strides = (1, 1), padding = "same"))
        model.add(Activation('relu'))

        # pooling layer3
        model.add(MaxPooling2D((3, 3), strides = (2, 2), padding="same"))
        model.add(Activation('relu'))

        # drop out layer
        model.add(Dropout(0.5))

        # MLPconv layer4
        model.add(Conv2D(3, (3, 3), strides = (1, 1), padding = "same"))
        model.add(BatchNormalization())
        model.add(Conv2D(3, (1, 1), strides = (1, 1), padding = "same"))
        model.add(Activation('relu'))

        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))

        return model