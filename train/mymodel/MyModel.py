import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.applications import MobileNetV2

class Mymodel:
    def mymodel(input_shape, num_classes):
        # Call the mobilenet.
        print("modliienet...")
        moblienet = MobileNetV2(weights="imagenet",
                                include_top=False,
                                input_shape=input_shape)
        model = Sequential()
        model.add(moblienet)
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))

        print('before :', len(model.trainable_weights))

        # Freeze all non-input and output layers in the model.
        # (transference learning => True)
        moblienet.trainable = True

        # Fine tuning
        set_trainable = False
        for layer in moblienet.layers:
            # Set the learning layer and beyond to be trained.
            if layer.name == "block_14_depthwise":
                set_trainable = True
            # Whether or not to freeze the layers.
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        print('after :', len(model.trainable_weights))

        return model