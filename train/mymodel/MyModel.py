import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D

import yaml

class MyModel:
    # setting instance
    def __init__(self, optimizer, modelloss, base_lr):
        self.optimizers = optimizer
        self.modelloss = modelloss
        self.base_lr = base_lr

    # Read the network architecture from the YAML file.
    def load_model(self , model_path, weight_path, trainable):
        print("Load model...")
        # open congig yaml file.
        with open(model_path) as file:
            yml = yaml.safe_load(file)
        model = model_from_yaml(yml)
        print("Load model weight.")
        # Read the weight of the model.
        model.load_weights(weight_path)

        # model freeze?
        if not trainable:
            model.trainable = False

        # compile. summury.
        model.compile(loss=self.set_modelloss(), 
                      optimizer=self.set_optimizers(),
                      metrics=["accuracy"])
        print("compile ok. summmury")
        # Display the results of the compiled model.
        model.summary()

        return model

    # Create a new model.
    def create_model(self, networkarchitecture, input_shape, classes, trainable):
        print("Create model....")
        # networkarchitecture setting.
        print("network architecture setting.")
        print("load model : ", networkarchitecture)
        ############################################
        #       Generate a sequential model.       #
        ############################################
        if networkarchitecture == "nin" or networkarchitecture == "NiN":
            model = self.nin(input_shape, len(classes))
        else:
            print("It's an unconfigured model. Use an appropriate network.")
            model = self.nin(input_shape, len(classes))
        print("network architecture setting ... ok.")
        
        # model freeze?
        if not trainable:
            model.trainable = False

        # compile. summury.
        model.compile(loss=self.set_modelloss(), 
                      optimizer=self.set_optimizers(), 
                      metrics=["accuracy"])
        print("compile ok. summmury")
        # Display the results of the compiled model.
        model.summary()

        return model

    # Set the optimizer and return it.
    def set_optimizers(self):
        print("optimizers setting.")
        print("optimizers : ", self.optimizers)
        # Configure the optimizer from config.
        if self.optimizers == "adam" or self.optimizers == "Adam":
            opt = keras.optimizers.Adam(lr=self.base_lr)
        elif self.optimizers == "sgd" or self.optimizers == "SGD":
            opt = keras.optimizers.SGD(lr=self.base_lr)
        elif self.optimizers == "adagrad" or self.optimizers == "Adagrad":
            opt = keras.optimizers.Adagrad(lr=self.base_lr)
        elif self.optimizers == "adadelta" or self.optimizers == "Adadelta":
            opt = keras.optimizers.Adadelta(lr=self.base_lr)
        else:
            print("This is an unconfigured optimizer that uses Adam instead.")
            opt = keras.optimizers.Adam(lr=self.base_lr)
        print("optimizer setting ... ok.")

        return opt
    
    # Set the loss function and return it.
    def set_modelloss(self):
        print("model loss setting.")
        print("model loss : ", self.modelloss)
        # Configure the loss function from config.
        if self.modelloss == "categorical_crossentropy":
            loss = losses.categorical_crossentropy
        elif self.modelloss == "mean_squared_error":
            loss = losses.mean_squared_error
        elif self.modelloss == "binary_crossentropy":
            loss = losses.binary_crossentropy
        elif self.modelloss == "kullback_leibler_divergence":
            loss = losses.kullback_leibler_divergence
        else:
            print("An unconfigured loss function is used. Instead, we use categorical cross-entropy.")
            loss = loss.categorical_crossentropy
        print("model loss setting... ok.")

        return loss

    # Return the structure of Network in Network.
    def nin(self, input_shape, num_classes):
        model = Sequential()
        # MLPconv layer1
        model.add(Conv2D(96, (11, 11), strides = (4, 4), padding = "same", input_shape = input_shape))
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
        model.add(MaxPooling2D((3, 3), strides = (2, 2), padding = "same"))
        # MLPconv layer3
        model.add(Conv2D(384, (3, 3), strides = (1, 1), padding = "same"))
        model.add(BatchNormalization())
        model.add(Conv2D(384, (1, 1), strides = (1, 1), padding = "same"))
        model.add(Activation('relu'))
        # pooling layer3
        model.add(MaxPooling2D((3, 3), strides = (2, 2), padding = "same"))
        model.add(Activation('relu'))
        # drop out layer
        model.add(Dropout(0.5))
        # MLPconv layer4
        model.add(Conv2D(3, (3, 3), strides = (1, 1), padding = "same"))
        model.add(BatchNormalization())
        model.add(Conv2D(3, (1, 1), strides = (1, 1), padding = "same"))
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation("softmax"))
        
        return model