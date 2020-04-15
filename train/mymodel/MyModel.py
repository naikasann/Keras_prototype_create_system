import keras
from keras import optimizers
from keras import losses
from keras.models import Model
from keras.models import load_model
from keras.layers import BatchNormalization, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential
from keras.models import Model

class MyModel:
    def __init__(self, yaml):
        self.yml = yaml
        self.input_shape = (yaml["Resourcedata"]["img_row"], yaml["Resourcedata"]["img_col"], 3)
        self.num_classes = len(yaml["Resourcedata"]["classes"])

    def load_model(self):
        pass
        
    def create_model(self):
        print("Create model....")

        # networkarchitecture setting.
        networkarchitecture = self.yml["Modelsetting"]["network_architecture"]
        print("network architecture setting.")
        print("load model : ", networkarchitecture)
        if networkarchitecture == "nin" or networkarchitecture == "NiN":
            model = self.nin()
        else:
            print("It's an unconfigured optimizer. Use an appropriate network.")
            model = self.nin()
        print("network architecture setting ... ok.")
        
        # model freeze?
        if not self.yml["Modelsetting"]["trainable"]:
            model.trainable = False

        # optimizers setting.
        optimizers = self.yml["Modelsetting"]["optimizers"]
        base_lr = self.yml["Trainsetting"]["learnrate"]
        print("optimizers setting.")
        print("optimizers : ", optimizers)
        if optimizers == "adam" or optimizers == "Adam":
            opt = keras.optimizers.Adam(lr=base_lr)
        elif optimizers == "sgd" or optimizers == "SGD":
            opt = keras.optimizers.SGD(lr=base_lr)
        elif optimizers == "adagrad" or optimizers == "Adagrad":
            opt = keras.optimizers.Adagrad(lr=base_lr)
        else:
            print("This is an unconfigured optimizer that uses Adam instead.")
            opt = keras.optimizers.Adam(lr=base_lr)
        print("optimizer setting ... ok.")

        # model loss setting.
        model_loss = self.yml["Modelsetting"]["model_loss"]
        print("model loss setting.")
        print("model loss : ", model_loss)
        if model_loss == "categorical_crossentropy":
            loss = losses.categorical_crossentropy
        elif model_loss == "mean_squared_error":
            loss = losses.mean_squared_error
        elif model_loss == "binary_crossentropy":
            loss = losses.binary_crossentropy
        elif model_loss == "kullback_leibler_divergence":
            loss = losses.kullback_leibler_divergence
        else:
            print("An unconfigured loss function is used. Instead, we use categorical cross-entropy.")
            loss = loss.categorical_crossentropy
        print("model loss setting... ok.")
        print("The model is now set up. Start compiling...")

        # compile. summury.
        model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
        print("compile ok. summmury")
        model.summary()

        return model

    def nin(self):
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
        model.add(Dense(self.num_classes, activation='softmax'))
        return model