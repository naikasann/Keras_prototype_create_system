import keras
from keras import optimizers
from keras import losses
from keras.models import Model
from keras.models import load_model
from keras.layers import BatchNormalization, Flatten, Reshape
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import UpSampling2D
from keras.models import Sequential
from keras.models import model_from_yaml

import yaml

class MyModel:
    def __init__(self, optimizer, modelloss, base_lr):
        self.optimizers = optimizer
        self.modelloss = modelloss
        self.base_lr = base_lr

    def load_model(self , model_path, weight_path, trainable):
        print("Load model...")
        # oprn congig yaml file.
        with open(model_path) as file:
            yml = yaml.safe_load(file)
        model = model_from_yaml(yml)
        print("Load model weight.")
        model.load_weights(weight_path)

        # model freeze?
        if not trainable:
            model.trainable = False

        # compile. summury.
        model.compile(loss=self.set_modelloss(), 
                      optimizer=self.set_optimizers(),
                      metrics=["accuracy"])
        print("compile ok. summmury")
        model.summary()

        return model

        
    def create_model(self, networkarchitecture, input_shape, classes, trainable):
        print("Create model....")

        # networkarchitecture setting.
        print("network architecture setting.")
        print("load model : ", networkarchitecture)
        if networkarchitecture == "nin" or networkarchitecture == "NiN":
            model = self.nin(input_shape, len(classes))
        elif networkarchitecture == "segnet" or networkarchitecture == "Segnet":
            model = self.Camvid_SegNet(input_shape, len(classes))
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
        model.summary()

        return model

    def set_optimizers(self):
        # optimizers setting.
        optimizers = self.optimizers
        base_lr = self.base_lr
        print("optimizers setting.")
        print("optimizers : ", optimizers)
        if optimizers == "adam" or optimizers == "Adam":
            opt = keras.optimizers.Adam(lr=base_lr)
        elif optimizers == "sgd" or optimizers == "SGD":
            opt = keras.optimizers.SGD(lr=base_lr)
        elif optimizers == "adagrad" or optimizers == "Adagrad":
            opt = keras.optimizers.Adagrad(lr=base_lr)
        elif optimizers == "adadelta" or optimizers == "Adadelta":
            opt = keras.optimizers.Adadelta(lr=base_lr)
        else:
            print("This is an unconfigured optimizer that uses Adam instead.")
            opt = keras.optimizers.Adam(lr=base_lr)
        print("optimizer setting ... ok.")

        return opt
    
    def set_modelloss(self):
        # model loss setting.
        model_loss = self.modelloss
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

        return loss

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
        model.add(Dense(num_classes, activation='softmax'))
        
        return model
    
    def Camvid_SegNet(self, input_shape, num_classes):
        # https://github.com/alexgkendall/SegNet-Tutorial
        # example model : bayesian_segnet_camvid.prototxt
        # https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt

        model = Sequential()
        # Encode layer1
        model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        # Encode layer2
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        # Encode layer3
        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        # Encode layer4
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # Decoder
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # Upsampling decode layer1
        model.add(UpSampling2D(size = (2, 2)))
        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # Upsampling decode layer2
        model.add(UpSampling2D(size = (2, 2)))
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # Upsampling decode layer3
        model.add(UpSampling2D(size = (2, 2)))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(num_classes, (1, 1), padding = "same"))
        model.add(Reshape((input_shape[0], input_shape[1], num_classes)))
        model.add(Activation("softmax"))

        return model