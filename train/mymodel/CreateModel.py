import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml, model_from_json

import yaml
import os

from mymodel.Mymodel import Mymodel

class CreateModel:
    # setting instance
    def __init__(self, yml):
        self.yml = yml

    # Create a new model.
    def create_model(self):
        print("Create model....")
        input_shape = (self.yml["Resourcedata"]["img_row"], self.yml["Resourcedata"]["img_col"], 3)
        model = Mymodel.mymodel(input_shape, len(self.yml["Resourcedata"]["classes"]))
        # model freeze?
        if not self.yml["Modelsetting"]["trainable"]:
            model.trainable = False

        # model compile. summury.
        model.compile(loss=self.set_modelloss(self.yml["Modelsetting"]["model_loss"]),
                      optimizer=self.set_optimizers(self.yml["Modelsetting"]["optimizers"],
                                                    self.yml["Trainsetting"]["learnrate"]),
                      metrics=["accuracy"])
        print("compile ok. summmury")
        # Display the results of the compiled model.
        model.summary()

        return model

    # load model
    def load_model(self):
        print("Load model...")
        # open model file.
        model_path = self.yml["Modelsetting"]["model_path"]
        if "json" in model_path:
            model = model_from_json(open(model_path).read())
        elif "yaml" in model_path:
            model = model_from_yaml(open(model_path).read())
        else:
            model = load_model(model_path)

        # Read the weight of the model.
        if self.yml["Modelsetting"]["retrain_model"]:
            print("Load model weight.")
            model.load_weights(self.yml["Modelsetting"]["weight_path"])

        # model freeze?
        if not self.yml["Modelsetting"]["trainable"]:
            model.trainable = False

        # model compile. summury.
        model.compile(loss=self.set_modelloss(self.yml["Modelsetting"]["model_loss"]),
                      optimizer=self.set_optimizers(self.yml["Modelsetting"]["optimizers"],
                                                    self.yml["Trainsetting"]["learnrate"]),
                      metrics=["accuracy"])
        print("compile ok. summmury")
        # Display the results of the compiled model.
        model.summary()

        return model

    # Set the optimizer and return it.
    def set_optimizers(self, select_optimizer, select_lr):
        print("optimizers setting.")
        print("optimizers : ", select_optimizer)
        # Configure the optimizer from config.
        if select_optimizer == "adam" or select_optimizer == "Adam":
            opt = optimizers.Adam(lr=select_lr)
        elif select_optimizer == "sgd" or select_optimizer == "SGD":
            opt = optimizers.SGD(lr=select_lr)
        elif select_optimizer == "adagrad" or select_optimizer == "Adagrad":
            opt = optimizers.Adagrad(lr=select_lr)
        elif select_optimizer == "adadelta" or select_optimizer == "Adadelta":
            opt = optimizers.Adadelta(lr=select_lr)
        else:
            print("This is an unconfigured optimizer that uses Adam instead.")
            opt = optimizers.Adam(lr=select_lr)
        print("optimizer setting ... ok.")

        return opt

    # Set the loss function and return it.
    def set_modelloss(self, select_loss):
        print("model loss setting.")
        print("model loss : ", select_loss)
        # Configure the loss function from config.
        if select_loss == "categorical_crossentropy":
            loss = losses.categorical_crossentropy
        elif select_loss == "mean_squared_error":
            loss = losses.mean_squared_error
        elif select_loss == "binary_crossentropy":
            loss = losses.binary_crossentropy
        elif select_loss == "kullback_leibler_divergence":
            loss = losses.kullback_leibler_divergence
        else:
            print("An unconfigured loss function is used. Instead, we use categorical cross-entropy.")
            loss = losses.categorical_crossentropy
        print("model loss setting... ok.")

        return loss