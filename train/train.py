import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import datetime as dt
import os
import h5py
import yaml
import shutil
# Self-made dataset lib
from datasetgenerator.DatasetGenerator import DatasetGenerator
import datasetgenerator.BCLearningGenerator as BCLearning
from mymodel.MyModel import MyModel
from graphplot.GraphPlot import GraphPlot

#--------------------< function >--------------------
# Check to see if you have a folder, and if you don't, create a folder.
def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#----------------------------------------------------

#--------------------< main sequence >--------------------
def main():
    print("######################################")
    print("# Keras Framework. Training program. #")
    print("#      Final Update Date : 2020/7/31 #")
    print("######################################")

    # open congig yaml file.
    print("open config file...")
    with open("config.yaml") as file:
        print("complete!")
        yml = yaml.safe_load(file)

    # Store the execution time in a variable.
    execute_time = dt.datetime.now().strftime("%m_%d_%H_%M")
    makedir("result/")
    makedir("result/" + execute_time)
    makedir("result/" + execute_time + "/model")
    # Copy the YAML file that contains the execution environment.
    shutil.copy("config.yaml", "result/" + execute_time)

    # Dataset Generator loadding.
    dataset = DatasetGenerator()
    print("dataset generator loading...")
    print("--------- dataset ---------")
    generator, datacount = dataset.imagedatagenerator(yml)
    print("Use BClearning? :", yml["Trainsetting"]["isBClearning"])
    if yml["Trainsetting"]["isBClearning"]:
        print("Use BCLearning to learn!")
        print("Convert the generator to a BCLearning generator...")
        generator = BCLearning.BCLearningGenerator(generator)
    print("---------------------------")
    print("------- val dataset -------")
    # Validation setting.
    print("Validation use...?  : ", yml["Validation"]["isUse"])
    val_generator, val_datacount = dataset.imagedatagenerator(yml, isValidation=True)
    print("---------------------------")


    # Model loading.
    mymodel = MyModel(yml["Modelsetting"]["optimizers"],
                      yml["Modelsetting"]["model_loss"],
                      yml["Trainsetting"]["learnrate"])
    if yml["Modelsetting"]["retrain_model"]:
        model = mymodel.load_model(yml["Modelsetting"]["model_path"],
                                   yml["Modelsetting"]["weight_path"],
                                   yml["Modelsetting"]["trainable"])
    else:
        input_shape = (yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"], 3)
        model = mymodel.create_model(yml["Modelsetting"]["network_architecture"],
                                     input_shape,
                                     yml["Resourcedata"]["classes"],
                                     yml["Modelsetting"]["trainable"])

    # callback function(tensorboard, modelcheckpoint) setting.
    # first modelcheckpoint setting.
    modelCheckpoint = ModelCheckpoint(filepath = "./result/"+ execute_time + "/model/" + yml["Trainingresult"]["model_name"] +"_{epoch:02d}.h5",
                                  monitor=yml["callback"]["monitor"],
                                  verbose=yml["callback"]["verbose"],
                                  save_best_only=yml["callback"]["save_best_only"],
                                  save_weights_only=yml["callback"]["save_weights_only"],
                                  mode=yml["callback"]["mode"],
                                  period=yml["callback"]["period"])

    # next tensorboard setting.
    tensorboard = TensorBoard(log_dir = yml["callback"]["tensorboard"], histogram_freq=yml["callback"]["tb_epoch"])

    # training! if => is used Validation?
    if not yml["Validation"]["isUse"]:
        history = model.fit(
            generator,
            steps_per_epoch = int(np.ceil(datacount / yml["Trainsetting"]["batchsize"])),
            epochs = yml["Trainsetting"]["epoch"],
            shuffle=yml["Trainsetting"]["shuffle"],
            callbacks=[modelCheckpoint, tensorboard]
        )
    else:
        history = model.fit(
            generator,
            steps_per_epoch = int(np.ceil(datacount / yml["Trainsetting"]["batchsize"])),
            epochs = yml["Trainsetting"]["epoch"],
            validation_data = val_generator,
            validation_steps = int(np.ceil(val_datacount / yml["Trainsetting"]["batchsize"])),
            shuffle=yml["Trainsetting"]["shuffle"],
            callbacks=[modelCheckpoint, tensorboard]
        )

    # save weights and model.
    model.save("./result/" + execute_time + "/model/" + yml["Trainingresult"]["model_name"] + "_end_epoch.h5")
    # write network architecture.
    f = open("./result/" + execute_time + "/model/model_architecture.yaml", "w")
    f.write(yaml.dump(model.to_yaml()))
    f.close()
    # result graph write.
    GraphPlot.write_graph(history, yml["Trainingresult"]["graph_write"], execute_time, yml["Validation"]["isUse"])
#---------------------------------------------------------

if __name__ == "__main__":
    main()