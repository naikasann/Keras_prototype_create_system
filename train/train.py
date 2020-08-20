import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

import numpy as np
import datetime as dt
import os
import h5py
import yaml
# Self-made dataset lib
from datasetgenerator.DatasetGenerator import DatasetGenerator
from datasetgenerator.BCLearningGenerator import BCLearningGenerator
from datasetgenerator.MixupGenerator import MixupGenerator
from mymodel.CreateModel import CreateModel
from setcallback.SetCallback import SetCallback
from processingresults.ProcessingResults import ProcessingResults

#-------------------< learning rate >---------------------
def learning_rate_scheduler(epoch, learning_rate):
    if epoch < 100:
        return learning_rate
    else:
        return learning_rate / 10
#---------------------------------------------------------

#--------------------< main sequence >--------------------
def main():
    print("######################################")
    print("# Keras Framework. Training program. #")
    print("#      Final Update Date : 2020/8/18 #")
    print("######################################")

    # open congig yaml file.
    print("open config file...")
    with open("config.yaml") as file:
        print("complete!")
        yml = yaml.safe_load(file)

    # do prepare result. (file settting)
    processingresults = ProcessingResults()
    processingresults.Prepareresult()

    # Dataset Generator loadding.
    dataset = DatasetGenerator()
    print("dataset generator loading...")
    print("--------- dataset ---------")
    generator, datacount = dataset.imagedatagenerator(yml)
    print("Use BClearning? :", yml["Trainsetting"]["isBClearning"])
    if yml["Trainsetting"]["isBClearning"]:
        print("Use BCLearning to learn!")
        print("Convert the generator to a BCLearning generator...")
        bc_gen = BCLearningGenerator()
        generator = bc_gen.bclearning_generator(generator, yml["Trainsetting"]["batchsize"])
    if yml["Trainsetting"]["isMixup"]:
        print("Use Mixup to learn!")
        print("Convert the generator to a Mixup generator...")
        traindata = dataset.get_dataset(yml)
        mixup_gen = MixupGenerator(traindata, yml)
        generator = mixup_gen.mixup_generator(generator, yml["Trainsetting"]["batchsize"])
    print("---------------------------")
    print("------- val dataset -------")
    # Validation setting.
    print("Validation use...?  : ", yml["Validation"]["isUse"])
    val_generator, val_datacount = dataset.imagedatagenerator(yml, isValidation=True)
    print("---------------------------")

    # Model loading.
    createmodel = CreateModel(yml)
    if yml["Modelsetting"]["isLoad"]:
        model = createmodel.load_model()
    else:
        model = createmodel.create_model()

    # set callbacks
    setcallback = SetCallback(yml, processingresults, learning_rate_scheduler)
    callbacks = setcallback.getcall_back()

    # training! if => is used Validation?
    if not yml["Validation"]["isUse"]:
        history = model.fit(
            generator,
            steps_per_epoch = int(np.ceil(datacount / yml["Trainsetting"]["batchsize"])),
            epochs = yml["Trainsetting"]["epoch"],
            shuffle=yml["Trainsetting"]["shuffle"],
            callbacks=callbacks
        )
    else:
        history = model.fit(
            generator,
            steps_per_epoch = int(np.ceil(datacount / yml["Trainsetting"]["batchsize"])),
            epochs = yml["Trainsetting"]["epoch"],
            validation_data = val_generator,
            validation_steps = int(np.ceil(val_datacount / yml["Trainsetting"]["batchsize"])),
            shuffle=yml["Trainsetting"]["shuffle"],
            callbacks=callbacks
        )

    # save model archtechture and weight
    processingresults.PreservationResult(model, yml["Trainingresult"]["model_name"])
    # result graph write.
    processingresults.write_graph(history, yml["Trainingresult"]["graph_write"], yml["Validation"]["isUse"])
#---------------------------------------------------------

if __name__ == "__main__":
    main()