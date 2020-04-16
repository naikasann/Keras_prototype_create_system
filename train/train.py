import keras
from keras import optimizers
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

import tensorflow as tf
from tensorflow.python.client import device_lib

import numpy as np
import datetime as dt
import os
import h5py
import yaml
import shutil
# Self-made dataset lib
from datasetgenerator.DatasetGenerator import DatasetGenerator
from mymodel.MyModel import MyModel

#--------------------< function >--------------------
# Check to see if you have a folder, and if you don't, create a folder.
def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# Checking the status of the GPU
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    dev_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if dev_list is None:
        return False
    return True
#----------------------------------------------------

#--------------------< main sequence >--------------------
# read config file.
def main():
    print("######################################")
    print("# Keras Framework. Training program. #")
    print("#      Final Update Date : 2020/4/16 #")
    print("######################################")

    # oprn congig yaml file.
    print("open config file...")
    with open("config.yaml") as file:
        print("complete!")
        yml = yaml.safe_load(file)

    # GPU checking.
    print("GPU use option checking...")
    if yml["running"]["GPU"]:
        print("use gpu. setting...")
        # if GPU setting.
        if not get_available_gpus():
            print("GPU is not available.You should review your configuration files and GPU preferences.")
            exit(1)
        # GPU setting
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=gpu_config))
    else:
        # It warns you not to use the GPU.
        print("gpu dont use. It does not use a GPU. It takes a lot of time. Are you ready? (y/n)")
        text = input()
        if text == "y" or text == "Y":
            print("ok. It will be executed as it is.")
        else:
            print("ok. You should review your configuration files and GPU preferences.")
            exit(0)

    # Store the execution time in a variable.
    execute_time = dt.datetime.now().strftime("%m_%d_%H_%M")
    makedir("result/")
    makedir("result/" + execute_time)
    makedir("result/" + execute_time + "/model")
    # Copy the YAML file that contains the execution environment.
    shutil.copy("config.yaml", "result/" + execute_time)

    # Dataset Generator loadding.
    dataset = DatasetGenerator()
    if yml["Resourcedata"]["readdata"] == "text":
        generator = dataset.text_dataset_generator( yml["Resourcedata"]["resourcepath"],
                                                    (yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"]),
                                                    yml["Resourcedata"]["classes"],
                                                    yml["Trainsetting"]["batchsize"]
                                                    )
        datacount = dataset.text_datacounter(yml["Resourcedata"]["resourcepath"])
    else:
        pass
    
    # Validation setting.
    print("Validation use...?  : ", yml["Validation"]["Usedata"])
    if yml["Validation"]["Usedata"]:
        # use validation data.
        val_dataset = DatasetGenerator()
        print("Create a generator to use the validation.")
        if yml["Validation"]["readdata"]:
            val_generator = val_dataset.text_dataset_generator(yml["Validation"]["resourcepath"],
                                                          (yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"]),
                                                          yml["Resourcedata"]["classes"],
                                                          yml["Trainsetting"]["batchsize"]
                                                          )
            val_datacount = val_dataset.text_datacounter(yml["Validation"]["resourcepath"])
    else:
        # dont use validation data. Continue learning.
        print("It does not use any validation.")

    # Model loading.
    mymodel = MyModel(yml)
    if yml["Modelsetting"]["retrain_model"]:
        model = mymodel.load_model(yml["Modelsetting"]["model_path"])
    else:
        model = mymodel.create_model()
    
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
    
    # training!
    if not yml["Validation"]["Usedata"]:
        # no validation
        history = model.fit_generator(
            generator = generator,
            steps_per_epoch = int(np.ceil(datacount / yml["Trainsetting"]["batchsize"])),
            epochs = yml["Trainsetting"]["epoch"],
            callbacks=[modelCheckpoint]
        )
    else:
        #validation
        history = model.fit_generator(
            generator = generator,
            steps_per_epoch = int(np.ceil(datacount / yml["Trainsetting"]["batchsize"])),
            epochs = yml["Trainsetting"]["epoch"],
            validation_data = val_generator,
            validation_steps = int(np.ceil(val_datacount / yml["Trainsetting"]["batchsize"])),
            callbacks=[modelCheckpoint]
        )

    # save weights and model.
    model.save_weights("./result/" + execute_time + "/model/" + yml["Trainingresult"]["model_name"] + "_end_epoch" + ".h5")    
#---------------------------------------------------------

if __name__ == "__main__":
    main()