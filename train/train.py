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

#--------------------< function >--------------------
def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

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
    print("#      Final Update Date : 2020/4/13 #")
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
        generator = dataset.text_dataset_generator(yml)
    else:
        pass


    f.close()
#---------------------------------------------------------

if __name__ == "__main__":
    main()