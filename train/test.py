import keras
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array, load_img

from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import yaml
import sys
import os
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasetgenerator.DatasetGenerator import DatasetGenerator

#--------------------< function >--------------------
# Check to see if you have a folder, and if you don't, create a folder.
def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#----------------------------------------------------

#--------------------< main sequence >--------------------
def main():
    print("######################################")
    print("#    Keras Framework.  TEST program. #")
    print("#      Final Update Date : 2020/4/22 #")
    print("######################################")

    # open congig yaml file.
    print("open config file...")
    with open("config.yaml") as file:
        print("complete!")
        yml = yaml.safe_load(file)
    
    # Dataset Generator loadding.
    dataset = DatasetGenerator()
    print("test dataset loading...")
    print("--------- dataset ---------")
    if yml["testresourcedata"]["readdata"] == "text" or yml["testresourcedata"]["readdata"] == "TEXT":
        xdata, ydata = dataset.text_dataset(yml["testresourcedata"]["resourcepath"])
        datacount = dataset.text_datacounter(yml["testresourcedata"]["resourcepath"])
    elif yml["testresourcedata"]["readdata"] == "onefolder" or yml["testresourcedata"]["readdata"] == "Onefolder":
        xdata, ydata = dataset.onefolder_dataet(yml["testresourcedata"]["resourcepath"])
        datacount = dataset.onefolder_datacounter(yml["testresourcedata"]["resourcepath"])
    elif yml["testresourcedata"]["readdata"] == "folder" or yml["testresourcedata"]["readdata"] == "Folder":
        xdata, ydata = dataset.folder_dataset(yml["testresourcedata"]["resourcepath"])
        datacount = dataset.folder_datacounter(yml["testresourcedata"]["resourcepath"])
    else:
        print("It appears that you have selected a data loader that is not specified. Stops the program.")
        exit(1)
    print("test data : ", datacount) 
    print("---------------------------")

    print("---------  model  ---------")
    # open congig yaml file.
    with open(yml["TESTModel"]["model_path"]) as file:
        model_architecture = yaml.safe_load(file)
    model = model_from_yaml(model_architecture)
    print("Load model weight...")
    model.load_weights(yml["TESTModel"]["weight_path"])
    model.summary()
    print("---------------------------")

    predict_list, y_list = [], []
    bar = tqdm(total = datacount)
    bar.set_description('Progression of predictions ')
    for count, (x, y) in tqdm(enumerate(zip(xdata, ydata))):
        # image file open.
        try:
            # I'll load the image, and if it doesn't work, I'll terminate the program.
            image = img_to_array(load_img(x, color_mode="rgb", target_size=(yml["testresourcedata"]["img_row"], yml["testresourcedata"]["img_col"])))
            # Normalize image.
            image /= 255.
        except Exception as e:
            print("Failed to load data.")
            print("ERROR : ", e)
            exit(1)

        # model predict => pred_label
        predict = model.predict(np.asarray([image], dtype=np.float32), batch_size=1)
        pred_label = np.argmax(predict)
        predict_list.append(pred_label)
        # y => answer list (In fact, I don't use it. It is for real-time processing.)
        y_list.append(int(y))

        # Raw data of predicted values
        #print(predict)
        
        bar.update(1)

    result = classification_report(y_list, predict_list, target_names = yml["testresourcedata"]["classes"],output_dict=True)
    print(" ")
    print("--------------< result >--------------")
    print(result)
    print("--------------------------------------")
    class_pd = pd.DataFrame(result)
    class_pd.to_csv("tanipai.csv")

if __name__ == "__main__":
    main()