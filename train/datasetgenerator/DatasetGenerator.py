import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import os

class DatasetGenerator:
    # for test program. It's better to alter it later.
    def text_dataset(self, resourcepath):
        train_data = []
        # open text file.
        with open(resourcepath) as f:
            readlines = f.readlines()

        # Input data.
        for line in readlines:
            linebuffer = line.split(" ")
            input = linebuffer[0]
            answer = linebuffer[1].rstrip("\n")
            train_data.append([input, str(answer)])
            # return [imagepath], [label]
        return train_data

    # for test program. It's better to alter it later.
    def onefolder_dataet(self, resourcepath, classes):
        train_data = []

        input = os.listdir(resourcepath)
        for image in input:
            for count, category,  in enumerate(classes):
                # Find out which category. Identify the image name and assign a label.
                if category in image:
                    input = image
                    answer = count
                    train_data.append([input, str(answer)])
        # return [imagepath], [label]
        return train_data

    # for test program. It's better to alter it later.
    def folder_dataset(self, resourcepath):
        train_data = []

        # folder check.
        labeldir = os.listdir(resourcepath)
        # Loop the number of label folders.
        for category, labelname in enumerate(labeldir):
            label_path = os.path.join(resourcepath, labelname)
            images = os.listdir(label_path)
            # Extracts data from a set of image files.
            for imagepath in images:
                input = os.path.join(label_path, imagepath)
                answer = category
                train_data.append([input, str(answer)])
        # return [imagepath], [label]
        return train_data

    def get_dataset(self, yml):
        # create dataset list
        if yml["Resourcedata"]["readdata"] == "text" or yml["Resourcedata"]["readdata"] == "TEXT":
            train_data = self.text_dataset(yml["Resourcedata"]["resourcepath"])
        elif yml["Resourcedata"]["readdata"] == "onefolder" or yml["Resourcedata"]["readdata"] == "Onefolder":
            train_data = self.onefolder_dataet(yml["Resourcedata"]["resourcepath"],
                                                yml["Resourcedata"]["classes"])
        elif yml["Resourcedata"]["readdata"] == "folder" or yml["Resourcedata"]["readdata"] == "Folder":
            train_data = self.folder_dataset(yml["Resourcedata"]["resourcepath"])
        else:
            print("It seems to have input an unsupported generator name. Stops the program.")
            exit(1)

        return train_data


    def imagedatagenerator(self, yml, isValidation=False):
        # use image data generator(use padding?)
        if isValidation:
            datagen = ImageDataGenerator(rescale = 1./255)
        else:
            if yml["Trainsetting"]["useAugment"]:
                print("Data augmentation using a data generator")
                params = {
                    "rescale"                       : 1./255,
                    "featurewise_center"            : yml["Trainsetting"]["featurewise_center"],
                    "samplewise_center"             : yml["Trainsetting"]["samplewise_center"],
                    "featurewise_std_normalization" : yml["Trainsetting"]["featurewise_std_normalization"],
                    "samplewise_std_normalization"  : yml["Trainsetting"]["samplewise_std_normalization"],
                    "zca_whitening"                 : yml["Trainsetting"]["zca_whitening"],
                    "rotation_range"                : yml["Trainsetting"]["rotation_range"],
                    "width_shift_range"             : yml["Trainsetting"]["width_shift_range"],
                    "height_shift_range"            : yml["Trainsetting"]["height_shift_range"],
                    "horizontal_flip"               : yml["Trainsetting"]["horizontal_flip"],
                    "vertical_flip"                 : yml["Trainsetting"]["vertical_flip"],
                    "shear_range"                   : yml["Trainsetting"]["shear_range"]
                }
                datagen = ImageDataGenerator(**params)
            else:
                print("Operate as a generator without data augmentation")
                datagen = ImageDataGenerator(rescale = 1./255)

        # Avoiding Lint
        train_data = None
        if isValidation:
            # Create dataset list(for Validation.)
            if yml["Validation"]["readdata"] == "text" or yml["Validation"]["readdata"] == "TEXT":
                train_data = self.text_dataset(yml["Validation"]["resourcepath"])
            elif yml["Validation"]["readdata"] == "onefolder" or yml["Validation"]["readdata"] == "Onefolder":
                train_data = self.onefolder_dataet(yml["Validation"]["resourcepath"],
                                                    yml["Resourcedata"]["classes"])
            elif yml["Validation"]["readdata"] == "folder" or yml["Validation"]["readdata"] == "Folder":
                train_data = self.folder_dataset(yml["Validation"]["resourcepath"])
            else:
                print("It seems to have input an unsupported generator name. Stops the program.")
        else:
            # Create dataset list
            if yml["Resourcedata"]["readdata"] == "text" or yml["Resourcedata"]["readdata"] == "TEXT":
                train_data = self.text_dataset(yml["Resourcedata"]["resourcepath"])
            elif yml["Resourcedata"]["readdata"] == "onefolder" or yml["Resourcedata"]["readdata"] == "Onefolder":
                train_data = self.onefolder_dataet(yml["Resourcedata"]["resourcepath"],
                                                    yml["Resourcedata"]["classes"])
            elif yml["Resourcedata"]["readdata"] == "folder" or yml["Resourcedata"]["readdata"] == "Folder":
                train_data = self.folder_dataset(yml["Resourcedata"]["resourcepath"])
            else:
                print("It seems to have input an unsupported generator name. Stops the program.")
                exit(1)

        # take inputshape(width, height) and classes(list)
        inputshape = (yml["Resourcedata"]["img_row"], yml["Resourcedata"]["img_col"])

        # choice class mode(class 2 => binary, other => categorical)
        classes = yml["Resourcedata"]["classes"]
        print("=class name=")
        for i, c in enumerate(classes):
            print("{} => {}(Convert later to a one-hot expression)".format(c, i))
        classes = [str(i) for i in range(len(classes))]
        if len(classes) == 2:
            classmode = "binary"
        else:
            classmode = "categorical"
        print("class mode : {}".format(classmode))

        # check number of data
        datacount = len(train_data)
        print("number of data : {}".format(datacount))
        print("============")

        dataframe = pd.DataFrame(train_data, index = None, columns = ["filename", "class"])
        generator = datagen.flow_from_dataframe(dataframe,
                                                x_col="filename",
                                                y_col="class",
                                                target_size=inputshape,
                                                color_mode="rgb",
                                                classes=classes,
                                                class_mode=classmode,
                                                batch_size=yml["Trainsetting"]["batchsize"],
                                                shuffle=yml["Trainsetting"]["shuffle"])
        return generator, datacount