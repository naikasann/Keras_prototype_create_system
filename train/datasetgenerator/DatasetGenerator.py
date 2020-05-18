import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

import numpy as np
import os

class DatasetGenerator:
    # initial function
    def __init__(self):
        self.reset()
    
    # Empty the images list & labels list.
    def reset(self):
        self.images = []
        self.labels = []

    # for test program. It's better to alter it later.
    def text_dataset(self, resourcepath):
        input, answer = [], []
        # open text file.
        with open(resourcepath) as f:
            readlines = f.readlines()
        
        # Input data.
        for line in readlines:
            linebuffer = line.split(" ")
            input.append(linebuffer[0])
            answer.append(linebuffer[1])
            # return [imagepath], label
        return input, answer

    # count the number of data.
    def text_datacounter(self,datapath):
        # open resorse.
        with open(datapath) as f:
            readlines = f.readlines()
        # Returning the number of data
        return len(readlines)
    
    # Generator to read text.
    def text_dataset_generator(self, resourcepath, input_shape, classes, batchsize, shuffle=True):
        # open resorse.
        with open(resourcepath) as f:
            readlines = f.readlines()

        # data shuffle.
        if shuffle:
            shuffle_idx = np.random.permutation(len(readlines))
            readlines = readlines[shuffle_idx]

        # loop for generator.
        while True:
            for line in readlines:
                # ["image path" "label"] => linebuffer 
                linebuffer = line.split(" ")
                try:
                    # I'll load the image, and if it doesn't work, I'll terminate the program.
                    image = img_to_array(load_img(linebuffer[0], target_size=input_shape))
                except Exception as e:
                    print("Failed to load data.")
                    print("ERROR : ", e)
                    exit(1)
                 # Normalize image.
                image /= 255.0
                # list append.
                self.images.append(image)
                self.labels.append(to_categorical(linebuffer[1], len(classes)))

                # When the batch size is reached, it yields.
                if(len(self.images) == batchsize):
                    # Convert to Numpy array type
                    inputs = np.asarray(self.images, dtype = np.float32)
                    targets = np.asarray(self.labels, dtype = np.float32)
                    # Resetting an Instance
                    self.reset()
                    # for generator.
                    yield inputs, targets

    # for test program. It's better to alter it later.
    def onefolder_dataet(self, resourcepath, classes):
        input , answer = [], []

        input = os.listdir(resourcepath)
        for image in input:
            for count, category,  in enumerate(classes):
                # Find out which category. Identify the image name and assign a label.
                if category in image:
                    answer.append(count)
        # return [imagepath], [label]
        return input, answer

    # count the number of data.
    def onefolder_datacounter(self, datapath):
        image_list = os.listdir(datapath)
        return len(image_list)

    # Generator to read folder.
    def onefolder_dataset_generator(self, resourcepath, input_shape, classes, batchsize, shuffle=True):
        # Refers to all the contents of a folder. (Assign a list of image paths.)
        image_list = os.listdir(resourcepath)

        # data shuffle.
        if shuffle:
            shuffle_idx = np.random.permutation(len(image_list))
            image_list = image_list[shuffle_idx]

        # for generator.
        while True:
            for image_path in image_list:
                # image path read.
                image_path = os.path.join(resourcepath, image_path)
                try:
                    # I'll load the image, and if it doesn't work, I'll terminate the program.
                    image = img_to_array(load_img(image_path, target_size=input_shape))
                except Exception as e:
                    print("Failed to load data.")
                    print("ERROR : ", e)
                    exit(1)
                # Normalize image.
                image /= 255
                # list append.
                self.images.append(image)

                # Labeling based on the name of the image and the name of the category.
                for count, category in enumerate(classes):
                    # Check each category to see which label it was.
                    if category in image_path:
                        # Assigning labels.
                        self.labels.append(to_categorical(count, len(classes)))
                        # Assign "True" to the variable to check if the category is added or not.
                        append = True
                        break
                # Check the success of the label assignment.
                if not append:
                    print("Failed to assign a label.")
                    exit(1)
                else:
                    # Resetting the check for non-applicable labels.
                    append = False

                # When the batch size is reached, it yields.
                if(len(self.images) == batchsize):
                    inputs = np.asarray(self.images, dtype = np.float32)
                    targets = np.asarray(self.labels, dtype = np.float32)
                    self.reset()

                    yield inputs, targets

    # for test program. It's better to alter it later.
    def folder_dataset(self, resourcepath):
        input = []
        answer = []

        # folder check.
        labeldir = os.listdir(resourcepath)
        # Loop the number of label folders.
        for category, labelname in enumerate(labeldir):
            label_path = os.path.join(resourcepath, labelname)
            images = os.listdir(label_path)
            # Extracts data from a set of image files.
            for imagepath in images:
                input.append(os.path.join(label_path, imagepath))
                answer.append(category)
        # return [imagepath], [label] 
        return input, answer

    # count the number of data.
    def folder_datacounter(self, datapath):
        imagespath = []

        # folder check.
        labeldir = os.listdir(datapath)
        # Loop the number of label folders.
        for labels in labeldir:
            label_path = os.path.join(datapath, labels)
            images = os.listdir(label_path)
            # Extracts data from a set of image files.
            for imagepath in images:
                imagespath.append(os.path.join(label_path, imagepath))

        return len(imagespath)
    
    # Generator to read folder.
    def folder_dataset_generator(self, resourcepath, input_shape, classes, batchsize, shuffle=True):
        imagespath = []
        labels = []

        # folder check.
        labeldir = os.listdir(resourcepath)
        # Loop the number of label folders.
        for category, labels in enumerate(labeldir):
            label_path = os.path.join(resourcepath, labels)
            images = os.listdir(label_path)
            # Extracts data from a set of image files.
            # [imagepath], [label] 
            for imagepath in images:
                imagespath.append(os.path.join(label_path, imagepath))
                labels.append(category)
        
        if shuffle:
            shuffle_idx = np.random.permutation(len(imagepath))
            imagespath = imagepath[shuffle_idx]
            labels =labels[shuffle_idx]
        
        # for genarator.
        while True:
            for image, label in zip(imagespath, label):
                # input image & label.
                self.images.append(img_to_array(load_img(image, target_size=input_shape)) / 255.)
                self.labels.append(to_categorical(label, len(classes)))

                # When the batch size is reached, it yields.
                if(len(self.images) == batchsize):
                    inputs = np.asarray(self.images, dtype = np.float32)
                    targets = np.asarray(self.labels, dtype = np.float32)
                    self.reset()

                    yield inputs, targets