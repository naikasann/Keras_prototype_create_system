import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

import numpy as np
import cv2
import os
import random

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
            random.shuffle(readlines)

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
            random.shuffle(image_list)

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
            trainlist = list(zip(imagepath, labels))
            random.shuffle(trainlist)
            imagespath, labels = zip(*trainlist)
        
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

    def bclearning_generator(self, resourcepath, input_shape, classes, batchsize, shuffle=True):
        # Retrieves a generator function in a class.
        generator = self.text_dataset_generator(resourcepath, input_shape, classes, batchsize, shuffle)
        while True:
            # Take out the original image for mixing.
            original_images , original_labels = next(generator)
            # onehot label => label
            labels = np.argmax(original_labels, axis = 1)
            for original_image, label, original_label in zip(original_images, labels, original_labels):
                # List whether it is a different category or not
                diff_labels = np.where(labels != label)
                # Randomly select the images to be mixed from the list.
                mix_idx = np.random.choice(diff_labels[0].tolist())
                # Generates a random percentage of images to be mixed.
                random_proportion = np.random.rand()
                # The amount of features of the main image is set to be more than that of the main image.
                # (may not be necessary?)
                if random_proportion < 0.5:
                    random_proportion - 1
                # Mix the images and labels.

                mix_img = random_proportion * original_image + (1 - random_proportion) * original_images[mix_idx]
                mix_label = random_proportion * original_label + (1 - random_proportion) * original_labels[mix_idx]
                
                # Save the BCLearning image appropriately.
                # Basically, comments, if you don't stop the program in the middle, the image will swell up.
                # mix_img = cv2.cvtColor(mix_img, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("./bclearning/bcresult" + str(random_proportion) + ".jpg", mix_img * 255)

                # input image & label.
                self.images.append(mix_img)
                self.labels.append(mix_label)
            
                # When the batch size is reached, it yields.
                if(len(self.images) == batchsize):
                    inputs = np.asarray(self.images, dtype = np.float32)
                    targets = np.asarray(self.labels, dtype = np.float32)
                    self.reset()

                    yield inputs, targets