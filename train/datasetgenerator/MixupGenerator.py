from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import cv2

class MixupGenerator:
    def __init__(self, traindata, yml, alpha=0.2):
        self.x_train = []
        self.y_train = []
        self.alpha = alpha
        self.yml = yml

        self.reset()

        for x, y in traindata:
            self.x_train.append(x)
            self.y_train.append(y)
        print(self.x_train)

    def reset(self):
        self.images = []
        self.labels = []
        self.x_list = []
        self.y_list = []

    def target_generator(self):
        inputshape = (self.yml["Resourcedata"]["img_row"], self.yml["Resourcedata"]["img_col"])

        p = list(zip(self.x_train, self.y_train))
        random.shuffle(p)
        x, y = zip(*p)

        while True:
            for count, (x, y) in enumerate(zip(self.x_train, self.y_train)):
                image = load_img(x, grayscale=False, color_mode='rgb', target_size=inputshape)
                image = img_to_array(image)/255.

                label = to_categorical(y, num_classes=len(self.yml["Resourcedata"]["classes"]))

                self.x_list.append(image)
                self.y_list.append(label)

                if count == self.batchsize:
                    target_images = np.asarray(self.x_list, np.float32)
                    target_labels = np.asarray(self.y_list, np.float32)

                    yield target_images, target_labels

    def mixup_generator(self, generator, batchsize):
        self.batchsize = batchsize
        # Take out the original image for mixing.
        target_generator = self.target_generator()
        while True:
            # Take out the original image for mixing.
            original_images, original_labels = next(generator)
            target_images, target_labels = next(target_generator)
            # Generate a lambda (a random number of beta distribution).
            l = np.random.beta(self.alpha, self.alpha, self.batchsize)

            for (image, label, target_image, target_label, mix) in zip(original_images, original_labels, target_images, target_labels, l):
                image = image * mix + target_image * (1 - mix)
                label = label * mix + target_label * (1 - mix)
                self.images.append(image)
                self.labels.append(label)

            inputs = np.asarray(self.images, np.float32)
            targets = np.asarray(self.labels, np.float32)
            self.reset()

            yield inputs, targets