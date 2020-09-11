import numpy as np
import cv2
from scipy.interpolate.fitpack import spalde

class BCLearningGenerator:
    # initial function
    def __init__(self):
        self.reset()

    # Empty the images list & labels list.
    def reset(self):
        self.images = []
        self.labels = []

    def bclearning_generator(self, generator, batchsize, sample_steps):
        while True:
            # Take out the original image for mixing.
            original_images , original_labels = next(generator)


            ###################################
            if original_images.shape[0] == sample_steps and original_labels[0] == sample_steps:
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
                    random_proportion = 1 - random_proportion
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
                    inputs = np.asarray(self.images, np.float32)
                    targets = np.asarray(self.labels, np.float32)
                    self.reset()

                    yield inputs, targets