import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

import numpy as np

class DatasetGenerator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.images = []
        self.labels = []
    
    def text_dataset_generator(self, yml):
        path = yaml["Resorsedata"]["resorsepath"]
        input_shape = (yml["Resorsedata"]["img_row"], yml["Resorsedata"]["img_col"])
        classes = yml["Resorsedata"]["classes"]

        with open(path) as f:
            readlines = f.readlines()

        while True:
            for line in readlines:
                linebuffer = line.split(" ")


                try:
                    image = img_to_array(load_img(linebuffer[0], target_size=input_shape))
                    image /= 255.0
                except Exception as e:
                    print("Failed to load data.")
                    exit(1)
                
                self.images.append(image)
                self.labels.append(to_categorical(linebuffer[1], len(classes)))

                if(len(x_data) == yml["Trainsetting"]["batchsize"]):
                    input = np.asarray(self.images, dtype = np.float32)
                    target = np.asarray(self.labels, dtype = np.float32)
                    self.reset()

                    yield input, target
