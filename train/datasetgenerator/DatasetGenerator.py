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
    
    def text_dataset_generator(self, yaml, file):
        path = yaml["Resorsedata"]["resorsepath"]

        print("read dataset...")
        # open text file.
        while True:
            try:
                print("open text file path : " + path)
                f = open(path)
            except Exception as e:
                print("An error occurred, such as a file not being found. exit program")
                print(error type : ", e)

            if(len(x_data) == yaml["Trainsetting"]["batchsize"]):
                input = np.asarray(self.images, dtype = np.float32)
                target = np.asarray(self.labels, dtype = np.float32)
                self.reset()

                yield input, target