import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical


class DatasetGenerator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x_data = []
        self.y_data = []
    
    def 