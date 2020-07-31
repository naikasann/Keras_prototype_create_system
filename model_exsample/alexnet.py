import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.initializers import TruncatedNormal, Constant

# input shape (128, 128, 3)
def alexnet(self, input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    #conv1
    x = Conv2D(96, (11, 11), padding='same', strides=(4,4), kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01), activation='relu', bias_initializer=Constant(value=0), name='conv1')(input_layer)
    #pool1
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(x)
    x = BatchNormalization()(x)
    #conv2
    x = Conv2D(256, (5, 5), padding='same', strides=(1, 1), kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01), activation='relu',  bias_initializer=Constant(value=1), name='conv2')(x)
    #pool2
    x = MaxPooling2D(pool_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization()(x)
    #conv3
    x = Conv2D(384, (3, 3), padding='same', strides=(1, 1), kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01), activation='relu', bias_initializer=Constant(value=0), name='conv3')(x)
    #conv4
    x = Conv2D(384, (3, 3), padding='same', strides=(1, 1), kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01), activation='relu', bias_initializer=Constant(value=1), name='conv4')(x)
    #conv5
    x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01), activation='relu', bias_initializer=Constant(value=1), name='conv5')(x)
    #pool5
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(x)
    x = BatchNormalization()(x)
    #fc6
    x = Flatten()(x)
    x = Dense(4096, name='dense1', activation='relu')(x)
    x = Dropout(0.5)(x)
    #fc7
    x = Dense(4096, name='dense2', activation='relu')(x)
    x = Dropout(0.5)(x)
    #fc8
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x, name='model')
    return model