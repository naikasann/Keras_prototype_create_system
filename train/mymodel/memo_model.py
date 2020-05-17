import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Input
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D

import yaml

class memo():
    def vgg16model(self, input_shape, num_classes):
        print("create vgg16 model...")

        input_layer = Input(shape=input_shape)

        # Block 1
        conv1_1 = Conv2D(64, (3, 3),name='conv1_1', activation='relu', padding='same')(input_layer)
        conv1_2 = Conv2D(64, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

        # Block 2
        conv2_1 = Conv2D(128, (3, 3),name='conv2_1', activation='relu', padding='same')(pool1)
        conv2_2 = Conv2D(128, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        # Block 3
        conv3_1 = Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(pool2)
        conv3_2 = Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

        # Block 4
        conv4_1 = Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(pool3)
        conv4_2 = Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)

        # Block 5
        conv5_1 = Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(pool4)
        conv5_2 = Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_3)
        x = Dropout(0.5)(pool5)
        
        x = Flatten()(x)
        x = Dense(4096)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=x)

        return model

    #input shape (224, 224, 3)
    def vgg16model_beta(self, input_shape, num_classes):
        input_layer = Input(shape=input_shape)
        # Block 1
        conv1_1 = Conv2D(64, (3, 3),name='conv1_1', activation='relu', padding='same')(input_layer)
        conv1_2 = Conv2D(64, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
        bn1 = BatchNormalization(axis=3)(conv1_2)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        drop1 = Dropout(0.5)(pool1)
        # Block 2
        conv2_1 = Conv2D(128, (3, 3),name='conv2_1', activation='relu', padding='same')(drop1)
        conv2_2 = Conv2D(128, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
        bn2 = BatchNormalization(axis=3)(conv2_2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        drop2 = Dropout(0.5)(pool2)
        # Block 3
        conv3_1 = Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(drop2)
        conv3_2 = Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
        conv3_4 = Conv2D(256, (3, 3),name='conv3_4', activation='relu', padding='same')(conv3_3)
        bn3 = BatchNormalization(axis=3)(conv3_4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
        drop3 = Dropout(0.5)(pool3)
        # Block 4
        conv4_1 = Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(drop3)
        conv4_2 = Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
        conv4_4 = Conv2D(512, (3, 3),name='conv4_4', activation='relu', padding='same')(conv4_3)
        bn4 = BatchNormalization(axis=3)(conv4_4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
        drop4 = Dropout(0.5)(pool4)
        # Block 5
        conv5_1 = Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(drop4)
        conv5_2 = Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
        conv5_4 = Conv2D(512, (3, 3),name='conv5_4', activation='relu', padding='same')(conv5_3)
        bn5 = BatchNormalization(axis=3)(conv5_4)
        pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)
        drop5 = Dropout(0.5)(pool5)
        x = Flatten()(drop5)
        x = Dense(4096)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=x)

        return model
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