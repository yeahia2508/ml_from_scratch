#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:10:26 2024

@author: yeahia
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout

from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

class AlexNet(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()
        self.add(Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape= input_shape, padding='valid'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.add(ZeroPadding2D(padding=2))
        self.add(Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
        self.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
        self.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(nb_classes, activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 1e-2,
            decay_steps = 10000,
            decay_rate = 0.98
        )
        
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
class CallData:
    def callCifar10(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()
        train_x = self.train_x / 255
        test_x = self.test_x / 255
        train_y = self.train_y
        test_y = self.test_y
        
        
        
