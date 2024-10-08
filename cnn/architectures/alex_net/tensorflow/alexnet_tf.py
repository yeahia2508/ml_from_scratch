#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:10:26 2024

@author: yeahia
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Input
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_loader import DatasetLoader
from utils.callbacks import TfCallback
import datetime

class AlexNet(Sequential):
    def __init__(self, input_shape, nb_classes, l2 = 0.001):
        super().__init__()
        self.add(Input(shape=input_shape)),
        self.add(Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(l2)))
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.add(ZeroPadding2D(padding=2))
        self.add(Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2)))
        self.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2)))
        self.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(l2)))
        self.add(Flatten())
        self.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l2)))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l2)))
        self.add(Dropout(0.5))
        self.add(Dense(nb_classes, activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 1e-4,
            decay_steps = 875,
            decay_rate = 0.95
        )
        
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.summary()
        


class Model:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        
    def train(self, train_data, validation_data, epochs = 50):
        history = self.model.fit(
              train_data,
              steps_per_epoch= train_data.samples // self.batch_size,
              validation_data=validation_data,
              validation_steps=validation_data.samples // self.batch_size,
              epochs= epochs)
        
        return history;
        
    def evaluate(self, data):
        loss, acc = self.model.evaluate(data)
        print(f"acc: {acc}, loss: {loss}")
        
        
        
        
        
