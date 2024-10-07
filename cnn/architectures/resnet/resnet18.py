#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:55:19 2024

@author: yeahia
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class ResNet18(Model):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        # Initial conv layer
        self.conv1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Stage 1
        self.conv_block1 = self._conv_block(filters=(64, 64), strides=(1, 1))
        self.identity_block1 = self._identity_block(filters=(64, 64))

        # Stage 2
        self.conv_block2 = self._conv_block(filters=(128, 128))
        self.identity_block2 = self._identity_block(filters=(128, 128))

        # Stage 3
        self.conv_block3 = self._conv_block(filters=(256, 256))
        self.identity_block3 = self._identity_block(filters=(256, 256))

        # Stage 4
        self.conv_block4 = self._conv_block(filters=(512, 512))
        self.identity_block4 = self._identity_block(filters=(512, 512))

        # Global average pooling and output layer
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def _identity_block(self, filters):
        f1, f2 = filters

        def block(x):
            shortcut = x
            x = layers.Conv2D(f1, kernel_size=(3, 3), padding='same')(x)  # Output: same as input
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(f2, kernel_size=(3, 3), padding='same')(x)  # Output: same as input
            x = layers.BatchNormalization()(x)

            x = layers.add([x, shortcut])
            x = layers.ReLU()(x)

            return x

        return block

    def _conv_block(self, filters, strides=(2, 2)):
        f1, f2 = filters

        def block(x):
            shortcut = layers.Conv2D(f2, kernel_size=(
                1, 1), strides=strides)(x)  # Output: reduced size
            shortcut = layers.BatchNormalization()(shortcut)

            x = layers.Conv2D(f1, kernel_size=(3, 3), strides=strides, padding='same')(x)  # Output: reduced size
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(f2, kernel_size=(3, 3), padding='same')(x)  # Output: same as input
            x = layers.BatchNormalization()(x)

            x = layers.add([x, shortcut])
            x = layers.ReLU()(x)

            return x

        return block

    def call(self, inputs):
        # Initial layers
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1
        x = self.conv_block1(x)
        x = self.identity_block1(x)

        # Stage 2
        x = self.conv_block2(x)
        x = self.identity_block2(x)

        # Stage 3
        x = self.conv_block3(x)
        x = self.identity_block3(x)

        # Stage 4
        x = self.conv_block4(x)
        x = self.identity_block4(x)

        # Global Average Pooling and final Dense Layer
        x = self.global_avg_pool(x)
        return self.fc(x)


class Model:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def train(self, train_data, validation_data, epochs=50):
        history = self.model.fit(
            train_data,
            steps_per_epoch=train_data.samples // self.batch_size,
            validation_data=validation_data,
            validation_steps=validation_data.samples // self.batch_size,
            epochs=epochs)

        return history

    def evaluate(self, data):
        loss, acc = self.model.evaluate(data)
        print(f"acc: {acc}, loss: {loss}")
