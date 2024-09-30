#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:53:46 2024

@author: yeahia
"""

from alexnet_tf import AlexNet, Model
from utils.dataset_loader import DatasetLoader
import matplotlib.pyplot as plt

dl = DatasetLoader()
train_data = dl.load_data_tf('/home/yeahia/Downloads/flower_photos', 32, 227, 227)
class_count = len(train_data.class_indices)
image_shape = train_data.image_shape

alex_net = AlexNet(image_shape, class_count)
model = Model(alex_net)
model.train(train_data)



