#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:53:46 2024

@author: yeahia
"""

from alexnet_tf import CallData
from utils.dataset_loader import DatasetLoader
import matplotlib.pyplot as plt


dl = DatasetLoader()
data = dl.load_data_tf('/home/yeahia/Downloads/flower_photos', batch_size = 32, image_height = 227, image_width = 227)


