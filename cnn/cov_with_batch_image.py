#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:36:17 2024

@author: yeahia
"""

from utils import ReadImageClass
from conv_layer import ConvLayer

        
readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('images', 2, [120,120])
convLayer1 = ConvLayer([3,3], 5)
convLayer2 = ConvLayer([5,5], 8)

convLayer1.forward(random_image_matrix, padding = 1, stride = 1)
convLayer2.forward(convLayer1.output, padding = 1, stride = 2)