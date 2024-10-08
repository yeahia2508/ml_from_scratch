# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:40:17 2024

@author: yeahia
"""

from cnn.utils import *
from cnn.conv_layer import *
from cnn.maxpool_layer import *
from cnn.flatten_layer import *
import matplotlib.pyplot as plt

readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('../images', 2, [100,100])
convLayer1 = ConvLayer([3,3], 5)
maxpool1 = MaxPoolLayer()
flatten_layer = FlattenLayer()

convLayer1.forward(random_image_matrix, padding = 0, stride = 1)
maxpool1.forward(convLayer1.output, kernelShape= [3,3], stride = 1)
flatten_layer.forward(maxpool1.output)
flatten_layer.backward(flatten_layer.output)

im1 = maxpool1.output[:, :, 0, 0]
im2 = flatten_layer.dinputs[:,:,0,0]

plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()