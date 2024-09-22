# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:21:17 2024

@author: yeahia
"""

from cnn.utils import *
from cnn.conv_layer import *
from cnn.pooling_layer.maxpool_layer import *
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

im1 = convLayer1.output[:, : , 0, 0]
im2 = maxpool1.output[:, :, 0, 0]
im3 = maxpool1.mask[:, :, 0, 0]

plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()
plt.imshow(im3)
plt.show()




