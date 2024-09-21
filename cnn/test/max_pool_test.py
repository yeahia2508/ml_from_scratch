# -*- coding: utf-8 -*-
from cnn.utils import *
from cnn.conv_layer import *
from cnn.pooling_layer.maxpool_layer import *
import matplotlib.pyplot as plt


readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class(
    '../images', 2, [120, 120])
convLayer1 = ConvLayer([3,3], 5)
maxpool1 = MaxPoolLayer()

convLayer1.forward(random_image_matrix, padding = 2, stride = 1)
maxpool1.forward(random_image_matrix, stride = 2)
maxpool1.backward(maxpool1.output)

im1 = convLayer1.output[:, : , 0, 0]
im2 = maxpool1.output[:, :, 0, 0]
im3 = maxpool1.mask[:, :, 0, 0]
im4 = maxpool1.dinputs[:, :, 0, 0]

plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()
plt.imshow(im3)
plt.show()
plt.imshow(im4)
plt.show()
