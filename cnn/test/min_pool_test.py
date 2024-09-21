# -*- coding: utf-8 -*-
from cnn.utils import *
from cnn.conv_layer import *
from cnn.maxpool_layer import *
from cnn.pooling_layer.minpool_layer import *
import matplotlib.pyplot as plt


readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('../images', 2, [120, 120])
convLayer1 = ConvLayer([3,3], 5)
minpool1 = MinPoolLayer()
maxpool = MaxPoolLayer()

convLayer1.forward(random_image_matrix, padding = 2, stride = 1)
minpool1.forward(random_image_matrix, stride = 1)
minpool1.backward(minpool1.output)
maxpool.forward(random_image_matrix, stride = 1)
maxpool.backward(maxpool.output)

im1 = convLayer1.output[:, : , 0, 0]
im2 = minpool1.output[:, :, 0, 0]
im3 = minpool1.mask[:, :, 0, 0]
im4 = minpool1.dinputs[:, :, 0, 0]
im5 = maxpool.dinputs[:, : , 0, 0]
im6 = maxpool.mask[:, : , 0, 0]

plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()
plt.imshow(im3)
plt.show()
plt.imshow(im4)
plt.show()
plt.imshow(im5)
plt.show()
plt.imshow(im6)
plt.show()
