# -*- coding: utf-8 -*-

from cnn.utils import *
from cnn.conv_layer import *
from cnn.pooling_layer.maxpool_layer import *
from activation_function.relu import *
import matplotlib.pyplot as plt

readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('../images', 2, [200, 200])

maxpool_1 = MaxPoolLayer()
maxpool_2 = MaxPoolLayer()
maxpool_3 = MaxPoolLayer()
relu1 = Activation_ReLU()
relu2 = Activation_ReLU()
relu3 = Activation_ReLU()
conv1 = ConvLayer([3,3], 5)
conv2 = ConvLayer([5,5], 4)
conv3 = ConvLayer([2,2], 4)


##Forward pass

conv1.forward(random_image_matrix, 0, 1)
relu1.forward(conv1.output)
maxpool_1.forward(relu1.output)

conv2.forward(maxpool_1.output, 2, 3)
relu2.forward(conv2.output)
maxpool_2.forward(relu2.output)

conv3.forward(maxpool_2.output, 2, 3)
relu3.forward(conv3.output)
maxpool_3.forward(relu3.output)


plt.imshow(conv1.output[:, : , 0, 0])
plt.title('conv1')
plt.show()

plt.imshow(maxpool_1.output[:, : , 0, 0])
plt.title('maxpool_1')
plt.show()

plt.imshow(conv2.output[:, : , 0, 0])
plt.title('conv2')
plt.show()

plt.imshow(maxpool_2.output[:, : , 0, 0])
plt.title('maxpool_2')
plt.show()

plt.imshow(conv3.output[:, : , 0, 0])
plt.title('conv3')
plt.show()

plt.imshow(maxpool_3.output[:, : , 0, 0])
plt.title('maxpool_3')
plt.show()

## Backward pass
maxpool_3.backward(maxpool_3.output)
relu3.backward(maxpool_3.dinputs)
conv3.backward(relu3.dinputs)

maxpool_2.backward(conv3.dinputs)
relu2.backward(maxpool_2.dinputs)
conv2.backward(relu2.dinputs)

maxpool_1.backward(conv2.dinputs)
relu1.backward(maxpool_1.dinputs)
conv1.backward(relu1.dinputs)


plt.imshow(conv1.dinputs[:, : , 0, 0])
plt.title('conv1 dinput')
plt.show()

plt.imshow(maxpool_1.dinputs[:, : , 0, 0])
plt.title('maxpool_1 dinput')
plt.show()

plt.imshow(conv2.dinputs[:, : , 0, 0])
plt.title('conv2 dinput')
plt.show()

plt.imshow(maxpool_2.dinputs[:, : , 0, 0])
plt.title('maxpool_2 dinput')
plt.show()

plt.imshow(conv3.dinputs[:, : , 0, 0])
plt.title('conv3 dinput')
plt.show()

plt.imshow(maxpool_3.dinputs[:, : , 0, 0])
plt.title('maxpool_3 dinput')
plt.show()






















