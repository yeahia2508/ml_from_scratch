from cnn.utils import *
from cnn.conv_layer import *
import matplotlib.pyplot as plt


readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('../images', 2, [120, 120])
convLayer1 = ConvLayer([3,3], 5)
convLayer1.forward(random_image_matrix, padding = 1, stride = 1)

im1 = convLayer1.output[:, :, 0, 0]
plt.imshow(im1)
plt.show()