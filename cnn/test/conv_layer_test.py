from cnn.utils import *
from cnn.conv_layer import *
from cnn.maxpool_layer import *
from cnn.pooling_layer.minpool_layer import *
import matplotlib.pyplot as plt


readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('../images', 2, [120, 120])
convLayer1 = ConvLayer([3,3], 5)