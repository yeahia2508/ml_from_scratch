# -*- coding: utf-8 -*-
from cnn.architectures.alex_net.alexnet import *
from cnn.utils import *

readImage = ReadImageClass()
trainY, trainX = readImage.get_batch_image_with_class('../../../images', 2, [227, 227])
testY, testX = readImage.get_batch_image_with_class('../../../images', 2, [227, 227])

dataset  = (trainX, trainY), (testX, testY)
model = AlexNet(dataset)
model.run_training()
model.evaluate()

