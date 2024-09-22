# -*- coding: utf-8 -*-

from dataset_loader import *
import matplotlib.pyplot as plt


#load mnist data
dc = DatasetLoader()
dc.load_mnist_data()

(trainX, trainY), (testX, textY) = dc.mnist_data
plt.imshow(trainX[:,:,0,0])
plt.show()

plt.imshow(testX[:,:,0,0])
plt.show()


