#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:04:17 2024

@author: yeahia
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve as Conv

I = plt.imread('dog.jpg')
plt.imshow(I)
plt.show()

K1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# edges
K2 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
K3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
K4 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# sharpen
K5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# blur
K6 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
K6 = K6/9
K7 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

K8 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
# misc
K9 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
K10 = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
K11 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
K12 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
K13 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])

filters = np.dstack((K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13))

img_channel_size = I.shape[2]
filter_size = filters.shape[2]

plot_grid_size = np.math.ceil(filter_size ** 0.5)

for channel_index in range(0,img_channel_size):
    plt.figure(figsize = (15,16))
    plt.suptitle('After convulution of channel' + str(channel_index+1), fontsize=20, y= 0.95)
    plt.subplots_adjust(hspace=0.5)
    for filter_index in range(0, filter_size):
        subplot = plt.subplot(plot_grid_size, plot_grid_size, filter_index + 1)
        conv = Conv(I[:,:,channel_index], filters[:,:,filter_index])
        subplot.imshow(conv, cmap='gray')
        subplot.set_title('Kernel:' + str(filter_index + 1))
    plt.show()