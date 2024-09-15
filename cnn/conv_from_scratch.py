# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:28:17 2024

@author: yeahia
"""

import matplotlib.pyplot as plt
import numpy as np

def Conv3d(Image, K, padding = 0, stride = 1):
    
    [xImageShape, yImageShape, channelSize] = Image.shape
    [xK, yK] = K.shape
    
    xOutput = int((xImageShape - xK + 2 * padding ) / stride + 1)
    yOutput = int((yImageShape - yK + 2 * padding ) / stride + 1)
    
    output = np.zeros((xOutput, yOutput, channelSize))
    imagePadded = np.zeros((xImageShape + 2 * padding, yImageShape + 2*padding, channelSize))
    imagePadded[int(padding): int(padding + xImageShape), int(padding): int(padding + yImageShape), :] = Image
    
    
    for channelIndex in range(channelSize):
        for yIndex in range(yOutput):
            for xIndex in range(xOutput):
                yStart = yIndex * stride
                yEnd = yStart + yK
                xStart = xIndex * stride
                xEnd = xStart + xK
                
                currentSlice = imagePadded[xStart:xEnd, yStart:yEnd, channelIndex]
                imgSlice_k_mul = np.multiply(currentSlice, K)
                output[xIndex, yIndex, channelIndex] = np.sum(imgSlice_k_mul)
               
    plt.imshow(output.sum(2) , cmap = 'gray')
    plt.title('after convulution with padding = ' + str(padding) + ' and stride = ' + str(stride))
    plt.show()
    
    return output

img = plt.imread('dog.jpg')
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
Conv3d(img, K5, padding = 0, stride = 2)
Conv3d(img, K5, padding = 20, stride = 1)
Conv3d(img, K5, padding = 20, stride = 2)
Conv3d(img, K2, padding = 10, stride = 3)
