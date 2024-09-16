#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 17:59:17 2024

@author: yeahia
"""

import numpy as np

class ConvLayer:
    def __init__(self, kShape, kNumber):
        self.xKShape = kShape[0]
        self.yKShape = kShape[1]
        self.kNumber = kNumber
        
        self.weights = np.random.rand(self.xKShape, self.yKShape, kNumber)
        self.biases = np.random.rand(1, kNumber)
    
    def forward(self, imageMatrix, padding = 0, stride = 1):
        self.padding = padding
        self.stride = stride
        [xImageShape, yImageShape, channelSize, batchSize] = imageMatrix.shape
        
        xOutput = int((xImageShape - self.xKShape + 2 * padding ) / stride + 1)
        yOutput = int((yImageShape - self.yKShape + 2 * padding ) / stride + 1)
        
        output = np.zeros((xOutput, yOutput, channelSize, self.kNumber, batchSize))
        
        imagePadded = np.zeros((xImageShape + 2 * padding, yImageShape + 2*padding, channelSize, self.kNumber, batchSize))
        
        for k in range(self.kNumber):
            imagePadded[int(padding): int(padding + xImageShape), int(padding): int(padding + yImageShape), :,k,:] = imageMatrix
        
        
        
        for i in range(batchSize):
            for k in range(self.kNumber):
                for c in range(channelSize):
                    for x in range(xOutput):
                        for y in range(yOutput):
                            yStart = y * stride
                            yEnd = yStart + self.yKShape
                            xStart = x * stride
                            xEnd = xStart + self.xKShape
                            
                            currentSlice = imagePadded[xStart:xEnd, yStart:yEnd, c, k, i]
                            imgSlice_k_mul = np.multiply(currentSlice, self.weights[:, :, k])
                            output[x, y, c, k, i] = np.sum(imgSlice_k_mul) + self.biases[0, k].astype(float)
                            
        self.output =  output.sum(2)
        self.input = imageMatrix
        self.paddedInput = imagePadded# -*- coding: utf-8 -*-

