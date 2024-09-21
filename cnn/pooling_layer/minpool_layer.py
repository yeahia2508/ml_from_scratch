#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:04:17 2024

@author: yeahia
"""
import numpy as np

class MinPoolLayer:
    def forward(self, imageMatrix, kernelShape = [2,2], stride = 1):
        
        [imageHeight, imageWidth, colorChannelSize, batchSize] = imageMatrix.shape
        
        kernelHeight = kernelShape[0]
        kernelWidth = kernelShape[1]
        
        xOutput = int((imageHeight - kernelHeight) / stride + 1)
        yOutput = int((imageWidth - kernelWidth) / stride + 1)
        
        output = np.zeros((xOutput, yOutput, colorChannelSize, batchSize))
        
        image_matrix_copy = imageMatrix.copy().astype(float) * 0
        
        storeMinIndexs = np.zeros((kernelHeight * kernelWidth, 2, xOutput * yOutput, colorChannelSize, batchSize))
        storeNMin = np.zeros((xOutput*yOutput, colorChannelSize, batchSize))
        
        for i in range(batchSize):
            for c in range(colorChannelSize):
                window_index = 0
                for w in range(xOutput):
                    for h in range(yOutput):
                        window_index += 1
                        
                        xStart = h * stride
                        xEnd = xStart + kernelHeight
                        yStart = w * stride
                        yEnd = yStart + kernelWidth
                        
                        currentSlice = imageMatrix[xStart:xEnd, yStart:yEnd, c, i]
                        min_value = float(currentSlice.min())
                        output[h, w, c , i] = min_value
                        
                        (x_min_indexs, y_min_indexs) = np.where(currentSlice == min_value)
                        
                        for ii, (xx,yy) in enumerate(zip(x_min_indexs, y_min_indexs)):
                            storeMinIndexs[ii, 0, window_index -1, c, i] = xx
                            storeMinIndexs[ii, 1, window_index -1, c, i] = yy
                            
                        storeNMin[window_index - 1, c, i] = ii + 1
                        image_matrix_copy[xStart:xEnd, yStart:yEnd, c, i] += np.equal(imageMatrix[xStart:xEnd, yStart:yEnd, c, i], min_value).astype(float)
        
        
        mask = image_matrix_copy
        mask = np.matrix.round(mask/(mask + 1e-7))
        
        self.output = output
        self.mask = mask
        self.minIndexs = storeMinIndexs
        self.windowMinCount = storeNMin
        self.kernelShape = kernelShape
        self.stride = stride
        self.input = imageMatrix
    
    def backward(self, dvalues):
        [dx, dy, cs, bs] = dvalues.shape
        stride = self.stride
        dinputs = np.zeros(self.input.shape)
        
        for i in range(bs):
            for c in range(cs):
                window_index = 0
                for y in range(dy):
                    for x in range(dx):
                        window_index += 1
                        yStart = y * stride
                        xStart = x * stride
                        min_count = int(self.windowMinCount[window_index - 1, c , i])
                        
                        for ii in range(min_count):
                            xx = int(self.minIndexs[ii, 0, window_index -1, c, i])
                            yy = int(self.minIndexs[ii, 1, window_index -1, c, i])
                            
                            dinputs[xStart + xx, yStart + yy, c, i] = dvalues[x, y, c, i]
                            
        self.dinputs = dinputs
        
    
        
        
    