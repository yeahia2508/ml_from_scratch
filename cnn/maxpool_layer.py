# -*- coding: utf-8 -*-
import numpy as np

class MaxPoolLayer:
        
    def forward(self, imageMatrix, kernelShape = [2,2], stride = 1):
        self.inputs = imageMatrix
        self.kernelShape = kernelShape
        self.stride = stride
        
        [imageHeight, imageWidth, colorChannelSize, batchSize] = imageMatrix.shape
        
        kernelHeight = kernelShape[0]
        kernelWidth = kernelShape[1]
        
        xOutput = int((imageHeight - kernelHeight) / stride + 1)
        yOutput = int((imageWidth - kernelWidth) / stride + 1)
        
        output = np.zeros((xOutput, yOutput, colorChannelSize, batchSize))
        
        image_matrix_copy = imageMatrix.copy() * 0
        
        for i in range(batchSize):
            for c in range(colorChannelSize):
                for w in range(xOutput):
                    for h in range(yOutput):
                        xStart = h * stride
                        xEnd = xStart + kernelHeight
                        yStart = w * stride
                        yEnd = yStart + kernelWidth
                        
                        currentSlice = imageMatrix[xStart:xEnd, yStart:yEnd, c, i]
                        max_value = float(currentSlice.max())
                        output[h, w, c , i] = max_value
                        
                        image_matrix_copy[xStart:xEnd, yStart:yEnd, c, i] += np.equal(imageMatrix[xStart:xEnd, yStart:yEnd, c, i], max_value).astype(float)
        
        mask = image_matrix_copy
        mask = np.matrix.round(mask/(mask + 1e-7))
        
        self.output = output

        self.mask = mask

