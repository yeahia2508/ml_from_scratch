# -*- coding: utf-8 -*-
import numpy as np

class MaxPoolLayer:
        
    def forward(self, imageMatrix, kernelShape = [2,2], stride = 1):
        
        [imageHeight, imageWidth, colorChannelSize, batchSize] = imageMatrix.shape
        
        kernelHeight = kernelShape[0]
        kernelWidth = kernelShape[1]
        
        xOutput = int((imageHeight - kernelHeight) / stride + 1)
        yOutput = int((imageWidth - kernelWidth) / stride + 1)
        
        output = np.zeros((xOutput, yOutput, colorChannelSize, batchSize))
        
        image_matrix_copy = imageMatrix.copy().astype(float) * 0
        
        storeMaxIndexs = np.zeros((kernelHeight * kernelWidth, 2, xOutput * yOutput, colorChannelSize, batchSize))
        storeNMax = np.zeros((xOutput*yOutput, colorChannelSize, batchSize))
        
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
                        max_value = float(currentSlice.max())
                        output[h, w, c , i] = max_value
                        
                        (x_max_indexs, y_max_indexs) = np.where(currentSlice == max_value)
                        
                        for ii, (xx,yy) in enumerate(zip(x_max_indexs, y_max_indexs)):
                            storeMaxIndexs[ii, 0, window_index -1, c, i] = xx
                            storeMaxIndexs[ii, 1, window_index -1, c, i] = yy
                            
                        storeNMax[window_index - 1, c, i] = ii + 1
                        image_matrix_copy[xStart:xEnd, yStart:yEnd, c, i] += np.equal(imageMatrix[xStart:xEnd, yStart:yEnd, c, i], max_value).astype(float)
        
        
        mask = image_matrix_copy
        mask = np.matrix.round(mask/(mask + 1e-7))
        
        self.output = output
        self.mask = mask
        self.maxIndexs = storeMaxIndexs
        self.windowMaxCount = storeNMax
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
                        max_count = int(self.windowMaxCount[window_index - 1, c , i])
                        
                        for ii in range(max_count):
                            xx = int(self.maxIndexs[ii, 0, window_index -1, c, i])
                            yy = int(self.maxIndexs[ii, 1, window_index -1, c, i])
                            
                            dinputs[xStart + xx, yStart + yy, c, i] = dvalues[x, y, c, i]
                            
        self.dinputs = dinputs
        
        

