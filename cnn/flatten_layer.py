# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:21:17 2024

@author: yeahia
"""

import numpy as np
class FlattenLayer:
    def forward(self, imageMatrix):
        [xImageSize, yImageSize, channelSize, batchSize] = imageMatrix.shape
        flatLayerSize = xImageSize * yImageSize * channelSize
        
        flat_data = np.zeros((batchSize, flatLayerSize))
        
        for i in range(batchSize):
            flat_data[i, :] = imageMatrix[:,:,:,i].reshape(1, flatLayerSize)
            
        self.output = flat_data
        self.input = imageMatrix
        
    
    def backward(self, dvalues):
        [xImage, yImage, channelSize, batchSize] = self.input.shape
        
        dinputs = np.zeros(self.input.shape)
        
        for i in range(batchSize):
            dinputs[:,:,:,i] = dvalues[i,:].reshape(xImage, yImage, channelSize)
            
        self.dinputs = dinputs
        
        