#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 07:59:01 2024

@author: yeahia
"""

import numpy as np

class MaxPoolLayer2D:
    def forward(self, input, ks = [2,2], stride = 1):
        [xInput, yInput, batch_size] = input.shape
        xK = ks[0]
        yK = ks[1]
        
        xOutput = int((xInput - xK) / stride + 1)
        yOutput = int((yInput - yK) / stride + 1)
        
        output = np.zeros((xOutput, yOutput, batch_size))
        
        storeMax = np.zeros((xK*yK, 2, xOutput * yOutput, batch_size))
        storeNMax = np.zeros((xOutput*yOutput, batch_size))
        
        
        
        mask = input.copy() * 0
        
        for i in range(batch_size):
            ct = 0
            for y in range(yOutput):
                for x in range(xOutput):
                    ct += 1
                    
                    yStart = y * stride
                    yEnd = yStart + yK
                    xStart = x * stride
                    xEnd = xStart + xK
                    
                    input_slice = input[xStart: xEnd, yStart: yEnd, i]
                    max_value = np.max(input_slice)
                    output[x, y, i] = max_value
                    
                    (xsm, ysm) = np.where(input_slice == max_value)
                    for ii, (xx, yy) in enumerate(zip(xsm, ysm)):
                        storeMax[ii, 0, ct-1, i] = int(xx)
                        storeMax[ii, 1, ct-1, i] = int(yy)
                    
                    storeNMax[ct-1, i] = ii+1
                    mask[xStart: xEnd, yStart:yEnd, i] += np.equal(input[xStart:xEnd, yStart:yEnd, i], max_value).astype(float)
                
        
        self.mask = np.matrix.round(mask/(mask + 1e-7))
        self.output = output
        self.input = input
        self.stride = stride
        self.storeMax = storeMax
        self.storeNMax = storeNMax
        
    def backward(self, dvalues):
        [xd, yd, batch_size] = dvalues.shape
        storeMax = self.storeMax
        storeNMax = self.storeNMax
        stride = self.stride
        
        dinputs = np.zeros(self.input.shape)
        
        for i in range(batch_size):
            ct = 0
            for y in range(yd):
                for x in range(xd):
                    ct += 1
                    yStart = y * stride
                    xStart = x * stride
                    nMax = int(storeNMax[ct-1, i])
                    
                    for ii in range(nMax):
                        xm = int(storeMax[ii, 0, ct-1, i])
                        ym = int(storeMax[ii, 1, ct -1, i])
                        
                        dinputs[xStart + xm, yStart + ym, i] = dvalues[x,y, i]
                    
        self.dinputs = dinputs
                
        


        



