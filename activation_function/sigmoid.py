# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:19:17 2024

@author: yeahia
"""
import numpy as np

class Sigmoid:
    def forward(self, input):
        sigm = np.clip(1/(1 + np.exp(-input)), 1e-7, 1-1e-7)
        self.output = sigm
        self.input = input
        
    def backward(self, dvalues):
        sigm = self.output
        deriv = np.multiply(sigm, (1-sigm))
        self.dinputs = np.multiply(deriv * dvalues)
        
