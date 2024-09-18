

import numpy as np

class Tanh:
    def forward(self, input):
        tanh = np.tanh(input)
        self.output = tanh
        self.input= input
    
    def backward(self, dvalues):
        deriv = 1 - self.output**2
        deriv = np.nan_to_num(deriv)
        self.dinputs = np.multiply(deriv * dvalues)