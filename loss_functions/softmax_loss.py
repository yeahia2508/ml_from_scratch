# -*- coding: utf-8 -*-
import numpy as np
from loss_functions.categorical_cross_entropy import CategoricalCrossEntropy
from activation_function.softmax import Activation_Softmax

class SoftmaxLossGrad():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        #the probabilities
        #calculates and returns mean loss
        return(self.loss.calculate(self.output, y_true))
        
    def backward(self, dvalues, y_true):
        Nsamples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        #calculating normalized gradient
        self.dinputs[range(Nsamples), y_true] -= 1
        self.dinputs = self.dinputs/Nsamples
