# -*- coding: utf-8 -*-
from keras.datasets import mnist
from cnn.conv_layer import *
from cnn.pooling_layer.average_pool_layer import *
from activation_function.tanh import *
from cnn.flatten_layer import *
from ann.layer_dense import *
from activation_function.softmax import *
from loss_functions.categorical_cross_entropy import *
from optimizers.sgd import *
import matplotlib.pyplot as plt
import random
import numpy as np

class LeNet:
    def __init__(self, dataset):
        self.extract_train_data(dataset)
        
        self.n_neuron = 84
        self.n_input = 480
        self.conv1 = ConvLayer([5,5], 6)
        self.conv2 = ConvLayer([5,5], 16)
        self.conv3 = ConvLayer([5,5], 120)
        self.ap1 = Average_Pool()
        self.ap2 = Average_Pool()
        self.activation_tanh = [Tanh() for i in range(4)]
        self.flat_layer = FlattenLayer()
        self.dense1 = Layer_Dense(self.n_input, self.n_neuron)
        self.dense_final = Layer_Dense(self.n_neuron, self.n_category)
        self.activation_softmax = Activation_Softmax()
        self.loss = CategoricalCrossEntropy()
        self.optimizer = Optimizer_SGD()
        
        
    def extract_train_data(self, dataset):
        (trainX, trainY), (testX, testY) = dataset
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.n_category = len(np.unique(self.trainY))
        self.n_total = range(len(trainY))
        
        
    
    def run_training(self,
                     minibatch_size=64,
                     iteration=1,
                     epochs=1,
                     learning_rate=0.1,
                     decay=0.001,
                     momentum=0.5,
                     saved_weights="NO"):
        n_total = self.n_total
        
        for e in range(epochs):
            idx = random.sample(n_total, minibatch_size)
            M = self.trainX[:,:,:, idx]
            C = self.trainY[idx]
            
            for i in range(iteration):
                self.conv1.forward(M,0,1)
                self.activation_tanh[0].forward(self.conv1.output)
                self.ap1.forward(self.activation_tanh[0].output, 2, 2)
                
                self.conv2.forward(self.ap1.output, 0,1)
                self.activation_tanh[1].forward(self.conv2.output)
                self.ap2.forward(self.activation_tanh[1].output, 2, 2)
                
                self.conv3.forward(self.ap2.output, 2, 3)
                self.activation_tanh[2].forward(self.conv3.output)
                
                self.flat_layer.forward(self.activation_tanh[2].output)
                self.dense1.forward(self.flat_layer.output)
                self.activation_tanh[3].forward(self.dense1.output)
                self.dense_final.forward(self.activation_tanh[3].output)
                
                self.activation_softmax.forward(self.dense_final.output)
                self.loss.forward(self.activation_softmax.output, C)
                
                self.predictions = np.argmax(self.activation_softmax.output, axis=1)
                if len(C.shape) == 2:
                    C = np.argmax(C, axis=1)
                self.accuracy = np.mean(self.predictions == C)
                
                
                ##backward
                self.loss.backward(self.activation_softmax.output, C)
                
                
                
                
        
        

        
        
        
        
        

