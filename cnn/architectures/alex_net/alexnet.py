# -*- coding: utf-8 -*-
from cnn.conv_layer import ConvLayer
from cnn.pooling_layer.maxpool_layer import MaxPoolLayer
from activation_function.relu import Activation_ReLU
from cnn.flatten_layer import FlattenLayer
from ann.layer_dense import Layer_Dense
from ann.layer_dropout import Layer_Dropout
from activation_function.softmax import Activation_Softmax
from loss_functions.categorical_cross_entropy import *
from optimizers.sgd import *
from loss_functions.softmax_loss import SoftmaxLossGrad
import matplotlib.pyplot as plt
import random
import numpy as np

class AlexNet:
    def __init__(self, dataset):
        self.extract_train_data(dataset)
        self.n_neuron = 4096
        self.n_input = 9216
        self.conv1 = ConvLayer([11,11], 96)
        self.conv1_relu = Activation_ReLU()
        self.mp1 = MaxPoolLayer()
        self.mp1_relu = Activation_ReLU()
        self.conv2 = ConvLayer([5,5], 256)
        self.conv2_relu = Activation_ReLU()
        self.mp2 = MaxPoolLayer()
        self.mp2_relu = Activation_ReLU()
        self.conv3 = ConvLayer([3,3], 384)
        self.conv3_relu = Activation_ReLU()
        self.conv4 = ConvLayer([3,3], 384)
        self.conv4_relu = Activation_ReLU()
        self.conv5 = ConvLayer([3,3], 256)
        self.mp3 = MaxPoolLayer()
        self.mp3_relu = Activation_ReLU()
        self.flat_layer = FlattenLayer()
        self.dense1 = Layer_Dense(self.n_input, self.n_neuron)
        self.dense1_relu = Activation_ReLU()
        self.dropout1 = Layer_Dropout(0.5)
        self.dense2 = Layer_Dense(self.n_neuron, self.n_neuron)
        self.dense2_relu = Activation_ReLU()
        self.dropout2 = Layer_Dropout(0.5)
        self.dense3 = Layer_Dense(self.n_neuron, self.n_category)
        self.loss_activation = SoftmaxLossGrad()
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
                     minibatch_size=1,
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
            ie      = iteration * epochs
            Monitor = np.zeros((ie,3))
            ct      = 0
            
            for it in range(iteration):
                self.conv1.forward(M, 0, 4)
                self.conv1_relu.forward(self.conv1.output)
                self.mp1.forward(self.conv1_relu.output, [3,3], 2)
                self.mp1_relu.forward(self.mp1.output)
                self.conv2.forward(self.mp1_relu.output, 2, 1)
                self.conv2_relu.forward(self.conv2.output)
                self.mp2.forward(self.conv2_relu.output, [3,3], 2)
                self.mp2_relu.forward(self.mp2.output)
                self.conv3.forward(self.mp2_relu.output, 1, 1)
                self.conv3_relu.forward(self.conv3.output)
                self.conv4.forward(self.conv3_relu.output, 1, 1)
                self.conv4_relu.forward(self.conv4.output)
                self.conv5.forward(self.conv4_relu.output, 1, 1)
                self.mp3.forward(self.conv5.output, [3,3], 2)
                self.mp3_relu.forward(self.mp3.output)
                self.flat_layer.forward(self.mp3_relu.output)
                self.dense1.forward(self.flat_layer.output)
                self.dense1_relu.forward(self.dense1.output)
                self.dropout1.forward(self.dense1_relu.output)
                self.dense2.forward(self.dense1.output)
                self.dense2_relu.forward(self.dense2.output)
                self.dropout2.forward(self.dense2_relu.output)
                self.dense3.forward(self.dense2_relu.output)
                loss = self.loss_activation.forward(self.dense3.output, C)

                self.predictions = np.argmax(self.loss_activation.output, axis=1)
                if len(C.shape) == 2:
                    C = np.argmax(C, axis=1)
                self.accuracy = np.mean(self.predictions == C)

                ##backpropagation
                self.loss_activation.backward(self.loss_activation.output, C)
                self.dense3.backward(self.loss_activation.dinputs)
                self.dropout2.backward(self.dense3.dinputs)
                self.dense2_relu.backward(self.dropout2.dinputs)
                self.dense2.backward(self.dense2_relu.dinputs)
                self.dropout1.backward(self.dense2.dinputs)
                self.dense1_relu.backward(self.dropout1.dinputs)
                self.dense1.backward(self.dense1_relu.dinputs)
                self.flat_layer.backward(self.dense1.dinputs)
                self.mp3_relu.backward(self.flat_layer.dinputs)
                self.mp3.backward(self.mp3_relu.dinputs)
                self.conv5.backward(self.mp3.dinputs)
                self.conv4_relu.backward(self.conv5.dinputs)
                self.conv4.backward(self.conv4_relu.dinputs)
                self.conv3_relu.backward(self.conv4.dinputs)
                self.conv3.backward(self.conv3_relu.dinputs)
                self.mp2_relu.backward(self.conv3.dinputs)
                self.mp2.backward(self.mp2_relu.dinputs)
                self.conv2_relu.backward(self.mp2.dinputs)
                self.conv2.backward(self.conv2_relu.dinputs)
                self.mp1_relu.backward(self.conv2.dinputs)
                self.mp1.backward(self.mp1_relu.dinputs)
                self.conv1.backward(self.mp1.dinputs)

                self.optimizer.pre_update_params()
    
                self.optimizer.update_params(self.dense1)
                self.optimizer.update_params(self.dense2)
                self.optimizer.update_params(self.dense3)
                
                self.optimizer.update_params(self.conv1)
                self.optimizer.update_params(self.conv2)
                self.optimizer.update_params(self.conv3)
                self.optimizer.update_params(self.conv4)
                self.optimizer.update_params(self.conv5)
                
                self.optimizer.post_update_params()

                ct += 1
                
                print(f'epoch: {e}, ' +
                      f'iteration: {it}' +
                      f'accuracy: {self.accuracy: .3f}, ' +
                      f'loss: {loss: .3f}, ' +
                      f'current learning rate: {self.optimizer.current_learning_rate: .5f}')
                
    def evaluate(self, n = 100):
        testX = self.testX
        testY = self.testY
        
        idx = random.sample(range(len(testY)), n)
        
        M = testX[:,:,:,idx]
        C = testY[idx]
        
        self.conv1.forward(M, 0, 4)
        self.conv1_relu.forward(self.conv1.output)
        self.mp1.forward(self.conv1_relu.output, [3,3], 2)
        self.mp1_relu.forward(self.mp1.output)
        self.conv2.forward(self.mp1_relu.output, 2, 1)
        self.conv2_relu.forward(self.conv2.output)
        self.mp2.forward(self.conv2_relu.output, [3,3], 2)
        self.mp2_relu.forward(self.mp2.output)
        self.conv3.forward(self.mp2_relu.output, 1, 1)
        self.conv3_relu.forward(self.conv3.output)
        self.conv4.forward(self.conv3_relu.output, 1, 1)
        self.conv4_relu.forward(self.conv4.output)
        self.conv5.forward(self.conv4_relu.output, 1, 1)
        self.mp3.forward(self.conv5.output, [3,3], 2)
        self.mp3_relu.forward(self.mp3.output)
        self.flat_layer.forward(self.mp3_relu.output)
        self.dense1.forward(self.flat_layer.output)
        self.dense1_relu.forward(self.dense1.output)
        self.dropout1.forward(self.dense1_relu.output)
        self.dense2.forward(self.dense1.output)
        self.dense2_relu.forward(self.dense2.output)
        self.dropout2.forward(self.dense2_relu.output)
        self.dense3.forward(self.dense2_relu.output)
        
        softmax = Activation_Softmax()
        softmax.forward(self.dense3.output)
        
        probabilities = softmax.output
        
        fig = plt.figure(figsize=(15, 7))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,\
                            wspace=0.05)

        # plot the images: each image is 227x227 pixels
        for i in range(n):
            ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(M[:,:,0,i].reshape((227,227)),cmap=plt.cm.gray_r,\
                      interpolation='nearest')
            
            predclass = np.argmax(probabilities[i,:])
            trueclass = np.argmax(C[i])
            
            S = str(predclass)
          
            if predclass == trueclass:
                # label the image with the blue text
                P = str(round(probabilities[i,predclass],2))#probability
                ax.text(0, 3, S + ', P = ' + P, color = [0,128/255,0])
            else:
                # label the image with the red text
                ax.text(0, 3, S, color=[178/255,34/255,34/255])
                
        plt.savefig('evaluation results.pdf')
        plt.show()
        
    