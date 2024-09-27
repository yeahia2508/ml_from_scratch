# -*- coding: utf-8 -*-
from keras.datasets import mnist
from cnn.architectures.lenet.conv_layer_lenet import ConvLayer
from cnn.pooling_layer.average_pool_layer import *
from activation_function.tanh import *
from cnn.flatten_layer import *
from ann.layer_dense import *
from activation_function.softmax import *
from loss_functions.categorical_cross_entropy import *
from optimizers.sgd import *
from loss_functions.softmax_loss import SoftmaxLossGrad
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
            ie      = iteration * epochs
            Monitor = np.zeros((ie,3))
            ct      = 0
            
            for it in range(iteration):
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
                
                loss = self.loss_activation.forward(self.dense_final.output, C)
                
                self.predictions = np.argmax(self.loss_activation.output, axis=1)
                if len(C.shape) == 2:
                    C = np.argmax(C, axis=1)
                self.accuracy = np.mean(self.predictions == C)
                
                
                ##backward
                self.loss_activation.backward(self.loss_activation.output, C)
                self.dense_final.backward(self.loss_activation.dinputs)
                self.activation_tanh[3].backward(self.dense_final.dinputs)
                self.dense1.backward(self.activation_tanh[3].dinputs)
                self.flat_layer.backward(self.dense1.dinputs)
                self.activation_tanh[2].backward(self.flat_layer.dinputs)
                self.conv3.backward(self.activation_tanh[2].dinputs)
                self.ap2.backward(self.conv3.dinputs)
                self.activation_tanh[1].backward(self.ap2.dinputs)
                self.conv2.backward(self.activation_tanh[1].dinputs)
                self.ap1.backward(self.conv2.dinputs)
                self.activation_tanh[0].backward(self.ap1.dinputs)
                self.conv1.backward(self.activation_tanh[0].dinputs)
                
                self.optimizer.pre_update_params()
                    
                self.optimizer.update_params(self.dense1)
                self.optimizer.update_params(self.dense_final)
                                    
                self.optimizer.update_params(self.conv1)
                self.optimizer.update_params(self.conv2)
                self.optimizer.update_params(self.conv3)
                    
                self.optimizer.post_update_params()
                
                Monitor[ct,0] = self.accuracy
                Monitor[ct,1] = loss
                Monitor[ct,2] = self.optimizer.current_learning_rate
                
                ct += 1
                
                print(f'epoch: {e}, ' +
                      f'iteration: {it}' +
                      f'accuracy: {self.accuracy: .3f}, ' +
                      f'loss: {loss: .3f}, ' +
                      f'current learning rate: {self.optimizer.current_learning_rate: .5f}')
                
            
            ## Save weights
            
            np.save('dense_weights1.npy', self.dense1.weights)
            np.save('dense_weights2.npy', self.dense_final.weights)
            
            np.save('dense_biases1.npy', self.dense1.biases)
            np.save('dense_biases2.npy', self.dense_final.biases)
            
            np.save('weightsC1.npy', self.conv1.weights)
            np.save('weightsC1.npy', self.conv2.weights)
            np.save('weightsC1.npy', self.conv3.weights)
            
            np.save('biasesC1.npy', self.conv1.biases)
            np.save('biasesC2.npy', self.conv2.biases)
            np.save('biasesC3.npy', self.conv3.biases)
            
            np.savetxt('Monitor.txt', Monitor)
            
    
    def evaluate(self, n = 5):
        testX = self.testX
        testY = self.testY
        
        idx = random.sample(range(len(testY)), n)
        
        M = testX[:,:,:,idx]
        C = testY[idx]
        
        n_neuron = self.n_neuron
        n_class = len(np.unique(testY))
        n_inputs = 480
        
        Conv1 = ConvLayer([5,5],6)
        Conv2 = ConvLayer([5,5],16)
        Conv3 = ConvLayer([5,5],120)
    
        AP1 = Average_Pool()
        AP2 = Average_Pool()
    
        T = [Tanh() for i in range(4)]
        F = FlattenLayer()
        
        dense1 = Layer_Dense(n_inputs, n_neuron)
        dense2 = Layer_Dense(n_neuron, n_class)
        
        Conv1.forward(M,0,1)
        T[0].forward(Conv1.output)
        AP1.forward(T[0].output,2,2)
        
        Conv2.forward(AP1.output,0,1)
        T[1].forward(Conv2.output)
        AP2.forward(T[1].output,2,2)
        
        Conv3.forward(AP2.output,2,3)
        T[2].forward(Conv3.output)
 
        F.forward(T[2].output)
        x = F.output
 
        dense1.forward(x)
        T[3].forward(dense1.output)
        dense2.forward(T[3].output)
        
        softmax = Activation_Softmax()
        softmax.forward(dense2.output)
        
        probabilities = softmax.output
        
        
        fig = plt.figure(figsize=(15, 7))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,\
                            wspace=0.05)

        # plot the images: each image is 28x28 pixels
        for i in range(n):
            ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(M[:,:,0,i].reshape((28,28)),cmap=plt.cm.gray_r,\
                      interpolation='nearest')
            
            predclass = np.ar3gmax(probabilities[i,:])
            trueclass = np.argmax(C[i])
            
            S = str(predclass)
          
            if predclass != trueclass:
                # label the image with the blue text
                P = str(round(probabilities[i,predclass],2))#probability
                ax.text(0, 3, S + ', P = ' + P, color = [0,128/255,0])
            else:
                # label the image with the red text
                ax.text(0, 3, S, color=[178/255,34/255,34/255])
                
        plt.savefig('evaluation results.pdf')
        plt.show()
            
    
                
        
        

        
        
        
        
        

