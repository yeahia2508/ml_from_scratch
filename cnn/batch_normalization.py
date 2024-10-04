# -*- coding: utf-8 -*-
import numpy as np

class BatchNormalization:
    def __init__(self, n_features, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum

        
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))

        
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

    def forward(self, X, training=True):
        if training:
            
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            self.X_normalized = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = self.gamma * self.X_normalized + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            X_normalized = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_normalized + self.beta

        return out

    def backward(self, dout):
        N, D = dout.shape

        dgamma = np.sum(dout * self.X_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)

        dX_normalized = dout * self.gamma

        dvar = np.sum(dX_normalized * (self.X - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0)

        dmean = np.sum(dX_normalized * -1.0 / np.sqrt(self.batch_var + self.epsilon), axis=0) + \
                dvar * np.sum(-2.0 * (self.X - self.batch_mean), axis=0) / N

        dX = dX_normalized * 1.0 / np.sqrt(self.batch_var + self.epsilon) + dvar * 2.0 * (self.X - self.batch_mean) / N + dmean / N

        return dX, dgamma, dbeta

