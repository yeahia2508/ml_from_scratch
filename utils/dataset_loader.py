# -*- coding: utf-8 -*-
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from pathlib import Path

class DatasetLoader:
    def load_mnist_data(self):
        (trainX, trainY), (testX, testY) = mnist.load_data()
        
        #convert shape (sampleSize x Height x Width) to (Height x Width x sampleSize)
        trainX = trainX.transpose(1,2,0)
        testX = testX.transpose(1,2,0)
        
        #convert single channel image into 3 channel image for both test and train data
        shape_train = trainX.shape
        shape_test = testX.shape
        
        trainX_3d = np.zeros((shape_train[0], shape_train[1], 3, shape_train[2]))
        textX_3d = np.zeros((shape_test[0], shape_test[1], 3, shape_test[2]))
        
        for imageChannelIndex in range(2):
            trainX_3d[:,:,imageChannelIndex,:] = trainX
            textX_3d[:,:,imageChannelIndex,:] = testX
            
        self.mnist_data = (trainX_3d, trainY), (textX_3d, testY)
        
    def load_data_tf(self, data_dir, batch_size = 32, image_height = 227, image_width = 227):
        data_dir = Path(data_dir)
        class_names = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" ])
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(image_height, image_width), #Resizing the raw dataset
                                                     classes = list(class_names))
        return train_data_gen
        
            
        

