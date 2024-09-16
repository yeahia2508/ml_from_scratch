#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:36:17 2024

@author: yeahia
"""
import glob as gl
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np

class FileExtractor:
    
    def get_folder_image_paths(self, path):
        self.folders = []
        self.image_paths = []
        for folder in os.listdir(path):
            self.folders.append(folder)
            folder_path = os.path.join(path, folder)
            folder_image_paths = []
            for image in os.listdir(folder_path):
                folder_image_paths.append( os.path.join(folder_path, image))
            
            self.image_paths.append(folder_image_paths)
        
        return self.folders, self.image_paths
    

class ReadImageClass:
    def get_image_class(self, path):
        fileExtractor = FileExtractor()
        img_folders, images = fileExtractor.get_folder_image_paths(path)
        all_classes = []
        for index, folder in enumerate(img_folders):
            img_class_values = np.zeros((len(images[index]))) + index
            all_classes.append(img_class_values)
            
        return np.array(all_classes), np.array(images)
    
    
    def get_random_sample(self, path, batch_size):
        classes, images = self.get_image_class(path)
        flat_classes = classes.flatten()
        flat_images = images.flatten()
        total_samples = len(flat_images)
        idx = random.sample(range(total_samples), batch_size)
        return flat_classes[idx], flat_images[idx]
    
    def get_batch_image_with_class(self, path, batch_size, image_shape):
        img_classes, img_files = self.get_random_sample(path, batch_size)
        
        imageHeight = image_shape[0]
        imageWidth = image_shape[1]
        imageMatrix = np.zeros((imageHeight, imageWidth, 3, batch_size), dtype=np.uint8)
        
        for i in range(batch_size):
            I = Image.open(img_files[i])
            Ire = I.resize((imageHeight, imageWidth))
            imageArray = np.array(Ire)
            
            if(len(imageArray.shape) != 3):
                I3D = np.zeros((imageHeight, imageWidth, 3))
                I3D[:,:,0] = imageArray
                I3D[:,:,1] = imageArray
                I3D[:,:,2] = imageArray
                imageArray = I3D
            
            imageMatrix[:,:,:,i] = imageArray
            
        imageMatrix.astype(np.uint8)
        
        return (img_classes, imageMatrix)
    
    
class ConvLayer:
    def __init__(self, kShape, kNumber):
        self.xKShape = kShape[0]
        self.yKShape = kShape[1]
        self.kNumber = kNumber
        
        self.weights = np.random.rand(self.xKShape, self.yKShape, kNumber)
        self.biases = np.random.rand(1, kNumber)
    
    def forward(self, imageMatrix, padding = 0, stride = 1):
        self.padding = padding
        self.stride = stride
        [xImageShape, yImageShape, channelSize, batchSize] = imageMatrix.shape
        
        xOutput = int((xImageShape - self.xKShape + 2 * padding ) / stride + 1)
        yOutput = int((yImageShape - self.yKShape + 2 * padding ) / stride + 1)
        
        output = np.zeros((xOutput, yOutput, channelSize, self.kNumber, batchSize))
        
        imagePadded = np.zeros((xImageShape + 2 * padding, yImageShape + 2*padding, channelSize, self.kNumber, batchSize))
        
        for k in range(self.kNumber):
            imagePadded[int(padding): int(padding + xImageShape), int(padding): int(padding + yImageShape), :,k,:] = imageMatrix
        
        
        
        for i in range(batchSize):
            for k in range(self.kNumber):
                for c in range(channelSize):
                    for x in range(xOutput):
                        for y in range(yOutput):
                            yStart = y * stride
                            yEnd = yStart + self.yKShape
                            xStart = x * stride
                            xEnd = xStart + self.xKShape
                            
                            currentSlice = imagePadded[xStart:xEnd, yStart:yEnd, c, k, i]
                            imgSlice_k_mul = np.multiply(currentSlice, self.weights[:, :, k])
                            output[x, y, c, k, i] = np.sum(imgSlice_k_mul) + self.biases[0, k].astype(float)
                            
        self.output =  output.sum(2)
        self.input = imageMatrix
        self.paddedInput = imagePadded
        
        

        
readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('images', 2, [120,120])
convLayer1 = ConvLayer([3,3], 5)
convLayer2 = ConvLayer([5,5], 8)

convLayer1.forward(random_image_matrix, padding = 1, stride = 1)
convLayer2.forward(convLayer1.output, padding = 1, stride = 2)