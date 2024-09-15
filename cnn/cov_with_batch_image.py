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
        imageMatrix = np.zeros((imageHeight, imageWidth, 3, batch_size))
        
        for i in range(batch_size):
            I = Image.open(img_files[i])
            Ire = I.resize((imageHeight, imageWidth))
            imageArray = np.array(Ire)
            
            if(len(imageArray.shape) != 3):
                I3D = np.zeros(imageHeight, imageWidth, 3)
                I3D[:,:,0] = imageArray
                I3D[:,:,1] = imageArray
                I3D[:,:,2] = imageArray
                imageArray = I3D
            
            imageMatrix[:,:,:,i] = imageArray
        
        return (img_classes, imageMatrix)
            
            
        
        
readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('images', 2, [50,50])
plt.imshow(random_image_matrix[:,:,2,1])
plt.show()

# image_extensions = ('*.jpg', '*.jpeg', '*.png')
# path = 'images'

# cats = []
# dogs = []

# for ext in image_extensions:
#     cats.extend(gl.glob(os.path.join(path, 'cat', ext)))
#     dogs.extend(gl.glob(os.path.join(path, 'dog', ext)))\

# img = Image.open(cats[1])
# img.resize([50,50])
# plt.imshow(img)
# plt.show()