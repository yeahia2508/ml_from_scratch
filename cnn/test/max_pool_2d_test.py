# -*- coding: utf-8 -*-
from cnn.utils import *
from PIL import Image
from cnn.maxpool_layer_2d import *
import matplotlib.pyplot as plt


readImage = ReadImageClass()
random_classes, random_image_matrix = readImage.get_batch_image_with_class('../images', 2, [100,100])
gImages = readImage.rgb_to_grayscale_batch(random_image_matrix)

mpl2d = MaxPoolLayer2D()
mpl2d.forward(gImages, ks=[3,3], stride= 2)
mpl2d.backward(mpl2d.output)

D = mpl2d.dinputs
mask = mpl2d.mask

Im1 = mpl2d.input[:,:,0]
Im2 = mpl2d.output[:,:,0]
Im3 = mpl2d.mask[:,:,0]
Im4 = mpl2d.dinputs[:,:,0]
plt.imshow(Im1)
plt.show()
plt.imshow(Im2)
plt.show()
plt.imshow(Im3)
plt.show()
plt.imshow(Im4)
plt.show()

    

