import numpy as np

class ConvLayer:
    def __init__(self, kShape, kNumber):
        self.xKShape = kShape[0]
        self.yKShape = kShape[1]
        self.kNumber = kNumber

        self.weights = np.random.rand(self.xKShape, self.yKShape, kNumber)
        self.biases = np.random.rand(1, kNumber)

        L2 = 1e-2
        self.weights_L2 = L2
        self.biases_L2  = L2

    def forward(self, imageMatrix, padding = 0, stride = 1):
        [xImageShape, yImageShape, channelSize, batchSize] = imageMatrix.shape

        xOutput = int((xImageShape - self.xKShape + 2 * padding ) / stride + 1)
        yOutput = int((yImageShape - self.yKShape + 2 * padding ) / stride + 1)

        output = np.zeros((xOutput, yOutput, channelSize, self.kNumber, batchSize))

        imagePadded = np.zeros((xImageShape + 2 * padding, yImageShape + 2*padding, channelSize, self.kNumber, batchSize))

        for k in range(self.kNumber):
            imagePadded[int(padding): int(padding + xImageShape), int(padding): int(padding + yImageShape), :,k,:] = imageMatrix


        if channelSize == 6 & self.kNumber == 16:
            filt = np.array([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
                             [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
                             [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
                             [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
                             [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
                             [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]])

        else:
            filt = np.ones([channelSize,self.kNumber])

        self.filt = filt

        for i in range(batchSize):
            for k in range(self.kNumber):
                for c in range(channelSize):
                    if filt[c, k] == 1:
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
        self.padding = padding
        self.stride = stride


    def backward(self, dvalues):
        stride = self.stride
        padding = self.padding
        xk = self.xKShape
        yk = self.yKShape
        nk = self.kNumber
        weights = self.weights
        paddedImage = self.paddedInput

        paddedImageShape = paddedImage.shape
        dinputs = np.zeros((paddedImageShape[0], paddedImageShape[1], paddedImageShape[2], paddedImageShape[4]))

        dbiases = np.zeros(self.biases.shape)
        dweights = np.zeros(self.weights.shape)

        xd = dvalues.shape[0]
        yd = dvalues.shape[1]
        numChan = paddedImageShape[2]
        batchSize = paddedImageShape[4]

        imagePadded = paddedImage[:,:,:,0,:]
        filt = self.filt

        for i in range(batchSize):
            for c in range(numChan):
                for k in range(nk):
                    if filt[c, k] == 1:
                      for y in range(yd):
                          for x in range(xd):
                              yStart = y * stride
                              yEnd = yStart + yk
                              xStart = x * stride
                              xEnd = xStart + xk

                              sx = slice(xStart, xEnd)
                              sy = slice(yStart, yEnd)

                              currentSlice = imagePadded[sx, sy, c, i]
                              dweights[:,:,k] += currentSlice * dvalues[x,y,k,i]
                              dinputs[sx, sy, c, i] += dweights[:,:,k] * dvalues[x,y,k,i]

                    dbiases[0,k] = np.sum(np.sum(weights[:,:,k], axis = 0), axis = 0)

        dinputs = dinputs[padding: paddedImageShape[0]-padding, padding: paddedImageShape[1]-padding, :, :]
        self.dinputs = dinputs

        self.dbiases  = dbiases  + 2 * self.biases_L2 * self.biases
        self.dweights = dweights + 2 * self.weights_L2 * self.weights