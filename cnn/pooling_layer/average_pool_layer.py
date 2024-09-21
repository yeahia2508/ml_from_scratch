# -*- coding: utf-8 -*-

import numpy as np

class Average_Pool:
    
    def forward(self, M, stride = 1, KernShape = 2):
        
        xImgShape  = M.shape[0]
        yImgShape  = M.shape[1]
        numChans   = M.shape[2]
        numImds    = M.shape[3]
        
        self.inputs = M
        
        xK = KernShape
        yK = KernShape
        
        xOutput = int(((xImgShape - xK) / stride) + 1)
        yOutput = int(((yImgShape - yK) / stride) + 1)
        
        imagePadded = M
        
        output = np.zeros((xOutput,yOutput,numChans,numImds))
        
        for i in range(numImds):# loop over number of images
            currentIm_pad = imagePadded[:,:,:,i] #select ith padded image
            for y in range(yOutput):# loop over vert axis of output
                for x in range(xOutput):# loop over hor axis of output
                    for c in range(numChans):# loop over channels (= #filters)
                    
                    # finding corners of the current "slice" 
                        y_start = y*stride
                        y_end   = y*stride + yK
                        x_start = x*stride 
                        x_end   = x*stride + xK
                        
                        sx      = slice(x_start,x_end)
                        sy      = slice(y_start,y_end)
                    
                        current_slice = currentIm_pad[sx,sy,c]
                        
                        #actual average pool
                        slice_mean         = float(current_slice.mean())
                        output[x, y, c, i] = slice_mean
                        
        
        #storing info, also for backpropagation
        self.xKernShape = xK
        self.yKernShape = yK
        self.output     = output
        self.impad      = imagePadded
        self.stride     = stride
    
    def backward(self, dvalues):
        
        xd = dvalues.shape[0]
        yd = dvalues.shape[1]
        
        numChans = dvalues.shape[2]
        numImds  = dvalues.shape[3]
        
        imagePadded = self.impad
        dinputs     = np.zeros(imagePadded.shape)
        Ones        = np.ones(imagePadded.shape)#for backprop
        
        stride  = self.stride
        xK      = self.xKernShape
        yK      = self.yKernShape
        
        Ones    = Ones/xK/yK # normalization that came from average pool
        
        for i in range(numImds):# loop over number of images
            for y in range(yd):# loop over vert axis of output
                for x in range(xd):# loop over hor axis of output
                    for c in range(numChans):# loop over channels (= #filters)
                    
                        # finding corners of the current "slice" 
                        y_start = y*stride
                        y_end   = y*stride + yK
                        x_start = x*stride 
                        x_end   = x*stride + xK
                            
                        sx      = slice(x_start,x_end)
                        sy      = slice(y_start,y_end)
                            
                        dinputs[sx,sy,c,i]  += Ones[sx,sy,c,i]*dvalues[x,y,c,i]
                            
        
        self.dinputs = dinputs

