import numpy as np

class BatchNormLayer:
    def __init__(self, channels: int, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Parameters for each channel
        self.gamma = np.ones((1, 1, channels, 1), dtype="float32")
        self.beta = np.zeros((1, 1, channels, 1), dtype="float32")
        
        # Running (moving) mean and variance
        self.running_mean = np.zeros((1, 1, channels, 1), dtype="float32")
        self.running_var = np.ones((1, 1, channels, 1), dtype="float32")

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        height, width, channels, batch_size = x.shape
        
        # Compute mean and variance across the batch dimension
        if train:
            batch_mean = np.mean(x, axis=(3), keepdims=True)  # Mean across batch_size
            batch_var = np.var(x, axis=(3), keepdims=True)    # Variance across batch_size
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running statistics during inference
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalize the input (batch normalization)
        x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        
        # Scale and shift using gamma (scale) and beta (shift)
        out = self.gamma * x_normalized + self.beta
        
        return out