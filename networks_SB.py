import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Uniform noise layer --> DEPRECATED // NOT USED
# se deja para tener info en un futuro
'''
Applies noise to an input between two parameters:
    - low: lowest possible noise
    - high: highest possible noise
'''
class UniformNoiseLayer(nn.Module):
    def __init__(self, low = -0.1, high = 0.1):
        super(UniformNoiseLayer, self).__init__()

        # --- Parameters ---
        self.low = low
        self.high = high
    
    # Forward method
    def forward(self, x):
        # Applies random noise and clips it between both values
        return x + torch.rand_like(x) * (self.high - self.low) + self.low



# Resiudual block
'''
Resiudal block that performs the residual layer operation:
    - in_channels: input channels of the block
    - out_channels: output channels that the block produces
    - kernel_size: sampling of the image: != 1 to reduce image dimensionality
    - kernel_max: max pooling kernel
    - end_layer: flag that indicates wether the block is the last one of feature extractor
    - residual: flag to activate residual connections
    - device: feature extractor device
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, kernel_max = 2, end_layer = False, residual = False, device = "cpu"):
        super(ResidualBlock, self).__init__()

        # --- Parameters ---
        self.end_l = end_layer
        self.residual = residual
        dropout_prob = 0.3

        # Parameters as tensors for GPU conputation
        self.end_l_tensor = torch.tensor(not end_layer, dtype=torch.int8, device = device)
        self.residual_tensor = torch.tensor(residual, dtype=torch.int8, device = device)

        # Regular Layers
        '''
        First convoltion:
            - Convlutional with kernel and padding   --> reshaping
            - Max pooling                            --> Reduction
            - Convlutional for reshaping             --> Reshaping

        Second (residual):
            - ReLU, Convolutional, ReLU, Convolutional: with kernel and padding to mantain dimensionality
            - Convolutional for reshaping
        '''
        if not self.end_l:
            # First convolutional
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels  = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.MaxPool2d(kernel_size = kernel_max), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1),
                                    nn.Dropout(p = dropout_prob))
            
            # Residual convolutional
            self.conv2 = nn.Sequential(nn.ReLU(), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.ReLU(), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1),
                                    nn.Dropout(p = dropout_prob))               


        # Final layer: adds softmax and flatten with reshaping convolutionals 
        else:
            # Convlutional, softmax and faltten layers
            self.conv1 = nn.Sequential(nn.ReLU(),
                                       nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1), 
                                       nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1),
                                       nn.Dropout(p = dropout_prob),
                                       nn.Softmax2d(),
                                       nn.Flatten())
            
            # Empty convlution
            self.conv2 = nn.Sequential()
    
    # Forward method
    def forward(self, x):
        
        # First layer
        x = self.conv1(x)
        
        # Residual connection
        x = x + self.end_l_tensor * self.residual_tensor * self.conv2(x)
        
        return x


# Custom feature extractor class
'''
Feature extractor for the environment
    - Convolutions for the images
    - MLP for the vectors
'''
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, residual = True, channels = [2, 16, 32, 32, 32], kernel = 3, m_kernel = 2, n_layers = 4, h_size=200, w_size=200, out_vector_features = 16, features_dim = 128):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        
        # Parameters
        self.residual = residual                          # Resiudal flag
        self.channels = channels                          # List of input channels
        self.kernel = kernel                              # Kernel size for convolutional layers of image feature extractor
        self.m_kernel = m_kernel                          # Kernel size for max pooling layers
        self.n_layers = n_layers                          # Number of layers
        self.out_vector_features = out_vector_features    # Output size of the vector feature extractor

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # Device selection
        self.to(self.device)


        # --- Environment observation space ---
        q_space = observation_space["ee_position"]  # Robot End - Effector Positions
        image_space = observation_space["image"]    # Image

        # --- Feature extractors ---
        self.image_extractor = nn.Sequential(*(self.build_conv()))      # Images
        self.vector_extractor = nn.Sequential(nn.BatchNorm1d(num_features=7),                          # Position Vector
            nn.Linear(in_features=q_space.shape[0], out_features = self.out_vector_features),
            nn.Tanh())
        

        # Obtains the output dimensions of the flatten convoutioanl extractor layers
        with torch.no_grad():
            n_flatten = self.image_extractor(
                torch.tensor(observation_space.sample()["image"], dtype=torch.float32, device = self.device)
            )


        # Obtains the features dimensions combined
        self.features_dim_ = n_flatten.shape[0] * n_flatten.shape[1] + self.out_vector_features
        
        # MLP for combining features layers' outputs into a fixed dimension vector specified
        self.n_linear = nn.Sequential(nn.Linear(in_features = self.features_dim_, out_features = features_dim), nn.ReLU())
        

    # Forward method
    def forward(self, observations) -> torch.Tensor:

        # Obtain inputs as torch tensors for GPU computation
        image_tensor = torch.as_tensor(observations["image"], device=self.device, dtype=torch.float32)
        q_tensor = torch.as_tensor(observations["ee_position"], device=self.device, dtype=torch.float32)

        # Computes separate feature extractor
        image_features = self.image_extractor(image_tensor)
        vector_features = self.vector_extractor(q_tensor)

        ret = self.n_linear(torch.cat([image_features, vector_features], dim=1))

        # Returns combined features
        return ret
    

    # Build convolutional feature extractor with:
    '''
    - in_channels
    - out_channels: the input of the next layer
    - kernel_size for the convolutional
    - device
    - residual flag
    - m_kernel: kernel size for max pooling
    - endl_l: end layer flag
    '''
    def build_conv(self):

        return [ResidualBlock(in_channels = self.channels[i],
                              out_channels = self.channels[i + 1],
                              kernel_size = self.kernel, device = self.device, residual = self.residual,
                              kernel_max = self.m_kernel, end_layer = (i == self.n_layers-1)).to(self.device) for i in range(self.n_layers)]

        


