import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, kernel_max = 2, end_layer = False):
        super(ResidualBlock, self).__init__()

        self.end_l = end_layer
        
        if not self.end_l:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels  = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.MaxPool2d(kernel_size = kernel_max), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1))
            
            self.conv2 = nn.Sequential(nn.ReLU(), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.ReLU(), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1))               

        else:
            self.conv1 = nn.Sequential(nn.ReLU(),
                                       nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1), 
                                       nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1),
                                       nn.Softmax2d(),
                                       nn.Flatten())
            self.conv2 = nn.Sequential()

    def forward(self, x):
        x = self.conv1(x)
        x = x + int(1 - self.end_l)*self.conv2(x)

        return x

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, residual = True, channels = [2, 16, 32, 32, 32], kernel = 3, m_kernel = 2, n_layers = 4, h_size=200, w_size=200, out_q_features = 16, features_dim = 128):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        
        # Parameters
        self.residual = residual
        self.channels = channels
        self.kernel = kernel
        self.m_kernel = m_kernel
        self.n_layers = n_layers
        self.out_q_features = out_q_features


        q_space = observation_space["q_position"]
        image_space = observation_space["image"]

        self.image_extractor = nn.Sequential(*(self.build_conv()))
        self.vector_extractor = nn.Sequential(
            nn.Linear(in_features=q_space.shape[0], out_features = self.out_q_features),
            nn.ReLU())
        

        with torch.no_grad():
            n_flatten = self.image_extractor(
                torch.as_tensor(observation_space.sample()["image"]).float()
            )

        
        self.features_dim_ = n_flatten.shape[0] * n_flatten.shape[1] + self.out_q_features
        
        self.n_linear = nn.Sequential(nn.Linear(in_features = self.features_dim_, out_features = features_dim), nn.ReLU())
        

        self.device = "cpu"# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, observations) -> torch.Tensor:

        image_tensor = torch.as_tensor(observations["image"], device=self.device, dtype=torch.float32)
        q_tensor = torch.as_tensor(observations["q_position"], device=self.device, dtype=torch.float32)


        image_features = self.image_extractor(image_tensor)
        vector_features = self.vector_extractor(q_tensor)
        return self.n_linear(torch.cat([image_features, vector_features], dim=1))
    


    def build_conv(self):
        if self.residual:
            return [ResidualBlock(in_channels = self.channels[i],
                                  out_channels = self.channels[i + 1],
                                  kernel_size = self.kernel,
                                  kernel_max = self.m_kernel, end_layer = (i == self.n_layers-1)) for i in range(self.n_layers)]

        else:
            return



