import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv

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

        '''
        Si se quiere hacer residual pura el kernel de conv1_ != 1:
        residual = x
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        # Adjust dimensions of the residual if needed
        residual = self.adjust_dimensions(residual)

        x += residual  # Add residual connection
        '''

        
        return x

class Network(nn.Module):
    def __init__(self, lr, channels, kernel, m_kernel, n_layers, residual,
                 dir, w_size, h_size, act_in = 6, act_out = 10):
        super(Network, self).__init__()

        # Parameters
        self.lr = lr
        self.checkpoint_dir = dir
        self.channels = channels
        self.kernel = kernel
        self.m_kernel = m_kernel
        self.n_layers = n_layers
        self.residual = residual

        self.act_in = act_in
        self.act_out = act_out
        self.mlp_encoding_in = int(h_size * w_size * channels[-1] / (m_kernel * n_layers)**2)
        

        # -- Building model
        # Convlutional layers
        self.layers = self.build_model()
        
        self.device = "cpu"# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.layers(x)

        return x

    def build_conv(self):
        if self.residual:
            return [ResidualBlock(in_channels = self.channels[i],
                                  out_channels = self.channels[i + 1],
                                  kernel_size = self.kernel,
                                  kernel_max = self.m_kernel, end_layer = (i == self.n_layers-1)) for i in range(self.n_layers)]

        else:
            return
    
    def build_mlp(self):
        return nn.Sequential(nn.Linear(in_features = self.act_in, out_features = self.act_out),
                             nn.LayerNorm(),
                             nn.Tanh())

    def build_model(self):
        conv_layers = self.build_conv()
        mlp_layers = self.build_mlp()

        return  nn.Sequential(*conv_layers)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_dir))        

if __name__ == "__main__":
    X = torch.randn(1, 1, 120, 104)
    test = Network(lr = 0.001, channels=[1, 16,32, 32, 32], 
                   kernel=3, m_kernel = 2, n_layers=4, 
                   residual=True, dir="", w_size = X.shape[-2], h_size = X.shape[-1])
    

    out = test(X)
    print(out.shape)
