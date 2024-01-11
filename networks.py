import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_max):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               bias=False, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv1_ = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                bias=False, stride=1, padding=0)                    

        # Use the same kernel size for adjust_dimensions
        self.adjust_dimensions = nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1, padding=0)

        self.max_p = nn.MaxPool2d(kernel_size=kernel_max)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = x + self.conv1_(x)

        '''
        Si se quiere hacer residual pura el kernel de conv1_ != 1:
        residual = x
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        # Adjust dimensions of the residual if needed
        residual = self.adjust_dimensions(residual)

        x += residual  # Add residual connection
        '''

        
        return self.max_p(x)

class Network(nn.Module):
    def __init__(self, lr, in_channels, out_channels, kernel, m_kernel, n_layers, recurrent,
                 dir):
        super(Network, self).__init__()

        # Parameters
        self.lr = lr
        self.checkpoint_dir = dir
        self.input = in_channels
        self.out = out_channels
        self.kernel = kernel
        self.m_kernel = m_kernel
        self.n_layers = n_layers
        self.recurrent = recurrent

        # -- Building model
        # Convlutional layers
        self.layers = self.build_model()
        
        self.device = "cpu"# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.layers(x)

        return x

    def build_residual(self):
        rec_layers = [ResidualBlock(in_channels = self.out[i],
                                        out_channels = self.out[i + 1],
                                        kernel_size = self.kernel[i + 1],
                                        kernel_max = self.m_kernel) for i in range(self.n_layers - 1)]
        

        rec_layers.insert(0, ResidualBlock(in_channels = self.input,
                                        out_channels = self.out[0],
                                        kernel_size = self.kernel[0],
                                        kernel_max = self.m_kernel))
        
        rec_layers.insert(0, nn.MaxPool2d(kernel_size=self.m_kernel))

        return rec_layers
    
    def build_non_residual(self):
        rec_layers = [nn.Conv2d(in_channels = self.out[i],
                                    out_channels = self.out[i + 1],
                                    kernel_size = self.kernel[i + 1],
                                    bias = False) for i in range(self.n_layers - 1)]
        
        rec_layers.insert(0, nn.Conv2d(in_channels = self.input,
                                    out_channels = self.out[0],
                                    kernel_size = self.kernel[0],
                                    bias = False))
        rec_layers.insert(0, nn.MaxPool2d(kernel_size=self.m_kernel))

        return rec_layers

    def build_conv(self):
        if self.recurrent:
            return self.build_residual()

        else:
            return self.build_non_residual()
    
    def build_mlp(self):
        return 

    def build_model(self):
        conv_layers = self.build_conv()
        mlp_layers = self.build_mlp()

        return  nn.Sequential(*conv_layers)


    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_dir))        

if __name__ == "__main__":
    test = Network(lr = 0.001, in_channels=1, out_channels=[16,32], 
                   kernel=[2, 2], m_kernel = 3, n_layers=2, 
                   recurrent=True, dir="")

    image = np.random.randint(low = 0, high = 255, size = (96, 128), dtype=np.int16)
    input_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(test.device)
    
    out = test(input_image)