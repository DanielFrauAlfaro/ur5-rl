import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Resiudual block
'''
Resiudal block that performs the residual layer operation:
    Parameters:
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
            self.conv1 = nn.Sequential(nn.InstanceNorm2d(in_channels),
                                    nn.Conv2d(in_channels  = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.MaxPool2d(kernel_size = kernel_max), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1))
            
            torch.nn.init.xavier_uniform_(self.conv1[1].weight)
            torch.nn.init.xavier_uniform_(self.conv1[3].weight)
            
            # Residual convolutional
            self.conv2 = nn.Sequential(nn.ReLU(), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.ReLU(), 
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = kernel_size // 2),
                                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1))               
            torch.nn.init.xavier_uniform_(self.conv2[1].weight)
            torch.nn.init.xavier_uniform_(self.conv2[3].weight)

        # Final layer: adds softmax and flatten with reshaping convolutionals 
        else:
            # Convlutional, softmax and faltten layers
            self.conv1 = nn.Sequential(nn.ReLU(),
                                       nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1), 
                                       nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1),
                                       nn.Softmax2d(),
                                       nn.Flatten(),
                                       nn.Dropout(p = dropout_prob))
            torch.nn.init.xavier_uniform_(self.conv1[1].weight)
            torch.nn.init.xavier_uniform_(self.conv1[2].weight)
            
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
Feature extractor for the environment:
    - Convolutions for the images
    - MLP for the vectors

    Parameters:
        - observation_space (dict): dictionary of observations that represents the observations of the environment
        - residual (bool): activate the residual connections
        - channels (list): evolution of the channels throughout the network
        - kernel (int): kernel for convolutional layers
        - m_kernel (int): kernel for pooling layers
        - n_layers (int): number of ResidualBlocks used
        - out_vector_features (int): desired size of output vector features
        - features_dim (int): desired size of output image features
'''
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, residual = True, channels = [2, 16, 32, 32, 32], kernel = 3, m_kernel = 2, n_layers = 4, out_vector_features = 16, features_dim = 128):
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
        self.image_extractor_1 = nn.Sequential(*(self.build_conv()))      # Images
        self.image_extractor_2 = nn.Sequential(*(self.build_conv()))
        self.image_extractor_3 = nn.Sequential(*(self.build_conv()))

        self.vector_extractor = nn.Sequential(nn.BatchNorm1d(num_features=6),                          # Position Vector
            nn.Linear(in_features=q_space.shape[0], out_features = self.out_vector_features),
            nn.Tanh())
        torch.nn.init.xavier_uniform_(self.vector_extractor[1].weight)
        
        # Obtains the output dimensions of the flatten convoutioanl extractor layers
        with torch.no_grad():
            n_flatten = self.image_extractor_1(
                torch.tensor(observation_space.sample()["image"][:2], dtype=torch.float32, device = self.device)
            )

        # Obtains the features dimensions combined
        self.features_dim_ = n_flatten.shape[0] * n_flatten.shape[1] * 3 + self.out_vector_features
        
        # MLP for combining features layers' outputs into a fixed dimension vector specified
        self.n_linear = nn.Sequential(nn.Linear(in_features = self.features_dim_, out_features = features_dim), nn.Tanh())
        torch.nn.init.xavier_uniform_(self.n_linear[0].weight)

    # Forward method
    def forward(self, observations) -> torch.Tensor:

        # Obtain inputs as torch tensors for GPU computation
        image_tensor_1 = torch.as_tensor(observations["image"][:, :2], device=self.device, dtype=torch.float32)
        image_tensor_2 = torch.as_tensor(observations["image"][:, 2:4], device=self.device, dtype=torch.float32)
        image_tensor_3 = torch.as_tensor(observations["image"][:, 4:], device=self.device, dtype=torch.float32)
        q_tensor = torch.as_tensor(observations["ee_position"], device=self.device, dtype=torch.float32)

        # Computes separate feature extractor
        image_features_1 = self.image_extractor_1(image_tensor_1)
        image_features_2 = self.image_extractor_2(image_tensor_2)
        image_features_3 = self.image_extractor_3(image_tensor_3)

        image_features = torch.cat((image_features_1, image_features_2, image_features_3), dim=1)

        vector_features = self.vector_extractor(q_tensor)

        ret = self.n_linear(torch.cat([image_features, vector_features], dim=1))

        # Returns combined features
        return ret
    

    # Build convolutional feature extractor with:
    '''
    - in_channels: the input of the actual layer // output of the previous one
    - out_channels: the output of the actual layer // the input of the next layer
    - kernel_size: size of the convolutional kernel
    - device
    - residual: residual flag
    - m_kernel: kernel size for max pooling
    - endl_l: end layer flag
    '''
    def build_conv(self):

        return [ResidualBlock(in_channels = self.channels[i],
                              out_channels = self.channels[i + 1],
                              kernel_size = self.kernel, device = self.device, residual = self.residual,
                              kernel_max = self.m_kernel, end_layer = (i == self.n_layers-1)).to(self.device) for i in range(self.n_layers)]

        


