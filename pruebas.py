import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define a simple residual block with a 3x3 kernel
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()

        # Adjust dimensions if needed
        self.adjust_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        residual = x
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        print(residual.shape)

        # Adjust dimensions of the residual if needed
        residual = self.adjust_dimensions(residual)
        print(residual.shape)
        print(x.shape)
        print(self.adjust_dimensions)
        print("---\n\n")
        x += residual  # Add residual connection
        return x

# Define a simple convolutional network with a residual layer
class ConvNetWithResidual(nn.Module):
    def __init__(self):
        super(ConvNetWithResidual, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Residual block with a 3x3 kernel
        self.residual_block = ResidualBlock(16, 32, kernel_size=3)

        # Final convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.residual_block(x)
        x = self.relu2(self.conv2(x))
        return x

# Create an instance of the model
# model = ConvNetWithResidual()

# # Create a random 200x200 numpy array with int8 values
# numpy_array = np.random.randint(low=0, high=255, size=(1, 1, 200, 200), dtype=np.int16)

# # Convert the numpy array to a PyTorch tensor and cast it to float32
# input_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# # Perform the forward pass
# output_tensor = model(input_tensor)

# # Print the output shape
# print("Output shape:", output_tensor.shape)


print(1//2)
pos, orn = np.random.uniform([[0.3, 2], [0.2, 2]], [[1, 4],[40,4]])
print(pos)
print(orn)

collisions_to_check = [[1, 2],
                       [3, (1, "robotiq_finger_1_link_3")], 
                       [3, (1, "robotiq_finger_2_link_3")], 
                       [3, (1, "robotiq_finger_middle_link_3")]]

print(collisions_to_check[1:])

a = np.array([1,2,4])
b = np.array([43,5,6])
d = a

c = np.concatenate((a,b,d))
print(c)