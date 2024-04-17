import numpy as np
import torch


a = [1 ,2, 3, 4, 5]
a_ = torch.tensor(a).unsqueeze(dim=0)

print(a_)

a_ = a_.squeeze(dim=0)
print(a_.numpy())
