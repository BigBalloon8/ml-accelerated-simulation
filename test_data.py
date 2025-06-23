import safetensors.torch as safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F

'''ds =  safetensors.load_file("/Users/ZechengQI/Downloads/data.safetensors")
print(ds.keys())
print(ds["2_c"].shape)'''


if __name__ == "__main__":
    import json
    #with open("src/models/configs/mlp1.json", "r") as f:
        #config = json.load(f)
    print(F.dropout(torch.Tensor(list(range(10))), 1))



# Hyperparameters:
# number of layers 
# size of each layer
# Activation function (ReLU, SELU, GeLU, etc.)
# Regularization (dropout rate)
# Initialization method (Xavier, LeCun, etc.)

# Optimizer (Adam, SGD, etc.)
# Loss function (CrossEntropy, MSE, etc.)

# (for CNN)
# Strides
# Padding
# kernel size
# 

# Training parameters:
# Learning rate
# Batch size
# Training epochs