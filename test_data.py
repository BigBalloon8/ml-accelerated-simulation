import safetensors.torch as safetensors
import torch


'''ds =  safetensors.load_file("/Users/ZechengQI/Downloads/data.safetensors")
print(ds.keys())
print(ds["2_c"].shape)'''
x, y, z = 2, [3, 4], 5



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