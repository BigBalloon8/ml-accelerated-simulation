import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList


class CNN(nn.Module):
    """
    Convolution layers with customisable hyperparameters.
    Args:
        config (dict): A dictionary containing hyperparameters
            structure (dict): Structure of Model
                in_channels (int): Size of input channels
                hidden_channels (list): Size of hidden channels
                out_channels (int): Size of output channels
            kernel_sizes (int or list): Dimension of kernel
            strides (int or list): Step size that the kernel will take
            paddings (int or list): Width of padding
            dropouts (int, float or list): Dropout probability for each layer (except the last) 
                If a float or int, applies the same dropout to all layers.
                If a list, must match the number of layers minus one.
    """
    def __init__(self, config): 
        super().__init__()
        structure = paramToList(config["structures"], "structures")
        kernel_sizes, strides, paddings = paramToList(config["kernel_sizes"], "kernel_sizes", len(structure)-1), paramToList(config["strides"], "strides", len(structure)-1), paramToList(config["paddings"], "paddings", len(structure)-1)
        self.dropouts = paramToList(config["dropouts"], "dropouts", len(structure)-1)
        self.convs = nn.ModuleList([nn.Conv2d(structure[i], structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(len(structure)-1)])        
            
    def forward(self, x):
        for i, layer in enumerate(self.convs):
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu_(layer(x)), p=self.dropouts[i], training=True)
            else:
                break
        return layer(x)


if __name__ == "__main__":
    import json
    with open("src/models/configs/cnn1.json", "r") as f:
        config = json.load(f)
        cnn = CNN(config)
        print(cnn)