import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from models.tools import paramToList, structureLoader, getAct

class DenseBlock(nn.Module):
    """
    Dense neural network (DenseNet) block with customizable hyperparameters.
    Args:
        config (dict): A dictionary containing hyperparameters:\n 
            structure (dict): Structure of Model: (\n
                in_channels (int): Size of input channels,\n
                hidden_channels (list): Size of hidden channels,\n
                out_channels (int): Size of output channels)\n
            kernel_sizes* (int or list): Dimension of kernel\n
            strides* (int or list): Step size that the kernel will take\n
            paddings* (int or list): Width of padding\n
            group* (int or list): number of groups (must divide both in_channels and out_channels) (Set to 1 for default)\n
            dropouts* (int, float or list): Dropout probability for each layer (except the last) (Set to 0 for no dropout)\n
            activation_func (str): Name of desired activation function\n 
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])        
        structure = structureLoader(config["structures"])
        kernel_sizes, strides, paddings, group = paramToList(config["kernel_sizes"], len(structure)-1), paramToList(config["strides"], len(structure)-1), paramToList(config["paddings"], len(structure)-1), paramToList(config["group"], len(structure)-1)
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.layers = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(sum(structure[:i+1])), nn.Conv2d(sum(structure[:i+1]), structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], groups=group[i])]) for i in range(len(structure)-1)])
        self.conv1 = nn.ModuleList([nn.BatchNorm2d(sum(structure)), nn.Conv2d(sum(structure), structure[-1], kernel_size=1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            y = F.dropout(layer[1](self.act(layer[0](x))), p=self.dropouts[i], training=True)
            x = cat((x,y), dim=1) # concaternate along channels
        return self.conv1[1](self.act(self.conv1[0](x))) # reduce channel size down to the desired output size

if __name__ == "__main__":
    import json
    with open("src/models/configs/denseNetBlock1.json", "r") as f:
        config = json.load(f)
        densenet = DenseBlock(config)
        print(densenet)
    
