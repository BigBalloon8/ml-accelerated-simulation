import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList, structureLoader, getAct, getModel, getLayers

class ResNetBlock(nn.Module):
    """
    Residual neural network (ResNet) block with customizable hyperparameters.
    Args:
        config^ (dict): A dictionary containing hyperparameters:\n 
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
            1x1_conv: (bool): Whether to apply 1x1 convolution to input before combining with output\n
    (^):\n Use resNetBasicBlock.json for basic block.\n
    \t Use resNetBottleneckBlock.json for Bottleneck Block.\n
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])        
        structure = structureLoader(config["structures"])
        kernel_sizes, strides, paddings, group = paramToList(config["kernel_sizes"], len(structure)-1), paramToList(config["strides"], len(structure)-1), paramToList(config["paddings"], len(structure)-1), paramToList(config["group"], len(structure)-1)
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(structure[i], structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], groups=group[i]), nn.BatchNorm2d(structure[i+1])) for i in range(len(structure)-1)])
        try:
            if config["1x1_conv"]:
                self.conv1 = nn.Conv2d(structure[0], structure[-1], kernel_size=1)
            else: 
                self.conv1 = None
        except(KeyError):
            self.conv1 = None

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                y = F.dropout(self.act(layer(y)), p=self.dropouts[i], training=True)
        y = layer(y)
        if self.conv1 is not None:
                x = self.conv1(x)
        return self.act(x+y)



class ResNeXtBlock(nn.Module):
    """
    Residual neural network with Aggregated Transformation (ResNet) block with customizable hyperparameters.
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
            1x1_conv: (bool): Whether to apply 1x1 convolution to input before combining with output\n
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"]) 
        structure = structureLoader(config["structures"])
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.layers = nn.ModuleList(getLayers(getModel(config, "ResNetBlock"))[0])
        if config["1x1_conv"]:
            self.conv1 = nn.Sequential(nn.Conv2d(structure[0], structure[-1], kernel_size=1), nn.BatchNorm2d(structure[-1]))
        else: 
             self.conv1 = None

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                y = F.dropout(self.act(layer(y)), p=self.dropouts[i], training=True)
        y = layer(y)
        if self.conv1 is not None:
                x = self.conv1(x)
        return self.act(x+y)




if __name__ == "__main__":
    import json
    with open("src/models/configs/resNeXtBlock1.json", "r") as f:
        config = json.load(f)
        resnet = ResNeXtBlock(config[0])
        print(resnet)

