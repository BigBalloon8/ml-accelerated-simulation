import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList, structureLoader, getAct, getModel, getLayers

class ResNetBlock(nn.Module):
    """
    Residual neural network (ResNet) block with customizable hyperparameters.
    Args:
        config (dict): A dictionary containing hyperparameters 
    Return:


    Use resNetBasicBlock.json for basic block.\n
    Use resNetBottleneckBlock.json for Bottleneck Block.\n
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])        
        structure = structureLoader(config["structures"])
        kernel_sizes, strides, paddings, group = paramToList(config["kernel_sizes"], len(structure)-1), paramToList(config["strides"], len(structure)-1), paramToList(config["paddings"], len(structure)-1), paramToList(config["group"], len(structure)-1)
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(structure[i], structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], groups=group[i]), nn.BatchNorm2d(structure[i+1])) for i in range(len(structure)-1)])
        if config["1x1_Conv"]:
             self.conv1 = nn.Conv2d(structure[0], structure[-1], kernel_size=1)
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



class ResNeXtBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        structure = structureLoader(config["structures"])
        res_layers = getLayers(getModel("ResNetBlock", config))

        self.layers = res_layers["layers"]
        if config["1x1_Conv"]:
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
    with open("src/models/configs/resNetBasicBlock1.json", "r") as f:
        config = json.load(f)
        resnet = ResNeXtBlock(config)
        print(resnet)
