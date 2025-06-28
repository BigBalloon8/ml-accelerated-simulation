import torch.nn as nn
from KANs import FastKANConvNDLayer
from .tools import structureLoader, getAct

class FastKANConvND(nn.Module):
    def __init__(self, config):
        super(FastKANConvND, self).__init__()

        self.structure = structureLoader(config["structures"])
        self.base_activation = getAct(config.get("base_activation", "silu"))

        self.conv_class = config.get("conv_class", nn.Conv2d)
        self.norm_class = config.get("norm_class", nn.BatchNorm2d)
        
        self.kernel_size = config.get("kernel_size", 3)
        self.groups = config.get("groups", 1)
        self.padding = config.get("padding", (self.kernel_size-1)//2)
        self.stride = config.get("stride", 1)
        self.dilation = config.get("dilation", 1)
        self.ndim = config.get("ndim", 2)
        self.grid_size = config.get("grid_size", 8)
        self.grid_range = config.get("grid_range", [-2,2])
        self.dropout = config.get("dropout", 0.0)

        self.layers = nn.ModuleList([FastKANConvNDLayer(conv_class = self.conv_class,
                                                        norm_class = self.norm_class,
                                                        input_dim = self.structure[i],
                                                        output_dim = self.structure[i+1],
                                                        kernel_size = self.kernel_size,
                                                        groups = self.groups,
                                                        padding = self.padding,
                                                        stride = self.stride,
                                                        dilation = self.dilation,
                                                        ndim = self.ndim, 
                                                        grid_size = self.grid_size,
                                                        base_activation = self.base_activation, 
                                                        grid_range = self.grid_range,
                                                        dropout = self.dropout) for i in range(len(self.structure)-1)])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
        return x