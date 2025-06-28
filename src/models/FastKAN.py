import torch.nn as nn
from KANs import FastKANLayer
from .tools import structureLoader, getAct    

class FastKAN(nn.Module):
    def __init__(self, config):
        super(FastKAN, self).__init__()
        
        self.structure = structureLoader(config["structures"])
        self.base_activation = getAct(config.get("base_activation", "silu"))

        self.grid_min, self.grid_max = config["grid_range"]
        self.spline_weight_init_scale = config["spline_weight_init_scale"]
        self.use_base_update = config["use_base_update"]
        self.num_grids = config["num_grids"]

        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
        self.layers = nn.ModuleList([FastKANLayer(input_dim = self.structure[i],
                                                  output_dim = self.structure[i+1],
                                                  grid_min = self.grid_min,
                                                  grid_max = self.grid_max,
                                                  num_grids = self.num_grids,
                                                  use_base_update = self.use_base_update,
                                                  base_activation= self.base_activation,
                                                  spline_weight_init_scale = self.spline_weight_init_scale) for i in range(len(self.structure)-1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return x
