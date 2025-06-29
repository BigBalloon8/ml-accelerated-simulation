# taken from and based on https://github.com/IvanDrokin/torch-conv-kan/blob/main/kan_convs/fast_kan_conv.py
import torch.nn as nn
import torch

from .tools import structureLoader, getAct

class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, # norm_class, # EDITED
                 input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0): # EDITED: removed **norm_kwargs
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation
        self.grid_range = grid_range
        # self.norm_kwargs = norm_kwargs # EDITED

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                 output_dim // groups,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 groups=1,
                                                 bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        # self.layer_norm = nn.ModuleList([norm_class(input_dim // groups, **norm_kwargs) for _ in range(groups)]) # EDITED

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))
        if self.dropout is not None:
            x = self.dropout(x)
        # spline_basis = self.rbf(self.layer_norm[group_index](x)) # EDITED
        spline_basis = self.rbf(x) # EDITED
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](spline_basis)
        x = base_output + spline_output

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class FastKANConvND(nn.Module):
    def __init__(self, config):
        super(FastKANConvND, self).__init__()

        self.structure = structureLoader(config["structures"])
        self.base_activation = getAct(config.get("base_activation", "silu"))

        self.conv_class = config.get("conv_class", nn.Conv2d)
        # self.norm_class = config.get("norm_class", nn.BatchNorm2d) # EDITED
        
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
                                                      # norm_class = self.norm_class, # EDITED
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
            x = layer(x)
        return x