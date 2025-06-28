# taken from and based on https://github.com/IvanDrokin/torch-conv-kan/blob/main/kans/kan.py
# and https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/KALnet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
from .tools import structureLoader, getAct    

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

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
        self.grid = torch.nn.Parameter(grid, requires_grad=True)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class FastKAN(nn.Module):
    def __init__(self, config):
        super(FastKAN, self).__init__()

        self.structure = structureLoader(config["structures"])
        self.dim = config.get("dimension", [1, 1])
        self.base_activation = getAct(config.get("base_activation", "silu"))

        self.grid_min, self.grid_max = config["grid_range"]
        self.spline_weight_init_scale = config["spline_weight_init_scale"]
        self.use_base_update = config["use_base_update"]
        self.num_grids = config["num_grids"]

        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
        self.layers = nn.ModuleList([FastKANLayer(input_dim = self.structure[i]*self.dim[0]*self.dim[1],
                                                  output_dim = self.structure[i+1]*self.dim[0]*self.dim[1],
                                                  grid_min = self.grid_min,
                                                  grid_max = self.grid_max,
                                                  num_grids = self.num_grids,
                                                  use_base_update = self.use_base_update,
                                                  base_activation= self.base_activation,
                                                  spline_weight_init_scale = self.spline_weight_init_scale) for i in range(len(self.structure)-1)])

    def forward(self, x):
        input_shape = x.shape
        x = x.flatten(1, -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != len(self.layers):
                x = self.dropout(x)
        x = x.reshape(input_shape)
        return x
