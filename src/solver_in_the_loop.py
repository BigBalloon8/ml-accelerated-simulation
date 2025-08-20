import torch
import torch.nn as nn

from torch_cfd import grids, boundaries
from torch_cfd.initial_conditions import filtered_velocity_field

from torch_cfd.equations import stable_time_step
from torch_cfd.fvm import RKStepper, NavierStokes2DFVMProjection
from torch_cfd.forcings import KolmogorovForcing
import torch_cfd.finite_differences as fdm
import torch_cfd.tensor_utils as tensor_utils
import torch.utils._pytree as pytree


from tqdm import tqdm
import safetensors.torch as st
import argparse
import json
import os
from typing import Tuple
import hashlib

from data.dataloader import get_kolomogrov_flow_full_data
from log import Logger
from grid_utils import downsample_staggered_velocity, downsample_tensor, get_grid_data, tensor_to_grid

from models import buildModel

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

def get_model(name:str, config_file, checkpoint_path, logger)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)
    logger.log(f"Model Config: {config}")

    model_base = buildModel(config)
    
    if f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path):
        print(f"Model Found in {checkpoint_path}: {name}_{hash_dict(config)}.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.safetensors")
        model_weights = st.load_file(model_path)
        model_base.load_state_dict(model_weights)
    else:
        raise FileNotFoundError("Model not found")
    return model_base

class SOL(nn.Module):
    def __init__(self, model:nn.Module, device, sol_size=64):
        super().__init__()
        high_res = 1024
        low_res = 64
        density = 1.0
        max_velocity = 7.0
        peak_wavenumber = 4.0
        cfl_safety_factor = 0.5
        viscosity = 1e-3
        diam = 2 * torch.pi
        self.scale_factor = (high_res//low_res)
        self.sol_steps = sol_size//self.scale_factor
        self.input_scale = 7
        self.output_scale = 0.004482923474868822
        
        self.full_step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)
        self.coarse_step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)

        self.full_grid = grids.Grid((high_res, high_res), domain=((0, diam), (0, diam)), device=device)
        self.coarse_grid = grids.Grid((low_res, low_res), domain=((0, diam), (0, diam)), device=device)
        
        self.full_dt = stable_time_step(
            dx=min(self.full_grid.step),
            max_velocity=max_velocity,
            max_courant_number=cfl_safety_factor,
            viscosity=viscosity,
        )

        self.coarse_dt = self.full_dt*self.scale_factor

        self.v0_full = filtered_velocity_field(
                self.full_grid, max_velocity, peak_wavenumber, iterations=16, random_state=42,
                device=device, batch_size=32)

        self.v0_coarse = downsample_staggered_velocity(self.full_grid, self.coarse_grid, self.v0_full)
        self.v0_LC_coarse = downsample_staggered_velocity(self.full_grid, self.coarse_grid, self.v0_full)

        forcing_fn_full = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
            grid=self.full_grid, offsets=(self.v0_full[0].offset, self.v0_full[1].offset))

        forcing_fn_coarse = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
            grid=self.coarse_grid, offsets=(self.v0_coarse[0].offset, self.v0_coarse[1].offset))
        
        self.ns2d_full = NavierStokes2DFVMProjection(
            viscosity=viscosity,
            grid=self.full_grid,
            bcs=(self.v0_full[0].bc, self.v0_full[1].bc),
            density=density,
            drag=0.1,
            forcing=forcing_fn_full,
            solver=self.full_step_fn,
            # set_laplacian=False,
        ).to(self.v0_full.device)

        self.ns2d_coarse = NavierStokes2DFVMProjection(
            viscosity=viscosity,
            grid=self.coarse_grid,
            bcs=(self.v0_coarse[0].bc, self.v0_coarse[1].bc),
            density=density,
            drag=0.1,
            forcing=forcing_fn_coarse,
            solver=self.coarse_step_fn,
            # set_laplacian=False,
        ).to(self.v0_coarse.device)
    
        self.model = model
    
    def forward(self, x):
        v_full = tensor_to_grid(x, self.full_grid, self.v0_full)
        v_coarse = downsample_staggered_velocity(self.full_grid, self.coarse_grid, v_full)
        for _ in range(self.sol_steps):
            with torch.no_grad():
                for _ in range(self.scale_factor):
                    v_full, _ = self.full_step_fn.forward(v_full, self.full_dt, self.ns2d_full)

            v_coarse, _ = self.coarse_step_fn.forward(v_coarse, self.coarse_dt, equation=self.ns2d_coarse)
            coarse = get_grid_data(v_coarse).transpose(0,1)
            coarse_norm = coarse / self.input_scale
            delta_v = self.model.forward(coarse_norm)
            coarse += delta_v*self.output_scale
            v_coarse = tensor_to_grid(coarse.transpose(0,1), self.coarse_grid, v_coarse)
        
        with torch.no_grad():
            coarsened_full = torch.vmap(downsample_tensor, in_dims=1, out_dims=1)(get_grid_data(v_full)).squeeze_()

        
def main(data_path, model_type, model_config, checkpoint_path, log_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    logger = Logger(model_type, log_file)

    model = get_model(model_type, model_config, checkpoint_path, logger)

    train_dataloader, val_dataloader = get_kolomogrov_flow_full_data(data_path, batchsize=32)






