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

from models import MLP, CNN, Transformer
from src.models import FastKAN

def hash_dict(x:dict):
    return str(hash(json.dumps(x, sort_keys=True)))

def get_model(name:str, config_file, checkpoint_path)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)

    if name.upper() == "MLP":
        model_base = MLP(config)
    elif name.upper() == "CNN":
        model_base = CNN(config)
    elif name.upper() == "KAN":
        model_base = FastKAN(config)
    elif name.upper() == "TRANSFORMER":
        model_base = Transformer(config)
    else:
        raise ValueError(f"Model type [{name}] not supported please select from |MLP|CNN|KAN|TRANSFORMER|")
    
    if f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.safetensors")
        model_weights = st.load_file(model_path)
        metadata = model_weights.pop("__metadata__")
        model_base.load_state_dict(model_weights)
    else:
        raise FileNotFoundError("Model weights for the given config are not in the checkpoint path")
    return model_base, metadata


def main(model_type, model_config, checkpoint_path):
    #--------------Simulation Setup-----------------
    density = 1.0
    max_velocity = 7.0
    peak_wavenumber = 4.0
    cfl_safety_factor = 0.5
    viscosity = 1e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    diam = 2 * torch.pi
    simulation_time = 30

    step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)

    coarse_grid = grids.Grid((64, 64), domain=((0, diam), (0, diam)), device=device)

    dt = stable_time_step(
        dx=min(coarse_grid.step),
        max_velocity=max_velocity,
        max_courant_number=cfl_safety_factor,
        viscosity=viscosity,
    )


    v0 = filtered_velocity_field(
        coarse_grid, max_velocity, peak_wavenumber, iterations=16, random_state=42,
        device=device, batch_size=1,)
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v0)

    forcing_fn = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=coarse_grid, offsets=(v0[0].offset, v0[1].offset))

    ns2d = NavierStokes2DFVMProjection(
        viscosity=viscosity,
        grid=coarse_grid,
        bcs=(v0[0].bc, v0[1].bc),
        density=density,
        drag=0.1,
        forcing=forcing_fn,
        solver=step_fn,
        # set_laplacian=False,
    ).to(v0.device)

    #-----------ML setup------------------
    model, _ = get_model(model_type, model_config, checkpoint_path)
    model.to(device)
    

    for t in tqdm(range(round(simulation_time/dt))):
        v, p = step_fn.forward(v, dt, equation=ns2d)
        v += model(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--model_type", default="CNN", help="Model to train: [MLP, CNN, KAN, Transformer]")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    with torch.inference_mode():
        main(**ap.parse_args().__dict__)