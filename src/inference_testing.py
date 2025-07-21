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
from functools import partial

from models import buildModel
from log import Logger

def block_reduce(array, block_size, reduction_fn):
    new_shape = []
    for b, s in zip(block_size, array.shape):
        multiple, residual = divmod(s, b)
        if residual != 0:
            raise ValueError('`block_size` must divide `array.shape`;'
                            f'got {block_size}, {array.shape}.')
        new_shape += [multiple, b]
    multiple_axis_reduction_fn = reduction_fn
    for j in reversed(range(array.ndim)):
        multiple_axis_reduction_fn = torch.vmap(multiple_axis_reduction_fn, j)
    return multiple_axis_reduction_fn(array.reshape(new_shape))


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"invalid axis {axis} for ndim {ndim}")
    if axis < 0:
        axis += ndim
    return axis

def slice_along_axis(
    inputs, axis: int, idx, expect_same_dims: bool = True):

    arrays, tree_def = pytree.tree_flatten(inputs)
    ndims = set(a.ndim for a in arrays)
    if expect_same_dims and len(ndims) != 1:
        raise ValueError(
            "arrays in `inputs` expected to have same ndims, but have "
            f"{ndims}. To allow this, pass expect_same_dims=False"
        )
    sliced = []
    for array in arrays:
        ndim = array.ndim
        slc = tuple(
            idx if j == _normalize_axis(axis, ndim) else slice(None)
            for j in range(ndim)
        )
        sliced.append(array[slc])
    return pytree.tree_unflatten(sliced, tree_def)

def downsample_staggered_velocity_component(u, direction: int, factor: int):
    w = slice_along_axis(u, direction, slice(factor - 1, None, factor))
    block_size = tuple(1 if j == direction else factor for j in range(u.ndim))
    return block_reduce(w, block_size, torch.mean)


def downsample_staggered_velocity(
    source_grid: grids.Grid,
    destination_grid: grids.Grid,
    velocity
):
    factor = destination_grid.step[0] / source_grid.step[0]
    result = []
    for j, u in enumerate(velocity):
        def downsample(u: grids.GridVariable, direction: int,
                     factor: int) -> grids.GridVariable:
            array = torch.vmap(partial(downsample_staggered_velocity_component, direction=direction, factor=round(factor)))(u.data)
            grid_array = grids.GridVariable(array.squeeze(), offset=u.offset, grid=destination_grid, bc=u.bc)
            return grid_array
        result.append(downsample(u, j, round(factor)))
    return grids.GridVariableVector(tuple(result))

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

def get_model(name:str, config_file, checkpoint_path, logger, new_run)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)
    logger.log(f"Model Config: {config}")

    model_base = buildModel(config)
    
    if f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path) and not new_run:
        print(f"Model Found in {checkpoint_path}: {name}_{hash_dict(config)}.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.safetensors")
        opt_path = os.path.join(checkpoint_path, f"{name}_ADAM_{hash_dict(config)}.pt")
        model_weights = st.load_file(model_path)
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.json"), "r") as f:
            metadata = json.load(f)
        model_base.load_state_dict(model_weights)
        opt_state = torch.load(opt_path)
    else:
        metadata = {"last_epoch":-1}
        opt_state = None
    return model_base, metadata, opt_state

def get_segment_model(name, config_file, checkpoint_path):
    with open(config_file, "r") as f:
        config = json.load(f)

    model_base = buildModel(config)
    model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}_seg.safetensors")
    model_weights = st.load_file(model_path)
    model_base.load_state_dict(model_weights)
    return model_base

def get_mask(x, segment_model):
    with torch.no_grad():
        logits = torch.softmax(segment_model(x), dim=1)
        return torch.argmax(logits, dim=1, keepdim=True).to(torch.bool)

def main(model_type, model_config, checkpoint_path, log_file, no_segment, seg_model_name, seg_model_config):
    # ---------- Simulation Setup ---------------
    high_res = 1024
    low_res = 64
    density = 1.0
    max_velocity = 7.0
    peak_wavenumber = 4.0
    cfl_safety_factor = 0.5
    viscosity = 1e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    diam = 2 * torch.pi
    simulation_time = 30
    logger = Logger(model_type, log_file)


    full_step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)
    coarse_step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)
    LC_coarse_step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)



    full_grid = grids.Grid((high_res, high_res), domain=((0, diam), (0, diam)), device=device)
    coarse_grid = grids.Grid((low_res, low_res), domain=((0, diam), (0, diam)), device=device)
    LC_coarse_grid = grids.Grid((low_res, low_res), domain=((0, diam), (0, diam)), device=device)

    full_dt = stable_time_step(
        dx=min(full_grid.step),
        max_velocity=max_velocity,
        max_courant_number=cfl_safety_factor,
        viscosity=viscosity,
    )
    coarse_dt = full_dt*16

    v0_full = filtered_velocity_field(
        full_grid, max_velocity, peak_wavenumber, iterations=16, random_state=42,
        device=device, batch_size=1)

    v0_coarse = downsample_staggered_velocity(full_grid, coarse_grid, v0_full)
    v0_LC_coarse = downsample_staggered_velocity(full_grid, LC_coarse_grid, v0_full)

    forcing_fn_full = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=full_grid, offsets=(v0_full[0].offset, v0_full[1].offset))

    forcing_fn_coarse = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=coarse_grid, offsets=(v0_coarse[0].offset, v0_coarse[1].offset))
    
    forcing_fn_LC_coarse = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=LC_coarse_grid, offsets=(v0_LC_coarse[0].offset, v0_LC_coarse[1].offset))

    ns2d_full = NavierStokes2DFVMProjection(
        viscosity=viscosity,
        grid=full_grid,
        bcs=(v0_full[0].bc, v0_full[1].bc),
        density=density,
        drag=0.1,
        forcing=forcing_fn_full,
        solver=step_fn,
        # set_laplacian=False,
    ).to(v0_full.device)

    ns2d_coarse = NavierStokes2DFVMProjection(
        viscosity=viscosity,
        grid=coarse_grid,
        bcs=(v0_coarse[0].bc, v0_coarse[1].bc),
        density=density,
        drag=0.1,
        forcing=forcing_fn_coarse,
        solver=coarse_step_fn,
        # set_laplacian=False,
    ).to(v0_coarse.device)

    ns2d_LC_coarse = NavierStokes2DFVMProjection(
        viscosity=viscosity,
        grid=LC_coarse_grid,
        bcs=(v0_LC_coarse[0].bc, v0_LC_coarse[1].bc),
        density=density,
        drag=0.1,
        forcing=forcing_fn_LC_coarse,
        solver=LC_coarse_step_fn,
        # set_laplacian=False,
    ).to(LC_coarse_grid.device)

    #-----------ML setup------------------
    model, _ = get_model(model_type, model_config, checkpoint_path)
    model.to(device)
    if not no_segment:
        seg_model = get_segment_model(seg_model_name, seg_model_config, checkpoint_path)
        seg_model.to(device)

    error_fn = nn.MSELoss()

    mask_stream = torch.cuda.Stream()
    model_stream = torch.cuda.Stream()

    v_full = v0_full
    v_coarse = v0_coarse
    v_LC_coarse = v0_LC_coarse

    MAE = nn.L1Loss()

    control_errors = []
    LC_errors = []

    for i in range(64):
        for i in range(16):
            v_full = full_step_fn.forward(v_full, full_dt, equation=ns2d_full)
        
        v_coarse = coarse_step_fn.forward(v_coarse, coarse_dt, equation=ns2d_coarse)

        v_LC_coarse = LC_coarse_step_fn.forward(v_LC_coarse, coarse_dt, equation=ns2d_LC_coarse)
        coarse = v_LC_coarse.data.squeeze(1)
        if not no_segment:
            with torch.cuda.stream(mask_stream):
                mask = get_mask(coarse, seg_model)
            with torch.cuda.stream(model_stream):
                delta_v = model.forward(coarse)
            torch.cuda.synchronize()
            delta_v = delta_v.masked_fill(mask, 0)
            coarse += delta_v
        else:
            delta_v = model.forward(coarse)
            coarse += delta_v
        v_LC_coarse.data = coarse.unsqeeze(1)

        control_errors.append(MAE(v_full.data, v_coarse.data).item())
        LC_errors.append(MAE(v_full.data, coarse).item())









if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--model_type", default="CNN", help="Model to train: [MLP, CNN, KAN, Transformer]")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/inference.log", help="path to log file")
    ap.add_argument("--no_segment", action="store_true")
    ap.add_argument("--seg_model_name", default="segment_big", help="segment model name")
    ap.add_argument("--seg_model_config", default="models/configs/fullmodels/segnet.json", help="segment model config")
    with torch.inference_mode():
        main(**ap.parse_args().__dict__)