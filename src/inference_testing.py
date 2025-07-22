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

import matplotlib.pyplot as plt

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

def downsample_tensor(x):
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

    def downsample_staggered_velocity_component(u, direction: int, factor: int=16):
        w = slice_along_axis(u, direction, slice(factor - 1, None, factor))
        block_size = tuple(1 if j == direction else factor for j in range(u.ndim))
        return block_reduce(w, block_size, torch.mean)

    factor = round(1024//64)
    result = []
    for j, u in enumerate(x):
        result.append(downsample_staggered_velocity_component(u, j, factor=factor))
    return torch.stack(result)

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

def graph_vec_field(x, file):
    Ux = x[0].cpu().numpy()
    Uy = x[1].cpu().numpy()
    import numpy as np
    X, Y = np.meshgrid(np.arange(Ux.shape[1]), np.arange(Ux.shape[0]))
    fig, ax = plt.subplots(figsize=(6, 6))

    mag = np.sqrt(Ux**2 + Uy**2)
    im = ax.imshow(mag, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Magnitude')

    ax.quiver(X, Y, Ux, Uy, color='w',)  # adjust scale for visual clarity
    ax.set_aspect('equal')
    ax.set_title('Vector Field (quiver)')

    plt.savefig(file)

def get_grid_data(x):
    return torch.stack(x.data)

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
        solver=full_step_fn,
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
    model = get_model(model_type, model_config, checkpoint_path, logger)
    model.to(device)
    if not no_segment:
        seg_model = get_segment_model(seg_model_name, seg_model_config, checkpoint_path)
        seg_model.to(device)

    mask_stream = torch.cuda.Stream()
    model_stream = torch.cuda.Stream()

    v_full = v0_full

    # Warmup
    for i in tqdm(range(1024), desc="Warmup"):
        v_full,_ = full_step_fn.forward(v_full, full_dt, equation=ns2d_full)
    
    #Actual Run
    v0_coarse = downsample_staggered_velocity(full_grid, coarse_grid, v0_full)
    v0_LC_coarse = downsample_staggered_velocity(full_grid, LC_coarse_grid, v0_full)

    v_coarse = v0_coarse
    v_LC_coarse = v0_LC_coarse

    MAE = nn.L1Loss()

    control_errors = [0.0]
    LC_errors = [0.0]
    
    for i in range(64):
        for i in range(16):
            v_full,_ = full_step_fn.forward(v_full, full_dt, equation=ns2d_full)
        
        v_coarse,_ = coarse_step_fn.forward(v_coarse, coarse_dt, equation=ns2d_coarse)

        v_LC_coarse,_ = LC_coarse_step_fn.forward(v_LC_coarse, coarse_dt, equation=ns2d_LC_coarse)
        coarse = get_grid_data(v_LC_coarse).squeeze(1)
        input_scale = 7
        output_scale = 0.004482923474868822
        coarse /= input_scale
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
            coarse += delta_v*output_scale
        v_LC_coarse.data = coarse.unsqueeze(1)
        coarsened_full = torch.vmap(downsample_tensor, in_dims=1, out_dims=1)(get_grid_data(v_full))
        control_errors.append(MAE(coarsened_full, get_grid_data(v_coarse)).item())
        LC_errors.append(MAE(coarsened_full, coarse).item())
        logger.log(f"Control Error: {control_errors[-1]}")
        logger.log(f"LC Error: {LC_errors[-1]}")

    graph_vec_field(get_grid_data(coarsened_full).squeeze(), "full.png")
    graph_vec_field(get_grid_data(get_grid_data(v_coarse)).squeeze(), "coarse.png")
    graph_vec_field(get_grid_data(coarse).squeeze(), "LC.png")


    
    







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