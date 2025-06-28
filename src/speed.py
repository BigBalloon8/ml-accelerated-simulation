import torch
from torch_cfd import grids, boundaries
from torch_cfd.initial_conditions import filtered_velocity_field
from torch_cfd.equations import stable_time_step
from torch_cfd.fvm import RKStepper, NavierStokes2DFVMProjection
from torch_cfd.forcings import KolmogorovForcing
import torch_cfd.finite_differences as fdm
import torch_cfd.tensor_utils as tensor_utils
import torch.utils._pytree as pytree

from functools import partial
import time
import os
import argparse
import json

from models import buildModel
from log import Logger

def list_files_recursive(path, current_files=[]):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path, current_files)
        else:
            current_files.append(full_path)
    return current_files

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


def main(model_configs, log_file):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)

    logger = Logger("SPEED", log_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sample_size = 1024
    high_res = 1024
    low_res = 64
    batch_size = 64
    density = 1.0
    max_velocity = 7.0
    peak_wavenumber = 4.0
    cfl_safety_factor = 0.5
    viscosity = 1e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    burn_in_time = 1
    simulation_time = 30
    diam = 2 * torch.pi

    step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)

    full_grid = grids.Grid((high_res, high_res), domain=((0, diam), (0, diam)), device=device)

    coarse_grid = grids.Grid((low_res, low_res), domain=((0, diam), (0, diam)), device=device)

    dt = stable_time_step(
        dx=min(full_grid.step),
        max_velocity=max_velocity,
        max_courant_number=cfl_safety_factor,
        viscosity=viscosity,
    )

    delta_t = stable_time_step(
        dx=min(coarse_grid.step),
        max_velocity=max_velocity,
        max_courant_number=cfl_safety_factor,
        viscosity=viscosity,
    )


    inner_step = round(delta_t/dt)


    v0_full = filtered_velocity_field(
        full_grid, max_velocity, peak_wavenumber, iterations=16, random_state=42,
        device=device, batch_size=batch_size,)
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v0_full)

    v0_coarse = downsample_staggered_velocity(full_grid, coarse_grid, v0_full)


    forcing_fn_full = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=full_grid, offsets=(v0_full[0].offset, v0_full[1].offset))

    forcing_fn_coarse = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=coarse_grid, offsets=(v0_coarse[0].offset, v0_coarse[1].offset))

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
        solver=step_fn,
        # set_laplacian=False,
    ).to(v0_coarse.device)

    v = v0_full
    start = time.time()
    for i in range(round(0.25/dt)):
        v, p = step_fn.forward(v, dt, equation=ns2d_full)
    end = time.time()
    logger.log(f"Full Simulation, Time: {end-start}, N-Steps: {round(0.25/dt)}, dt={dt}")

    v = v0_coarse
    for i in range(round(0.25/delta_t)):
        v, p = step_fn.forward(v, delta_t, equation=ns2d_coarse)
    end = time.time()
    logger.log(f"Coarse Simulation, Time: {end-start}, N-Steps: {round(0.25/delta_t)}, dt={delta_t}")
    
    
    if os.path.isdir(model_configs) :
        model_config_files = list_files_recursive(model_configs)
    else:
        model_config_files = [model_configs]
    example_data = torch.rand(1, 2, 64, 64).to(device)

    for model_config_file in model_config_files:
        with open(model_config_file, "r") as f:
            model_config = json.load(f)
        logger.log(model_config_file)
        logger.log(model_config)
        model = buildModel(model_config).to(device)
        start = time.time()
        n_step = 1000

        for i in range(n_step):
            model(example_data)

        end = time.time()
        logger.log(f"Model Speed: {(end-start)/n_step:.4f}s/step\n")



if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--model_configs", default="./model.config", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/speed.log", help="path to log file")
    with torch.inference_mode():
        main(**ap.parse_args().__dict__)