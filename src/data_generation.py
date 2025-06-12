import torch
from torch_cfd import grids, boundaries
from torch_cfd.initial_conditions import filtered_velocity_field

from torch_cfd.equations import stable_time_step
from torch_cfd.fvm import RKStepper, NavierStokes2DFVMProjection
from torch_cfd.forcings import KolmogorovForcing
import torch_cfd.finite_differences as fdm
import torch_cfd.tensor_utils as tensor_utils
import torch.utils._pytree as pytree

from safetensors.torch import save_file, load_file

from tqdm import tqdm
import random
from functools import partial

SAVEFILENAME = "data.safetensors"

# Uncomment For Colab

# from google.colab import drive
# drive.mount('/content/drive')

# SAVEFILENAME = "/content/drive/My Drive/data.safetensors"

# import os.path
# if not os.path.isfile(SAVEFILENAME):
#     with open(SAVEFILENAME, "w") as f:
#         pass

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

import os.path
if not os.path.isfile(SAVEFILENAME):
    save_file({}, SAVEFILENAME)

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

def main():
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


    pairs = []

    rng = random.randint(0, int(1e9))
    for i in range(sample_size):
        v0 = filtered_velocity_field(
        full_grid, max_velocity, peak_wavenumber, iterations=50, random_state=rng+i,
        device=device, batch_size=batch_size,)

        v = v0
        nan_count = 0
        time_between_samples = 1


        print("Starting Sim")
        for j in range(round(((simulation_time/time_between_samples)/dt)//inner_step)):
            for _ in tqdm(range(round(time_between_samples/dt))):
                v, p = step_fn.forward(v, dt, equation=ns2d_full)

            print("Generating Sample")
            coarse_u_t = downsample_staggered_velocity(full_grid, coarse_grid, v)
            for _ in tqdm(range(inner_step)):
                v, p = step_fn.forward(v, dt, equation=ns2d_full)
            v_coarse, p = step_fn.forward(coarse_u_t, delta_t, equation=ns2d_coarse)
            pairs.append((v.clone(), v_coarse.clone()))

            dataset = {}
            num_samples = len(dataset.keys())//2
            for pi, pair in enumerate(pairs):
                for b in range(batch_size):
                    dataset[f"{num_samples+pi*batch_size+b}_f"] = torch.stack(pair[0].data)[:,b].cpu()
                    dataset[f"{num_samples+pi*batch_size+b}_c"] = torch.stack(pair[1].data)[:,b].cpu()
            save_file(dataset, SAVEFILENAME)

            file2 = drive.CreateFile({'title': f'data{rng+i}_{j}.safetensors'})
            file2.SetContentFile(SAVEFILENAME)
            file2.Upload()

            pairs = []




if __name__ == "__main__":
    with torch.inference_mode():
        main()
