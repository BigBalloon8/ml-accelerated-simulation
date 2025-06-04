import torch
from torch_cfd import grids, boundaries
from torch_cfd.initial_conditions import filtered_velocity_field

from torch_cfd.equations import stable_time_step
from torch_cfd.fvm import RKStepper, NavierStokes2DFVMProjection
from torch_cfd.forcings import KolmogorovForcing
import torch_cfd.finite_differences as fdm
import torch_cfd.tensor_utils as tensor_utils


from tqdm import tqdm

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


def downsample_staggered_velocity_component(u, direction: int, factor: int):
    w = tensor_utils.slice_along_axis(u, direction, slice(factor - 1, None, factor))
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
            array = downsample_staggered_velocity_component(u.data, direction,
                                                            round(factor))
            grid_array = grids.GridTensor(array, offset=u.offset, grid=destination_grid)
            return grids.GridVariable(grid_array, bc=u.bc)
        result.append(downsample(u, j, round(factor)))
        return tuple(result)

def coarsen(full_res):
    
    return full_res

def main():
    sample_size = 1024

    n = 2046
    batch_size = 16
    density = 1.0
    max_velocity = 7.0
    peak_wavenumber = 4.0
    cfl_safety_factor = 0.5
    viscosity = 1e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    burn_in_time = 10
    sim_steps = 2048+512
    diam = 2 * torch.pi

    step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)

    full_grid = grids.Grid((n, n), domain=((0, diam), (0, diam)), device=device)

    dt = stable_time_step(
        dx=min(full_grid.step),
        max_velocity=max_velocity,
        max_courant_number=cfl_safety_factor,
        viscosity=viscosity,
    )

    forcing_fn = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
        grid=full_grid, offsets=(v0[0].offset, v0[1].offset))

    
    v0 = filtered_velocity_field(
        full_grid, max_velocity, peak_wavenumber, iterations=3, random_state=42,
        device=device, batch_size=batch_size,)
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v0)

    ns2d = NavierStokes2DFVMProjection(
        viscosity=viscosity,
        grid=full_grid,
        bcs=(v0[0].bc, v0[1].bc),
        density=density,
        drag=0.1,
        forcing=forcing_fn,
        solver=step_fn,
        # set_laplacian=False,
    ).to(v0.device)



    for i in range(sample_size):
        v0 = filtered_velocity_field(
        full_grid, max_velocity, peak_wavenumber, iterations=3, random_state=i,
        device=device, batch_size=batch_size,)

        v = v0
        nan_count = 0

        for _ in range(round(burn_in_time/dt)):
            v, p = step_fn.forward(v, dt, equation=ns2d)
        



if __name__ == "__main__":
      main()



