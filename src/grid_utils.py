import torch
import torch_cfd.tensor_utils as tensor_utils
import torch.utils._pytree as pytree

from torch_cfd import grids

from functools import partial

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

def get_grid_data(x):
    return torch.stack(x.data)

def tensor_to_grid(x, grid, grid_variable_vector):
    grid_array_0 = grids.GridVariable(x[0], offset=grid_variable_vector[0].offset, grid=grid, bc=grid_variable_vector[0].bc)
    grid_array_1 = grids.GridVariable(x[1], offset=grid_variable_vector[1].offset, grid=grid, bc=grid_variable_vector[1].bc)
    return grids.GridVariableVector((grid_array_0, grid_array_1))