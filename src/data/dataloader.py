from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.utils._pytree as pytree

import safetensors

import os

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


class KolmogrovFlowData(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir        
        self.files = os.listdir(data_dir)
        self.n_samples = len(self.files)*64
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        file = self.files[idx//64]
        with safetensors.safe_open(os.path.join(self.data_dir, file), "pt") as f:
            c_full = f.get_tensor(f"{idx%64}_f")
            coarse = f.get_tensor(f"{idx%64}_c")
        # factor = round(full.shape[1]//coarse.shape[1])
        # result = []
        # for j, u in enumerate(full):
        #     result.append(downsample_staggered_velocity_component(u, j, factor=factor))
        # c_full = torch.stack(result)

        dif = c_full - coarse
        max_velocity = 7
        coarse /= max_velocity
        return coarse, dif


def get_kolomogrov_flow_data_loader(filename, batchsize=32, num_workers=8, prefetch_factor=2):
    dataset = KolmogrovFlowData(filename)
    train_ds, val_ds = random_split(dataset, [0.8,0.2])
    train_loader = DataLoader(
        train_ds,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
        )
    validation_loader = DataLoader(
        val_ds,
        batch_size=batchsize,
        shuffle=False,
        #num_workers=0,
        pin_memory=True
    )
    return train_loader, validation_loader

