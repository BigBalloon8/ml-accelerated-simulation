from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.utils._pytree as pytree

import safetensors

import os

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
        dif_mag_std = 0.004482923474868822
        dif /= dif_mag_std

        return coarse, dif

class KolmogrovFlowFullData(Dataset):
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
            full = f.get_tensor(f"{idx%64}_f")

        return full


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

def get_kolomogrov_flow_data_loader_train_only(filename, batchsize=32, num_workers=8, prefetch_factor=2):
    dataset = KolmogrovFlowData(filename)
    return DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

def get_kolomogrov_flow_full_data(filename, batchsize=32, num_workers=8, prefetch_factor=2):
    dataset = KolmogrovFlowFullData(filename)
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
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    return train_loader, validation_loader
