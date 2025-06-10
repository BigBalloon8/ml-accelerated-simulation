import torch
import torch.nn as nn

from data.dataloader import get_kolomogrov_flow_data_loader

FILENAME = ""

def train_step():
    ...

def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)

    train_dataloader, validation_dataloader = get_kolomogrov_flow_data_loader(FILENAME)

    