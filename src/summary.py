import torchsummary as ts
from models import buildModel
import json
import torch

with open("src/models/configs/resNetBasicBlock1.json", "r") as f:
    configs = json.load(f)
    unet = buildModel(configs)
    print(unet)
ts.summary(unet, depth=5, input_data=torch.rand(1,2, 64, 64))