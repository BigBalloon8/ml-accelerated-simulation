import torchsummary as ts
from models import buildModel
import json
import torch

with open("/home/andrewrightj/ml-accelerated-simulation/src/models/configs/fullmodels/fastkan1.json", "r") as f:
    configs = json.load(f)
    unet = buildModel(configs)
    print(unet)
unet.train()
ts.summary(unet)
print(unet(torch.rand(1, 2, 64, 64)).shape)