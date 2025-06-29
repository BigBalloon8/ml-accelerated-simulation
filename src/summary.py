import torchsummary as ts
from models import buildModel
import json
import torch

with open("src/models/configs/cnn1.json", "r") as f:
    configs = json.load(f)
    model = buildModel(configs)
    print(model)
print(model(torch.rand(1, 2, 64, 64)).shape)