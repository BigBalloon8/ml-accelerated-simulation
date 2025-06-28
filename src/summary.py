import torchsummary as ts
from models import buildModel
import json
import torch

with open("src/models/configs/fullmodels/smartConv.json", "r") as f:
    configs = json.load(f)
    model = buildModel(configs)
    print(model)
ts.summary(model, input_data=torch.rand(1, 2, 64, 64))