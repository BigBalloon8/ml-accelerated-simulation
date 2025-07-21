import torchsummary as ts
from models import buildModel
from models.tools import*
import json
import torch
import os

config_list = os.listdir("src/models/configs/fullmodels/")
print(config_list)
for name in config_list:
    if name[-5:] == ".json" and name[:4] != "BCAT":
        with open(f"src/models/configs/fullmodels/{name}", "r") as f:
            configs = json.load(f)
            model = buildModel(configs)
            #print(model)
        #print(model(torch.rand(1, 2, 64, 64)).shape)
        print(name)
        ts.summary(model, torch.rand(1, 2, 64, 64), depth=5)