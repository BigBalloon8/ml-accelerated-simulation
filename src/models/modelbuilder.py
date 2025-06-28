import torch.nn as nn
from .tools import getModel

   
class buildModel(nn.Module):
    """
    Neural network model builder.
    Args:
        configs (list): List of dictionaries of hyperparameters for buildings block of models
    """
    def __init__(self, configs):
        super().__init__()
        for i in range(len(configs)-1): # checking channel sizes
            if configs[i]["structures"]["out_channels"] != configs[i+1]["structures"]["in_channels"]:
                print(f"in_channels of {configs[i+1]['name']} does not match out_channels of {configs[i]['name']}.\nin_channels of {configs[i+1]['name']} has been corrected to out_channels of {configs[i]['name']}.")
                configs[i+1]["structures"]["in_channels"] = configs[i]["structures"]["out_channels"]
        self.models_name = [config["name"] for config in configs]

        self.models = nn.ModuleList([getModel(config) for config in configs])

    def forward(self, x):
        x1 = []
        for i, model in enumerate(self.models):
            if self.models_name[i].upper() == "UNETENCODERBLOCK":
                x, y = model(x)
                x1.append(y)
            elif self.models_name[i].upper() == "UNETDECODERBLOCK":
                x = model(x, x1.pop())
            else:
                x = model(x)
        return x

if __name__ == "__main__":
    import json
    with open("src/models/configs/fullmodels/uNet1.json", "r") as f:
        configs = json.load(f)
        unet = buildModel(configs)
        print(unet)