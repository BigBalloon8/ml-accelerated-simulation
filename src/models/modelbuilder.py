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

        self.models = nn.ModuleList([getModel(config) for config in configs])
        self.models_name = [config["name"] for config in configs]
        count = 0
        for i in range(len(configs)-1): # checking channel sizes
            if configs[i]["structures"]["out_channels"] != configs[i+1]["structures"]["in_channels"]:
                self.models.insert(i+count+1, nn.Sequential(nn.Conv2d(configs[i]["structures"]["out_channels"], configs[i+1]["structures"]["in_channels"], kernel_size=1), nn.BatchNorm2d(configs[i+1]["structures"]["in_channels"])) if configs[i]["bn"] else nn.Conv2d(configs[i]["structures"]["out_channels"], configs[i+1]["structures"]["in_channels"], kernel_size=1))
                self.models_name.insert(i+count+1, "conv1x1")
                print(f"in_channels of block {i+1} does not match out_channels of block {i}.\n a 1x1 convolution has been added to match channels")


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