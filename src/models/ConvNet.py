import torch.nn as nn
import torch.nn.functional as F
from tools import getModel


class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([getModel(y, z) for y, z in config.items()])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = nn.Flatten(layer(x))
            else:
                break
        return layer(x)
    
if __name__ == "__main__":
    import json
    with open("src/models/configs/convNet1.json", "r") as f:
        config = json.load(f)
        convnet = ConvNet(config)
        print(convnet.get_parameter)
