import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList


class CNN(nn.Module): # Just CNN layers
    def __init__(self, config): #in_channels, hidden, out_channels, kernel_sizes, strides, paddings, dropouts=0
        super().__init__()
        structure = paramToList(config["structures"], "structures")
        kernel_sizes, strides, paddings = paramToList(config["kernel_sizes"], "kernel_sizes", len(structure)-1), paramToList(config["strides"], "strides", len(structure)-1), paramToList(config["paddings"], "paddings", len(structure)-1)
        self.dropouts = paramToList(config["dropouts"], "dropouts", len(structure)-1)
        self.convs = nn.ModuleList([nn.Conv2d(structure[i], structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(len(structure)-1)])        
            
    def forward(self, x):
        for i, layer in enumerate(self.convs):
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu_(layer(x)), p=self.dropouts[i], training=True)
            else:
                break
        return layer(x)


if __name__ == "__main__":
    import json
    with open("src/models/configs/cnn1.json", "r") as f:
        config = json.load(f)
        cnn = CNN(config)
        print(cnn)