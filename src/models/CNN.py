import torch.nn as nn
import torch.nn.functional as F
from tools.tools import paramToList


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
    
    def __str__(self):
        """
        Returns a summary of the model's architecture.
        """
        summary = "CNN Architecture:\n"
        for i, layer in enumerate(self.convs):
            summary += f"Layer {i}: {layer}\n"
        return summary



class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_sizes, strides, paddings, dropouts=0):
        super().__init__()
        self.structure = [in_channels] + hidden + [out_channels]
        self.convs = nn.ModuleList([nn.Conv2d(self.structure[i], self.structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(len(self.structure)-1)])
        self.fcs = nn.ModuleList([nn.Flatten(), nn.LazyLinear()])
        self.dropouts = [dropouts] * (len(self.convs) - 1) if isinstance(dropouts, (float, int)) else dropouts

if __name__ == "__main__":
    import json
    with open("src/models/configs/cnn1.json", "r") as f:
        config = json.load(f)
        cnn = CNN(config)
        print(cnn)