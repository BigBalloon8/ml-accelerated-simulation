import torch
import torch.nn as nn

class SmartCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        vector_channels = config["vector_components"]
        full_channels = config["combined_cnn"]
        self.vector_component_convs = nn.ModuleList()
        self.full_channel_convs = nn.ModuleList()
        for i in range(len(vector_channels)-1):
            self.vector_component_convs.append(
                nn.Conv2d(vector_channels[i], vector_channels[i+1], 3, 1, 1, groups=2)
            )
        for i in range(len(full_channels)-1):
            self.full_channel_convs.append(
                nn.Conv2d(vector_channels[i], vector_channels[i+1])
            )
    
    def forward(self, x):
        for l in self.vector_component_convs:
            x = torch.relu(l(x))
        for l in self.full_channel_convs:
            x = torch.relu(l(x))
        return x


        