import torch
import torch.nn as nn

class ChannelAwareCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        vector_channels = config["structures"]["vector_components"]
        full_channels = config["structures"]["combined_cnn"]
        self.vector_component_convs = nn.ModuleList()
        self.full_channel_convs = nn.ModuleList()
        self.mixins = nn.ModuleList()
        for i in range(len(vector_channels)-1):
            self.vector_component_convs.append(
                nn.Conv2d(vector_channels[i], vector_channels[i+1], 3, 1, 1, groups=2)
            )
            if i+1%2 ==0:
                self.mixins.append(nn.Conv2d(vector_channels[i+1], vector_channels[i+1], 1, 1, 0))
            else:
                self.mixins.append(nn.Identity())
        
        for i in range(len(full_channels)-1):
            self.full_channel_convs.append(
                nn.Conv2d(full_channels[i], full_channels[i+1], 3 , 1, 1)
            )

        self.act = lambda x : x
    
    def forward(self, x):
        for conv, mixin in zip(self.vector_component_convs, self.mixins):
            x = mixin(self.act(conv(x)))
        
        for conv in self.full_channel_convs:
            x = self.act(conv(x))
        return x
