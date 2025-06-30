import torch
import torch.nn as nn

class SmartCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        vector_channels = config["structures"]["vector_components"]
        full_channels = config["structures"]["combined_cnn"]
        self.vector_component_convs = nn.ModuleList()
        self.full_channel_convs = nn.ModuleList()
        for i in range(len(vector_channels)-1):
            self.vector_component_convs.append(
                nn.Conv2d(vector_channels[i], vector_channels[i+1], 3, 1, 1, groups=2)
            )
        for i in range(len(full_channels)-1):
            self.full_channel_convs.append(
                nn.Conv2d(full_channels[i], full_channels[i+1], 3 , 1, 1)
            )
    
    def forward(self, x):
        for l in self.vector_component_convs:
            x = torch.relu(l(x))
        for l in self.full_channel_convs:
            x = torch.relu(l(x))
        return x



class SmartCNNBN(nn.Module):
    def __init__(self, config):
        super().__init__()
        vector_channels = config["structures"]["vector_components"]
        full_channels = config["structures"]["combined_cnn"]
        self.vector_component_convs = nn.ModuleList()
        self.vc_bn = nn.ModuleList()
        self.full_channel_convs = nn.ModuleList()
        self.fc_bn = nn.ModuleList()
        for i in range(len(vector_channels)-1):
            self.vector_component_convs.append(
                nn.Conv2d(vector_channels[i], vector_channels[i+1], 3, 1, 1, groups=2)
            )
            self.vc_bn.append(
                nn.ModuleList([nn.BatchNorm2d(vector_channels[i+1]//2),nn.BatchNorm2d(vector_channels[i+1]//2)])
            )
        for i in range(len(full_channels)-2):
            self.full_channel_convs.append(
                nn.Conv2d(full_channels[i], full_channels[i+1], 3 , 1, 1)
            )
            self.fc_bn.append(
                nn.BatchNorm2d(full_channels[i+1])
            )
        self.out_cnn = nn.Conv2d(full_channels[-2], full_channels[-1], 3, 1, 1)
    
    def forward(self, x):
        for l, bn in zip(self.vector_component_convs, self.vc_bn):
            x = torch.tanh_(l(x))
            x1, x2 = torch.chunk(x, 2, 1)
            x = torch.cat(
                (bn[0](x1), bn[1](x2)),
                dim=1
            )

        for l, bn in zip(self.full_channel_convs, self.fc_bn):
            x = torch.tanh_(l(x))
            x = bn(x)
        return self.out_cnn(x)