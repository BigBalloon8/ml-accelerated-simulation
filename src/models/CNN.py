import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module): # Just CNN layers
    def __init__(self, in_channels, hidden, out_channels, kernel_sizes, strides, paddings, dropouts=0, training=True):
        super().__init__()
        self.training = training
        self.structure = [in_channels] + hidden + [out_channels]
        self.convs = nn.ModuleList([nn.Conv2d(self.structure[i], self.structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(len(self.structure)-1)])
        self.dropouts = [dropouts] * (len(self.convs) - 1) if isinstance(dropouts, (float, int)) else dropouts
            
    def forward(self, x):
        for i, layer in enumerate(self.convs):
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu_(layer(x)), p=self.dropouts[i], training=self.training) ## Again ask abt efficiency
            else:
                break
        return layer(x)
    
    def __str__(self):
        pass ## TBC 



class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_sizes, strides, paddings, dropouts=0, training=True):
        super().__init__()
        self.training = training
        self.structure = [in_channels] + hidden + [out_channels]
        self.convs = nn.ModuleList([nn.Conv2d(self.structure[i], self.structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(len(self.structure)-1)])
        self.fcs = nn.ModuleList([nn.Flatten()nn.LazyLinear()])
        self.dropouts = [dropouts] * (len(self.convs) - 1) if isinstance(dropouts, (float, int)) else dropouts

        