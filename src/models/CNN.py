import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList, structureLoader, getAct


class CNN(nn.Module):
    """
    Convolution layers with customisable hyperparameters.
    Args:
        config (dict): A dictionary containing hyperparameters:\n
            structure (dict): Structure of Model: (\n
                in_channels (int): Size of input channels,\n
                hidden_channels (list): Size of hidden channels,\n
                out_channels (int): Size of output channels)\n
            kernel_sizes* (int or list): Dimension of kernel\n
            strides* (int or list): Step size that the kernel will take\n
            paddings* (int or list): Width of padding\n
            group* (int or list): number of groups (must divide both in_channels and out_channels) (Set to 1 for default)\n
            dropouts* (int, float or list): Dropout probability for each layer (except the last) (Set to 0 for no dropout)\n
            activation_func (str): Name of desired activation function\n 
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])        
        structure = structureLoader(config["structures"])
        kernel_sizes, strides, paddings, group = paramToList(config["kernel_sizes"], len(structure)-1), paramToList(config["strides"], len(structure)-1), paramToList(config["paddings"], len(structure)-1), paramToList(config["group"], len(structure)-1)
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)
        self.layers = nn.ModuleList([nn.Conv2d(structure[i], structure[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], groups=group[i]) for i in range(len(structure)-1)])
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.dropout(self.act(layer(x)), p=self.dropouts[i], training=True)
        return layer(x)



if __name__ == "__main__":
    import json
    with open("src/models/configs/cnn1.json", "r") as f:
        config = json.load(f)
        cnn = CNN(config[0])
        print(cnn)