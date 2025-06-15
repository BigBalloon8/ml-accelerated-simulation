import torch.nn as nn
from tools import getModel


class ConvNet(nn.Module):
    """
    Standard convolutional neural network with customizable hyperparameters.
    Args:
        config (dict): A dictionary containing hyperparameters
            CNN (dict): Hyperparameters for the convolutional layers
            MLP (dict): Hyperparameters for the fully-connected layers

    """
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
