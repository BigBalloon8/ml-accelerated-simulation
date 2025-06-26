import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList, structureLoader, getAct

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with customizable dropout rates.
    Args:
        config (dict): A dictionary containing hyperparameters:\n
            structure (dict): Structure of Model: (\n
                in_channels (int): Size of input in unit of channels,\n
                hidden_channels (list): Size of hidden layers in unit of channels,\n
                out_channels (int): Size of output in unit of channels)\n
            dropouts* (int, float or list): Dropout probability for each layer (except the last) (Set to 0 for no dropout)\n 
            dimension (list): dismension of input [width, height]
            activation_func (str): Name of desired activation function\n 
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])
        structure = structureLoader(config["structures"])
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)
        dim = config["dimension"]
        
        self.layers = nn.ModuleList([nn.Flatten()]+[nn.Linear(structure[i]*dim[0]*dim[1], structure[i+1]*dim[0]*dim[1]) for i in range(len(structure)-1)])

    def forward(self, x):
        input_shape = x.shape
        for i, layer in enumerate(self.layers):
            if i > 0 and i < len(self.layers)-1:
                x = F.dropout(self.act(layer(x)), p=self.dropouts[i], training=self.training)
        return layer(x).reshape((input_shape[0], -1, input_shape[2], input_shape[3]))



if __name__ == "__main__":
    import json
    with open("src/models/configs/mlp1.json", "r") as f:
        config = json.load(f)
        mlp = MLP(config[0])
        print(mlp.get_parameter)