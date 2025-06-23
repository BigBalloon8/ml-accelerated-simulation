import torch.nn as nn
import torch.nn.functional as F
from tools import paramToList, structureLoader, getAct

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with customizable dropout rates.
    Args:
        config (dict): A dictionary containing hyperparameters:\n
            structure (dict): Structure of Model: (\n
                input_size (int): Size of input,\n
                hidden_size (list): Size of hidden layers,\n
                output_size (int): Size of output)\n
            dropouts* (int, float or list): Dropout probability for each layer (except the last) (Set to 0 for no dropout)\n 
            activation_func (str): Name of desired activation function\n 
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])
        structure = structureLoader(config["structures"])
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)
        
        self.layers = nn.ModuleList([nn.Linear(structure[i], structure[i+1]) for i in range(len(structure)-1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                x = F.dropout(self.act(layer(x)), p=self.dropouts[i], training=True)
        return layer(x)



if __name__ == "__main__":
    import json
    with open("src/models/configs/mlp1.json", "r") as f:
        config = json.load(f)
        mlp = MLP(config)
        print(mlp.get_parameter)