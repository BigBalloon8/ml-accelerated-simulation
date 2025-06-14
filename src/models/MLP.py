import torch.nn as nn
import torch.nn.functional as F
from tools.tools import paramToList

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with customizable dropout rates.
    Args:
        structure (list): A list of integers defining the number of neurons in each layer.
        dropouts (float or list): Dropout probability for each layer except the last. 
                                  If a float, applies the same dropout to all layers.
                                  If a list, must match the number of layers minus one.
    Return:
        Hyperparameter in a list
    """
    def __init__(self, config):
        super(MLP, self).__init__()
        structure = paramToList(config["structures"], "structures")
        self.dropouts = paramToList(config["dropouts"], "dropouts", len(structure)-1)
        self.linears = nn.ModuleList([nn.Linear(structure[i], structure[i+1]) for i in range(len(structure)-1)])

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            if i < len(self.linears)-1:
                x = F.dropout(F.selu_(layer(x)), p=self.dropouts[i], training=True)
            else:
                break
        return layer(x)
    
    def __str__(self):
        """
        Returns a summary of the model's architecture.
        """
        summary = "MLP Architecture:\n"
        for i, layer in enumerate(self.linears):
            summary += f"Layer {i}: {layer}\n"
        return summary  

if __name__ == "__main__":
    import json
    with open("src/models/configs/mlp1.json", "r") as f:
        config = json.load(f)
        mlp = MLP(config)
        print(mlp)