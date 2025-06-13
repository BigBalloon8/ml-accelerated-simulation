import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with customizable dropout rates.
    Args:
        structure (list): A list of integers defining the number of neurons in each layer.
        dropouts (float or list): Dropout probability for each layer except the last. 
                                  If a float, applies the same dropout to all layers.
                                  If a list, must match the number of layers minus one.
        training (bool): Whether the model is in training mode (default: True).
    """
    def __init__(self, structure, dropouts=0, training=True):
        super(MLP, self).__init__()
        self.training = training
        self.linears = nn.ModuleList([nn.Linear(structure[i], structure[i+1]) for i in range(len(structure)-1)])
        
        # check and set dropout probabilities
        if isinstance(dropouts, float) or isinstance(dropouts, int):
            self.dropouts = [dropouts] * (len(self.linears)-1)
        elif isinstance(dropouts, list) and len(dropouts) != len(self.linears)-1:
            raise ValueError("Dropout list length must match the number of layers minus one.")
        elif not isinstance(dropouts, list):
            raise TypeError("Dropouts probability must be a float or a list of floats.")
        
    def forward(self, x):
        for i, layer in enumerate(self.linears):
            if i < len(self.linears)-1:
                x = F.dropout(F.selu_(layer(x)), p=self.dropouts[i], training=self.training) ## ask about efficuency compared to nn.dropout and nn.selu
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
        return summary ## complete 

if __name__ == "__main__":
    mlp = MLP([4, 10, 10, 3])