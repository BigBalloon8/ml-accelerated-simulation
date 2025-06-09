import torch 

class MLP(torch.nn.Module):
    def __init__(self, structure):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(structure[i], structure[i+1]) for i in range(len(structure)-1)])
        print(self.linears)

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            if i < len(self.linears) - 1:
                x = torch.nn.functional.selu_(layer(x))
            else:
                x = layer(x)
        return x

if __name__ == "__main__":
    mlp = MLP([4, 10, 10, 3])