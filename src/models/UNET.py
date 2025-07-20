import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from .tools import paramToList, structureLoader, getAct, getModel, getLayers, getPool, getUpsample

class UNetEncoderBlock(nn.Module):
    """
    Encoder Block for U-Net
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
            pooling (dict): Pooling parameters: (\n
                method (str): pooling method,\n
                kernel_sizes (int): pooling kernel size,\n
                strides (int): pooling strides)\n
            bn (bool, optional): Whether to apply batch normalisation after convolution\n 
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        structure = structureLoader(config["structures"])
        self.act = getAct(config["activation_func"], structure[1:]) if config["activation_func"].lower() == "prelu" else [getAct(config["activation_func"]) for i in range(len(structure)-1)]
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.layers = getLayers(getModel(config, "CNN"))[0]
        self.pool = getPool(config["pooling"])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.dropout(self.act[i](layer(x)), p=self.dropouts[i], training=self.training)
        return self.pool(x), x
    


class UNetDecoderBlock(nn.Module):
    """
    Decoder Block for U-Net
    Args:
        config (dict): A dictionary containing hyperparameters:\n 
            structure (dict): Structure of model: (\n
                in_channels (int): Size of input channels,\n
                hidden_channels (list): Size of hidden channels,\n
                out_channels (int): Size of output channels)\n
            kernel_sizes* (int or list): Dimension of kernel\n
            strides* (int or list): Step size that the kernel will take\n
            paddings* (int or list): Width of padding\n
            group* (int or list): number of groups (must divide both in_channels and out_channels) (Set to 1 for default)\n
            dropouts* (int, float or list): Dropout probability for each layer (except the last) (Set to 0 for no dropout)\n
            activation_func (str): Name of desired activation function\n
            upsample (dict): Parameters of upsampling (equivalent to the pooling overation to reverse): (\n
                method (str): Upsampling method,\n
                kernel_sizes (int): Upsampling kernel size,\n
                strides (int): Upsampling strides)\n
            bn (bool, optional): Whether to apply batch normalisation after convolution\n
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    """
    def __init__(self, config):
        super().__init__()
        structure = structureLoader(config["structures"])
        self.act = getAct(config["activation_func"], structure[1:]) if config["activation_func"].lower() == "prelu" else [getAct(config["activation_func"]) for i in range(len(structure)-1)]
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.upsample = getUpsample(structure[0], structure[0]//2, config["upsample"])
        self.layers = getLayers(getModel(config, "CNN"))[0]

    def forward(self, x, x1):
        x=self.upsample(x)
        dY, dX = x1.size()[2]-x.size()[2], x1.size()[3]-x.size()[3] # Match dimension of input
        x = F.pad(x, [dX//2, dX-dX//2, dY//2, dY-dY//2])
        x = cat((x,x1), dim=1)

        for i, layer in enumerate(self.layers):
            x = F.dropout(self.act[i](layer(x)), p=self.dropouts[i], training=self.training)
        return x




if __name__ == "__main__":
    import json
    with open("src/models/configs/uNetEncoderBlock1.json", "r") as f:
        config = json.load(f)
        unet = UNetEncoderBlock(config[0])
        print(unet)




