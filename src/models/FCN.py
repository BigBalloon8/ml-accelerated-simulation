import torch.nn as nn
import torch.nn.functional as F
from .tools import paramToList, structureLoader, getAct, getModel, getLayers, getUpsample


class FCNHead(nn.Module):
    '''
    Decoder head for fully convolutional networks (FCN)
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
            upsample (dict): Parameters of upsampling (equivalent to the pooling overation to reverse): (\n
                in_channels (int): Size of input channels,\n
                out_channels (int): Size of output channels,\n
                method (str): Upsampling method,\n
                kernel_sizes (int): Upsampling kernel size,\n
                strides (int): Upsampling strides)\n
            bn (bool, optional): Whether to apply batch normalisation after convolution\n
    (*):\n If a float or int, applies the same value to all layers.\n
    \t If a list, must match the number of layers minus one.
    '''
    def __init__(self, config):
        super().__init__()
        self.act = getAct(config["activation_func"])
        structure = structureLoader(config["structures"]) 
        self.dropouts = paramToList(config["dropouts"], len(structure)-1)

        self.layers = getLayers(getModel(config, "CNN"))[0]
        self.upsample = getUpsample(structure[-1], structure[-1], config["upsample"])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.dropout(self.act(layer(x)), p=self.dropouts[i], training=self.training)
        return self.upsample(layer(x))

