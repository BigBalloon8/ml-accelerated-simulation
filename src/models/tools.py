import torch

def paramToList(param, dimension):
        '''
        Check and convert hyperparameters into lists
        Args:
            param (float, int or list): hyperparameter
            dimension (int): target list dimension
        Return:
            A list of hyperparameter of length number of layers minus one
        '''
        if isinstance(param, list) and len(param) == dimension:
            return param
        elif isinstance(param, (float, int)) and dimension > 0:
            return [param] * (dimension)
        elif isinstance(param, list) and len(param) != dimension:
            raise ValueError("Hyperparameter list length must match the number of layers minus one.")
        elif not isinstance(param, list):
            raise TypeError("Hyperparameter must be a float or a list of floats.")
        else:
            raise ValueError("Dimension has to be greater than 0")


def structureLoader(structure):
    """
    Format the model structure into a list
    Args:
        structure (dict): Input size, Hidden layers sizes and Output size
    Return:
        A list of the structure of the neural network
    """
    if isinstance(structure, dict):
        hold = list(structure.values())
        return [hold[0]] + hold[1] + [hold[2]] if len(hold) == 3 else hold
    else:
        raise TypeError("Structure has to be a dictionary with 3 entries")


def getAct(name):
    """
    Load activation functions
    Args:
        name (str): name of activation function
    Return:
        The corresponding activation function
    """
    if name.lower() == "relu":
        return torch.relu_
    elif name.lower() == "selu":
        return torch.selu_
    elif name.lower() == "gelu":
        from torch.nn.functional import gelu
        return gelu
    elif name.lower() == "lrelu":
        from torch.nn.functional import leaky_relu_
        return leaky_relu_
    elif name.lower() == "rrelu":
        from torch.nn.functional import rrelu_
        return rrelu_
    elif name.lower() == "identity":
        return lambda x: x
    elif name.lower() == "tanh":
        return torch.tanh
    elif name.lower() == "sigmoid":
        return torch.sigmoid
    elif name.lower() == "shifted_sigmoid":
        @torch.compile
        def shifted_sigmoid(x:torch.Tensor):
            return 2*torch.sigmoid_(x) - 1
        return shifted_sigmoid
    else:
        raise ValueError(f"Activation function [{name}] not defined")


def getPool(config):
    """
    Load pooling functions
    Args:
        config (dict): configuration for pooling function
    Return:
        The corresponding pooling function with valid configuration
    """
    if config["method"].lower() == "max":
        from torch.nn import MaxPool2d
        return MaxPool2d(config["kernel_sizes"], config["strides"])
    elif config["method"].lower() == "avg":
        from torch.nn import AvgPool2d
        return AvgPool2d(config["kernel_sizes"], config["strides"])


def getModel(config, name=None): 
    '''
    Fetch and instantiate a deep learning model 
    Args:
        config (dict): hyperparameters to put in the model
        name (optional): name of model
    Return: model
    '''
    name = config["name"].upper() if name is None else name.upper()
    if  name == "MLP":
        from .MLP import MLP
        return MLP(config)
    elif name == "CNN":
        from .CNN import CNN
        return CNN(config)
    elif name == "RESNETBLOCK" or name == "RESNETBASICBLOCK" or name == "RESNETBOTTLENECKBLOCK":
        from .ResNet import ResNetBlock
        return ResNetBlock(config)
    elif name == "RESNEXTBLOCK":
        from .ResNet import ResNeXtBlock
        return ResNeXtBlock(config)
    elif name == "DENSENETBLOCK" or name == "DENSEBLOCK":
        from .DenseNet import DenseBlock
        return DenseBlock(config)
    elif name == "UNETENCODERBLOCK":
        from .UNET import UNetEncoderBlock
        return UNetEncoderBlock(config)
    elif name == "UNETDECODERBLOCK":
        from .UNET import UNetDecoderBlock
        return UNetDecoderBlock(config)
    elif name == "TRANSFORMER":
        from .Transformer import Transformer
        return Transformer(config)
    elif name == "KAN":
        from .KAN import KAN
        return KAN(config)
    elif name == "SMARTCONV":
        from .SmartCNN import SmartCNN
        return SmartCNN(config)
    elif name == "BCAT":
        from .BCAT import BCAT
        return BCAT(config, 64, 2)
    elif name == "CHANNELAWARE":
        from .ChannelAwareCNN import ChannelAwareCNN
        return ChannelAwareCNN(config)
    elif name == "3DCNN":
        from .CNN3D import CNN3D
        return CNN3D(config)
    elif name == "3DCNNSMART":
        from .CNN3D import CNN3DSmart
        return CNN3DSmart(config)
    elif name == "FCNHEAD":
        from .FCN import FCNHead
        return FCNHead(config)
    else:
        raise ValueError(f"Model Name [{name}] not defined")
    

def getLayers(model):
    '''
    Fetch the layers from a deep learning model
    Args:
        model (class): the model to fetch from
    Return: a dictionary of the layers in the model
    '''
    return list(model.children())


def getUpsample(in_channels, out_channels, config):
    '''
    Fetch upsampling methods
    Args:
        in_channels (int): Size of input channels,\n
        out_channels (int): Size of output channels,\n
        config (dict): Upsampling configuration: (\n
            method (str): Upsampling method,\n
            kernel_sizes (int): Upsampling kernel size,\n
            strides (int): Upsampling strides)\n
    Return: upsampling layers
    '''
    if config["method"].lower() == "bilinear":
        from torch.nn import Sequential, Conv2d, Upsample
        return Sequential(Upsample(scale_factor=config["stride"], mode="bilinear", align_corners=True), Conv2d(in_channels, out_channels, kernel_size=1))
    elif config["method"].lower() == "convtranspose" or config["method"].lower() == "convt":
        from torch.nn import ConvTranspose2d
        return ConvTranspose2d(in_channels, config["out_channels"], kernel_size=config["kernel_size"], stride=config["stride"])
    elif config["method"].lower() == "nearest":
        from torch.nn import Sequential, Conv2d, Upsample
        return Sequential(Upsample(scale_factor=config["stride"], mode="nearest", align_corners=True), Conv2d(in_channels, out_channels, kernel_size=1))
    elif config["method"].lower() == "bicubic":
        from torch.nn import Sequential, Conv2d, Upsample
        return Sequential(Upsample(scale_factor=config["stride"], mode="bicubic", align_corners=True), Conv2d(in_channels, out_channels, kernel_size=1))
    else:
        raise ValueError(f"Model Name [{config["method"].lower()}] not defined")