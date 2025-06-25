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
        from torch.nn.functional import relu_
        return relu_
    elif name.lower() == "selu":
        from torch.nn.functional import selu_
        return selu_
    elif name.lower() == "gelu":
        from torch.nn.functional import gelu
        return gelu
    else:
        raise ValueError(f"Activation function {name} does not exist")


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


def getModel(module): 
    '''
    Fetch and instantiate a deep learning model 
    Args:
        module (dict): hyperparameters to put in the model
    Return: model
    '''
    if module["name"].upper() == "MLP":
        from MLP import MLP
        return MLP(module)
    elif module["name"].upper() == "CNN":
        from CNN import CNN
        return CNN(module)
    elif module["name"].upper() == "RESNETBLOCK":
        from ResNet import ResNetBlock
        return ResNetBlock(module)
    elif module["name"].upper() == "RESNEXTBLOCK":
        from ResNet import ResNeXtBlock
        return ResNeXtBlock(module)
    elif module["name"].upper() == "DENSEBLOCK":
        from DenseNet import DenseBlock
        return DenseBlock(module)
    elif module["name"].upper() == "UNETENCODERBLOCK":
        from UNET import UNetEncoderBlock
        return UNetEncoderBlock(module)
    elif module["name"].upper() == "UNETDECODERBLOCK":
        from UNET import UNetDecoderBlock
        return UNetDecoderBlock(module)
    elif module["name"].upper() == "TRANSFORMER":
        pass
    elif module["name"].upper() == "KAN":
        pass
    

def getLayers(model):
    '''
    Fetch the layers from a deep learning model
    Args:
        model (class): the model to fetch from
    Return: a dictionary of the layers in the model
    '''
    return list(model.children())
