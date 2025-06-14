def paramToList(param, desc, dimension=0):
        '''
        Check and convert hyperparameters into lists
        Args:
            param (dict, float, int or list): hyperparameter
            desc (str): description of hyperparameter
            dimension (int): target list dimension (not applicable if type(param) == dict)
        '''
        if isinstance(param, dict) and dimension == 0:
            hold = list(param.values())
            return [hold[0]] + hold[1] + [hold[2]]
        elif isinstance(param, list) and len(param) == dimension:
            return param
        elif isinstance(param, (float, int)) and dimension > 0:
            return [param] * (dimension)
        elif isinstance(param, list) and len(param) != dimension:
            raise ValueError(f"{desc} list length must match the number of layers minus one.")
        elif not isinstance(param, list):
            raise TypeError(f"{desc} must be a float or a list of floats.")
        else:
            raise ValueError("dimension has to be greater than 0")


def getModel(name, config): ## ask abt relative imports not working (prefered to put tools.py in a folder in models)
    '''
    Fetch and instantiate a deep learning model 
    Args:
        name (str): name of model
        config (dict): hyperparameters to put in the model
    Return: model
    '''
    if name.upper() == "MLP":
        from MLP import MLP
        return MLP(config)
    elif name.upper() == "CNN":
        from CNN import CNN
        return CNN(config)
    elif name.upper() == "CONVNET":
        from ConvNet import ConvNet
        return ConvNet(config)
       