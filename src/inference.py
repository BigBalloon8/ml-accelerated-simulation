import torch
import torch.nn as nn

from tqdm import tqdm
import safetensors.torch as st

import argparse
import json
import os
from typing import Tuple

from models import MLP, CNN, Transformer, KAN

def hash_dict(x:dict):
    return str(hash(json.dumps(x, sort_keys=True)))

def get_model(name:str, config_file, checkpoint_path)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)

    if name.upper() == "MLP":
        model_base = MLP(config)
    elif name.upper() == "CNN":
        model_base = CNN(config)
    elif name.upper() == "KAN":
        model_base = KAN(config)
    elif name.upper() == "TRANSFORMER":
        model_base = Transformer(config)
    else:
        raise ValueError(f"Model type [{name}] not supported please select from |MLP|CNN|KAN|TRANSFORMER|")
    
    if f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.safetensors")
        model_weights = st.load_file(model_path)
        metadata = model_weights.pop("__metadata__")
        model_base.load_state_dict(model_weights)
    else:
        metadata = {"last_epoch:"-1}
    return model_base, metadata

def main(model_type, model_config, checkpoint_path):
    ...
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--model_type", default="CNN", help="Model to train: [MLP, CNN, KAN, Transformer]")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    with torch.inference_mode():
        main(**ap.parse_args().__dict__)