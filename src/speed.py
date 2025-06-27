import torch

import time
import os
import argparse
import json

from models import buildModel
from log import Logger

def list_files_recursive(path, current_files=[]):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path, current_files)
        else:
            current_files.append(full_path)
    return current_files

def main(model_configs, log_file):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)

    logger = Logger("SPEED", log_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if os.path.isdir(model_configs) :
        model_config_files = list_files_recursive(model_configs)
    else:
        model_config_files = [model_configs]
    example_data = torch.rand(1, 2, 64, 64).to(device)

    for model_config_file in model_config_files:
        with open(model_config_file, "r") as f:
            model_config = json.load(f)
        logger.log(model_config_file)
        logger.log(model_config)
        model = buildModel(model_config).to(device)
        start = time.time()
        n_step = 1000

        for i in range(n_step):
            model(example_data)

        end = time.time()
        logger.log(f"Model Speed: {(end-start)/n_step:.4f}s/step\n")



if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--model_configs", default="./model.config", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/speed.log", help="path to log file")
    with torch.inference_mode():
        main(**ap.parse_args().__dict__)