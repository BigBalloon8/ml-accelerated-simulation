import argparse
import json
import hashlib
import os

from segmentation_train import main as main_seg
from train_v3 import main as main_train
from inference_testing import main as inference_main
from log import Logger

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

def has_been_trained_seg(name:str, config_file, checkpoint_path):
    with open(config_file, "r") as f:
        config = json.load(f)

    model_exists = f"{name}_{hash_dict(config)}_seg.safetensors" in os.listdir(checkpoint_path)
    if model_exists:
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}_seg.json"), "r") as f:
            metadata = json.load(f)
        return metadata["last_epoch"] >= 99
    else:
        return False

def has_been_trained(name:str, config_file, checkpoint_path):
    with open(config_file, "r") as f:
        config = json.load(f)

    model_exists =  f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path)
    if model_exists:
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.json"), "r") as f:
            metadata = json.load(f)
        return metadata["last_epoch"] >= 99
    else:
        return False


def main(data_path, model_type, model_config, checkpoint_path, log_file):
    LAMBDAS = [0.00029794, 0.00050433, 0.00073358, 0.00100856, 0.00135783, 0.00182783, 0.0025126, 0.00364921, 0.00612721]

    logger = Logger("Lambda Testing", log_file)

    for l in LAMBDAS:
        logger.log(f"------------------- lambda: {l} --------------------")
        model_name = f"{model_type}_{l}"
        if not has_been_trained_seg(model_name, model_config, checkpoint_path):
            main_seg(data_path, model_name, model_config, checkpoint_path, log_file, False, l)
        if not has_been_trained(model_name, model_config, checkpoint_path):
            main_train(data_path, model_name, model_config, checkpoint_path, log_file, False, model_name, model_config)
        inference_main(model_name, model_config, checkpoint_path, log_file, False, model_name, model_config)

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/general.log", help="path to log file")
    main(**ap.parse_args().__dict__)