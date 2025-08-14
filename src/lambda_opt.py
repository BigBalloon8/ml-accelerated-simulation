import argparse

from segmentation_train import main as main_seg
from train_v3 import main as main_train
from log import Logger

def main(data_path, model_type, model_config, checkpoint_path, log_file):
    LAMBDAS = [0.00029794 0.00050433 0.00073358 0.00100856 0.00135783 0.00182783, 0.0025126  0.00364921 0.00612721]

    logger = Logger("Lambda Testing", log_file)

    for l in LAMBDAS:
        logger.log(f"------------------- lambda: {l} --------------------")
        model_name = f"{model_type}_{l}"
        main_seg(data_path, model_name, model_config, checkpoint_path, log_file, False, l)
        main_train(data_path, model_name, model_config, checkpoint_path, log_file, False, model_name, model_config)

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/general.log", help="path to log file")
    main(**ap.parse_args().__dict__)