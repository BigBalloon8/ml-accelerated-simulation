import torch
import torch.nn as nn

from tqdm import tqdm
import safetensors.torch as st

import argparse
import json
import os
from typing import Tuple
import random

from data.dataloader import get_kolomogrov_flow_data_loader
from models import buildModel
from log import Logger

def hash_dict(x:dict):
    return str(hash(json.dumps(x, sort_keys=True)))

def get_model(name:str, config_file, checkpoint_path, logger, new_run)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)
    logger.log(f"Model Config: {config}")

    model_base = buildModel(config)
    
    if f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path) and not new_run:
        print(f"Model Found in {checkpoint_path}: {name}_{hash_dict(config)}.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.safetensors")
        model_weights = st.load_file(model_path)
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.json"), "r") as f:
            metadata = json.loads(f)
        model_base.load_state_dict(model_weights)
    else:
        metadata = {"last_epoch":-1}
    return model_base, metadata
    
        
def save_model(model:nn.Module, model_type, checkpoint_path, model_config, metadata=None):
    with open(model_config, "r") as f:
        config = json.load(f)
    model_path = os.path.join(checkpoint_path, f"{model_type}_{hash_dict(config)}.safetensors")
    with open(os.path.join(checkpoint_path, f"{model_type}_{hash_dict(config)}.json"), "w") as f:
        json.dump(metadata, f)
    st.save_model(model, model_path)
    
    
def main(data_path, model_type, model_config, checkpoint_path, log_file, new_run):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    random.seed(2025)

    logger = Logger(model_type, log_file)

    EPOCHS = 10
    batchsize = 32
    gradient_accumulation_steps = 1
    local_batch_size = batchsize // gradient_accumulation_steps

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataloader, validation_dataloader = get_kolomogrov_flow_data_loader(data_path, batchsize=local_batch_size)

    model, metadata = get_model(model_type, model_config, checkpoint_path, logger, new_run)
    model = model.to(device)

    criterion = nn.MSELoss()
    val_criterion = nn.MSELoss(reduction="sum")

    opt = torch.optim.Adam(model.parameters())

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    for e in range(metadata["last_epoch"]+1, EPOCHS):
        model.train()
        with tqdm(total=len(train_dataloader)*local_batch_size,desc=f"Epoch {e+1} Training Loss: NaN") as pbar:
            for i, (coarse, dif) in enumerate(train_dataloader):
                coarse, dif = coarse.to(device), dif.to(device)
                pred = model.forward(coarse)
                loss = criterion.forward(pred, dif)
                loss.backward()
                if (i+1)%gradient_accumulation_steps==0:
                    opt.step()
                    opt.zero_grad()
                pbar.update(local_batch_size)
                pbar.set_description(f"Epoch {e+1} Loss: {loss.item():.8f}")
        logger.log(f"Train Loss at Epoch {e+1}: {loss}")

        model.eval()
        total_loss = 0
        with tqdm(total=len(validation_dataloader)*local_batch_size,desc=f"Epoch {e+1} Validation Loss: NaN") as pbar:
            for coarse, dif in validation_dataloader:
                coarse, dif = coarse.to(device), dif.to(device)
                pred = model.forward(coarse)
                loss = val_criterion.forward(pred, dif)
                total_loss += loss.item()

                pbar.update(local_batch_size)
                pbar.set_description(f"Epoch {e+1} Validation Loss: {loss.item():.8f}")
        logger.log(f"Validation Loss at Epoch {e+1}: {total_loss/(len(validation_dataloader)*local_batch_size)}")

        save_model(model, model_type, checkpoint_path, model_config, {"last_epoch:":e, "model_config":model_config})    

        

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train: [MLP, CNN, KAN, Transformer]")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/general.log", help="path to log file")
    ap.add_argument("--new_run", action="store_true")
    main(**ap.parse_args().__dict__)