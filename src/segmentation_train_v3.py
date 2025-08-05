import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import safetensors.torch as st

import argparse
import json
import os
from typing import Tuple
import random
import hashlib
import numpy as np

from data.dataloader import get_kolomogrov_flow_data_loader
from models import buildModel
from log import Logger


def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

def get_classes(classes:list):
    classes = np.array(classes)
    cl_width = 1/classes
    return np.array([[cl_width[i]*j for j in range(1, classes[i])] for i in range(len(classes))])

def mask_from_classes(x, classes):
    binary_masks = [(torch.norm(x, dim=1)>classes[i]).to(torch.int64) for i in range(len(classes))]
    return torch.sum(torch.stack(binary_masks), dim=0)


def get_model(name:str, config_file, checkpoint_path, logger, new_run)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)
    logger.log(f"Model Config: {config}")

    model_base = buildModel(config)
    
    if f"{name}_{hash_dict(config)}_seg.safetensors" in os.listdir(checkpoint_path) and not new_run:
        print(f"Model Found in {checkpoint_path}: {name}_{hash_dict(config)}_seg.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}_seg.safetensors")
        opt_path = os.path.join(checkpoint_path, f"{name}_ADAM_{hash_dict(config)}_seg.pt")
        model_weights = st.load_file(model_path)
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}_seg.json"), "r") as f:
            metadata = json.load(f)
        model_base.load_state_dict(model_weights)
        opt_state = torch.load(opt_path)
    else:
        metadata = {"last_epoch":-1}
        opt_state = None
    return model_base, metadata, opt_state
    
        
def save_model(model:nn.Module, opt:torch.optim.Optimizer, model_type, checkpoint_path, model_config, metadata=None):
    with open(model_config, "r") as f:
        config = json.load(f)
    
    metadata["model_config"] = config
    model_path = os.path.join(checkpoint_path, f"{model_type}_{hash_dict(config)}_seg.safetensors")
    opt_path = os.path.join(checkpoint_path, f"{model_type}_ADAM_{hash_dict(config)}_seg.pt")
    with open(os.path.join(checkpoint_path, f"{model_type}_{hash_dict(config)}_seg.json"), "w") as f:
        json.dump(metadata, f)
    st.save_model(model, model_path)
    #print(opt.state_dict())
    torch.save(opt.state_dict(), opt_path)
    

def get_dif_label(dif):
    with open("data/data_percentiles.json", "r") as f:
        data = json.load(f)
    percentiles, vals = data["percentages"], data["values"]

    dif_norm = torch.norm(dif, dim=1)
    labels = torch.zeros_like(dif_norm)

    for i in range(len(percentiles)):
        if i < len(percentiles)-1:
            labels[(dif_norm>vals[i]) & (dif_norm<vals[i+1])] = percentiles[i]
    labels[dif_norm>percentiles[i]] = percentiles[i]
    return labels


def class_accuracy(logits, dif_labels, classes=list(range(2, 6))):
    percentiles = get_classes(classes)
    acc = np.array([])
    for item in percentiles:
        logit = mask_from_classes(logits, item)
        dif_label = mask_from_classes(dif_labels, item)
        np.append(acc, (torch.argmax(logit, dim=1)==dif_label).sum()/torch.numel(dif_label))
    return acc

 
def main(data_path, model_type, model_config, checkpoint_path, log_file, new_run):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    random.seed(2025)

    logger = Logger(model_type, log_file)

    EPOCHS = 140
    batchsize = 32
    gradient_accumulation_steps = 1
    local_batch_size = batchsize // gradient_accumulation_steps

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataloader, validation_dataloader = get_kolomogrov_flow_data_loader(data_path, batchsize=local_batch_size)

    model, metadata, opt_state = get_model(model_type, model_config, checkpoint_path, logger, new_run)
    model = model.to(device)

    criterion = nn.MSELoss()

    opt = torch.optim.Adam(model.parameters())
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    for e in range(metadata["last_epoch"]+1, EPOCHS):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_dataloader)*local_batch_size,desc=f"Epoch {e+1} Training Loss: NaN") as pbar:
            for i, (coarse, dif) in enumerate(train_dataloader):
                coarse, dif = coarse.to(device), dif.to(device)
                dif_labels = get_dif_label(dif)
                logits = model.forward(coarse)
                loss = criterion(logits, dif_labels)
                loss.backward()
                total_loss += loss.item()
                #if (i+1)%gradient_accumulation_steps==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                opt.zero_grad()
                pbar.update(local_batch_size)
                pbar.set_description(f"Epoch {e+1} Loss: {loss.item():.8f}")
        logger.log(f"Train Loss at Epoch {e+1}: {total_loss/(len(train_dataloader))}")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = np.zeros(4)
            with tqdm(total=len(validation_dataloader)*local_batch_size,desc=f"Epoch {e+1} Validation Loss: NaN") as pbar:
                for coarse, dif in validation_dataloader:
                    coarse, dif = coarse.to(device), dif.to(device)
                    dif_labels = get_dif_label(dif)
                    logits = model.forward(coarse)
                    loss = criterion(logits, dif_labels)
                    total_loss += loss.item()
                    total_acc += class_accuracy(logits, dif_labels)
                    pbar.update(local_batch_size)
                    pbar.set_description(f"Epoch {e+1} Validation Loss: {loss.item():.8f}")
        logger.log(f"Validation Loss at Epoch {e+1}: {total_loss/(len(validation_dataloader))}")
        logger.log(f"Validation Accuracy at Epoch {e+1}: {(total_acc*100)/(len(validation_dataloader)):.4f}%")
        scheduler.step(total_loss/(len(validation_dataloader)*local_batch_size))

        save_model(model, opt, model_type, checkpoint_path, model_config, {"last_epoch":e})    

        

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train: [MLP, CNN, KAN, Transformer]")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/general.log", help="path to log file")
    ap.add_argument("--new_run", action="store_true")
    main(**ap.parse_args().__dict__)