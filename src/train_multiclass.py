import torch
import torch.nn as nn

from tqdm import tqdm
import safetensors.torch as st

import argparse
import json
import os
from typing import Tuple
import random
import hashlib

from data.dataloader import get_kolomogrov_flow_data_loader
from models import buildModel
from log import Logger


def grouping(config):
    config[st]

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()


def get_model(name:str, config_file, checkpoint_path, logger, new_run, grouping)-> Tuple[nn.Module, dict]:
    with open(config_file, "r") as f:
        config = json.load(f)
    logger.log(f"Model Config: {config}")

    model_base = buildModel(config)
    
    if f"{name}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path) and not new_run:
        print(f"Model Found in {checkpoint_path}: {name}_{hash_dict(config)}.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.safetensors")
        opt_path = os.path.join(checkpoint_path, f"{name}_ADAM_{hash_dict(config)}.pt")
        model_weights = st.load_file(model_path)
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}.json"), "r") as f:
            metadata = json.load(f)
        model_base.load_state_dict(model_weights)
        opt_state = torch.load(opt_path)
    else:
        metadata = {"last_epoch":-1}
        opt_state = None
    return model_base, metadata, opt_state

def get_segment_model(name, config_file, checkpoint_path):
    with open(config_file, "r") as f:
        config = json.load(f)

    num_classes = config[-1]["structures"]["out_channels"]//config[-1]["group"]
    model_base = buildModel(config)
    model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}_seg.safetensors")
    model_weights = st.load_file(model_path)
    model_base.load_state_dict(model_weights)
    return model_base, num_classes
        
def save_model(model:nn.Module, opt:torch.optim.Optimizer, model_type, checkpoint_path, model_config, metadata=None):
    with open(model_config, "r") as f:
        config = json.load(f)
    metadata["model_config"] = config
    model_path = os.path.join(checkpoint_path, f"{model_type}_{hash_dict(config)}.safetensors")
    opt_path = os.path.join(checkpoint_path, f"{model_type}_ADAM_{hash_dict(config)}.pt")
    with open(os.path.join(checkpoint_path, f"{model_type}_{hash_dict(config)}.json"), "w") as f:
        json.dump(metadata, f)
    st.save_model(model, model_path)
    #print(opt.state_dict())
    torch.save(opt.state_dict(), opt_path)


def get_mask(x, segment_model, num_classes):
    with torch.no_grad():
        logits = torch.softmax(segment_model(x), dim=1)
        out = torch.argmax(logits, dim=1, keepdim=True)
        return [out == i for i in range(1, num_classes)]
    
def main(data_path, model_type, model_config, checkpoint_path, log_file, new_run, seg_model_name, seg_model_config):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    random.seed(2025)

    logger = Logger(model_type, log_file)

    EPOCHS = 500
    batchsize = 32
    gradient_accumulation_steps = 1
    local_batch_size = batchsize // gradient_accumulation_steps

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataloader, validation_dataloader = get_kolomogrov_flow_data_loader(data_path, batchsize=local_batch_size)

    model, metadata, opt_state = get_model(model_type, model_config, checkpoint_path, logger, new_run)
    model = model.to(device)

    seg_model, num_classes = get_segment_model(seg_model_name, seg_model_config, checkpoint_path)
    seg_model.to(device)

    criterion = nn.MSELoss()

    opt = torch.optim.Adam(model.parameters())
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    mask_stream = torch.cuda.Stream()
    model_stream = torch.cuda.Stream()

    for e in range(metadata["last_epoch"]+1, EPOCHS):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_dataloader)*local_batch_size,desc=f"Epoch {e+1} Training Loss: NaN") as pbar:
            for i, (coarse, dif) in enumerate(train_dataloader):
                coarse, dif = coarse.to(device), dif.to(device)
                with torch.cuda.stream(mask_stream):
                    mask = get_mask(coarse, seg_model, num_classes)
                with torch.cuda.stream(model_stream):
                    pred = model.forward(coarse)
                torch.cuda.synchronize()
                pred = torch.cat([torch.masked_select(pred[:,2*i:2*(i+1)], mask[i]) for i in range(num_classes-1)], dim=0)
                dif = torch.cat([torch.masked_select(dif[:,2*i:2*(i+1)], mask[i]) for i in range(num_classes-1)], dim=0)
                #pred = torch.masked_select(pred, mask)
                #dif = torch.masked_select(dif, mask)
                loss = criterion.forward(pred, dif)
                loss.backward()
                total_loss += loss.item()
                #if (i+1)%gradient_accumulation_steps==0:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                opt.zero_grad()
                pbar.update(local_batch_size)
                pbar.set_description(f"Epoch {e+1} Loss: {loss.item():.8f}")
        logger.log(f"Train Loss at Epoch {e+1}: {total_loss/(len(train_dataloader))}")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            with tqdm(total=len(validation_dataloader)*local_batch_size,desc=f"Epoch {e+1} Validation Loss: NaN") as pbar:
                for coarse, dif in validation_dataloader:
                    coarse, dif = coarse.to(device), dif.to(device)
                    with torch.cuda.stream(mask_stream):
                        mask = get_mask(coarse, seg_model, num_classes)
                    with torch.cuda.stream(model_stream):
                        pred = model.forward(coarse)
                    torch.cuda.synchronize()
                    pred = torch.cat([torch.masked_select(pred[:,2*i:2*(i+1)], mask[i]) for i in range(num_classes-1)], dim=0)
                    dif = torch.cat([torch.masked_select(dif[:,2*i:2*(i+1)], mask[i]) for i in range(num_classes-1)], dim=0)
                    #pred = torch.masked_select(pred, mask)
                    #dif = torch.masked_select(dif, mask)
                    loss = criterion.forward(pred, dif)
                    total_loss += loss.item()

                    pbar.update(local_batch_size)
                    pbar.set_description(f"Epoch {e+1} Validation Loss: {loss.item():.8f}")
        logger.log(f"Validation Loss at Epoch {e+1}: {total_loss/(len(validation_dataloader))}")
        scheduler.step(total_loss/(len(validation_dataloader)*local_batch_size))

        save_model(model, opt, model_type, checkpoint_path, model_config, {"last_epoch":e})    

        

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default=".", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/general.log", help="path to log file")
    ap.add_argument("--new_run", action="store_true")
    ap.add_argument("--seg_model_name", default="segment_big", help="segment model name")
    ap.add_argument("--seg_model_config", default="models/configs/fullmodels/segnet.json", help="segment model config")
    main(**ap.parse_args().__dict__)