import torch
import torch.nn as nn

from tqdm import tqdm
import safetensors.torch as st

import argparse
import json
import os
from typing import Tuple, Any
import random
import hashlib

from data.dataloader import k_fold_split, get_k_fold_data_loader
from models import buildModel
from log import Logger

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

#def get_classes(classes:int):
#    cl_width = 1/classes
#    return [cl_width*j for j in range(1, int(classes))]

def mask_from_classes(x:torch.Tensor, classes):
    binary_masks = [(x>classes[i]).to(torch.int64) for i in range(len(classes))]
    return torch.sum(torch.stack(binary_masks), dim=0)



def get_percentiles(num_tiles:10|25|100|1000=10):
    with open("data/data_percentiles.json", "r") as f:
        data = json.load(f)
    return torch.tensor([0.0] + data[f"percentages_{num_tiles}"]) 

def get_segment_model(name, config_file, checkpoint_path):
    with open(config_file, "r") as f:
        config = json.load(f)
    model_base = buildModel(config)

    try:
        model_path = os.path.join(os.path.dirname(checkpoint_path), f"{name}_{hash_dict(config)}_seg.safetensors")
        model_weights = st.load_file(model_path)
    except(FileNotFoundError):
        model_path = os.path.join(checkpoint_path, f"{name}_{hash_dict(config)}_seg.safetensors")
        model_weights = st.load_file(model_path)
    model_base.load_state_dict(model_weights)
    return model_base

def load_metadata(name:str, checkpoint_path, new_run):
    if f"{name}_{hash_dict(name)}.json" in os.listdir(checkpoint_path) and not new_run:
        print(f"Metadata Found in {checkpoint_path}: {name}_{hash_dict(name)}.json")
        with open(os.path.join(checkpoint_path, f"{name}_{hash_dict(name)}.json"), "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"last_m":-1, "last_k":-1, "last_epoch":-1, "last_loss":0, "best_loss":1e4, "m_losses":[]}
    return metadata

def get_model(name:str, config_file, checkpoint_path, logger, new_run, metadata, groups=1)-> Tuple[nn.Module, Any]:
    with open(config_file, "r") as f:
        config = json.load(f)
    if groups > 1:
        for i in config:
            i["group"] = groups
            i["structures"]["in_channels"] *= groups
            i["structures"]["out_channels"] *= groups
            i["structures"]["hidden_channels"] = [groups*j for j in i["structures"]["hidden_channels"]]
    logger.log(f"Model Config: {config}")

    model_base = buildModel(config)
    
    if f"{name}_{metadata['last_m']}_{metadata['last_k']}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path) and not new_run:
        print(f"Model Found in {checkpoint_path}: {name}_{metadata['last_m']}_{metadata['last_k']}_{hash_dict(config)}.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{metadata['last_m']}_{metadata['last_k']}_{hash_dict(config)}.safetensors")
        opt_path = os.path.join(checkpoint_path, f"{name}_{metadata['last_m']}_{metadata['last_k']}_ADAM_{hash_dict(config)}.pt")
        model_weights = st.load_file(model_path)
        model_base.load_state_dict(model_weights)
        opt_state = torch.load(opt_path)
    else:
        opt_state = None
    return model_base, opt_state
        
def save_model(model:nn.Module, opt:torch.optim.Optimizer, model_type, checkpoint_path, model_config, metadata=None, groups=1):
    with open(model_config, "r") as f:
        config = json.load(f)
    if groups > 1:
        for i in config:
            i["group"] = groups
            i["structures"]["in_channels"] *= groups
            i["structures"]["out_channels"] *= groups
            i["structures"]["hidden_channels"] = [groups*j for j in i["structures"]["hidden_channels"]]

    metadata["model_config"] = config
    model_path = os.path.join(checkpoint_path, f"{model_type}_{metadata['last_m']}_{metadata['last_k']}_{hash_dict(config)}.safetensors")
    opt_path = os.path.join(checkpoint_path, f"{model_type}_{metadata['last_m']}_{metadata['last_k']}_ADAM_{hash_dict(config)}.pt")
    with open(os.path.join(checkpoint_path, f"{model_type}_{hash_dict(model_type)}.json"), "w") as f:
        json.dump(metadata, f)
    st.save_model(model, model_path)
    #print(opt.state_dict())
    torch.save(opt.state_dict(), opt_path)


def get_mask(x, segment_model, classes:list):
    with torch.no_grad():
        #classes = get_classes(num_classes)
        out = mask_from_classes(segment_model(x), classes)
        return [out == i for i in range(1, len(classes) + 1)]

    
def main(data_path, model_type, model_config, checkpoint_path, log_file, new_run, seg_model_name, seg_model_config, num_classes:int, K:int):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    random.seed(2025)

    logger = Logger(model_type, log_file)

    EPOCHS = 130
    batchsize = 32
    gradient_accumulation_steps = 1
    local_batch_size = batchsize // gradient_accumulation_steps
    lambda_m = get_percentiles(10)
    criterion = nn.MSELoss()

    ds = k_fold_split(data_path, K=K)
    metadata = load_metadata(model_type, checkpoint_path, new_run)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seg_model = get_segment_model(seg_model_name, seg_model_config, checkpoint_path)
    seg_model.to(device)

    mask_stream = torch.cuda.Stream()
    model_stream = torch.cuda.Stream()


    for m in range(metadata["last_m"]+1, len(lambda_m)):
        for k in range(metadata["last_k"]+1, K):
            train_dataloader, validation_dataloader = get_k_fold_data_loader(ds, k, batchsize=local_batch_size)

            model, opt_state = get_model(model_type, model_config, checkpoint_path, logger, new_run, metadata, num_classes-1)
            model = model.to(device)

            opt = torch.optim.Adam(model.parameters())
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

            for e in range(metadata["last_epoch"]+1, EPOCHS):
                model.train()
                total_loss = 0
                with tqdm(total=len(train_dataloader)*local_batch_size,desc=f"Epoch {e+1} Training Loss: NaN") as pbar:
                    for i, (coarse, dif) in enumerate(train_dataloader):
                        coarse, dif = coarse.to(device), dif.to(device)
                        with torch.cuda.stream(mask_stream):
                            mask = get_mask(coarse, seg_model, [lambda_m[m]])
                        with torch.cuda.stream(model_stream):
                            pred = model.forward(coarse.repeat(1, num_classes-1, 1, 1))
                        torch.cuda.synchronize()
                        pred = torch.cat([torch.masked_select(pred[:,2*i:2*(i+1)], mask[i]) for i in range(len(mask))], dim=0)
                        dif = torch.cat([torch.masked_select(dif, mask[i]) for i in range(len(mask))], dim=0)
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
                                mask = get_mask(coarse, seg_model, [lambda_m[m]])
                            with torch.cuda.stream(model_stream):
                                pred = model.forward(coarse.repeat(1, num_classes-1, 1, 1))
                            torch.cuda.synchronize()
                            pred = torch.cat([torch.masked_select(pred[:,2*i:2*(i+1)], mask[i]) for i in range(len(mask))], dim=0)
                            dif = torch.cat([torch.masked_select(dif, mask[i]) for i in range(len(mask))], dim=0)
                            loss = criterion.forward(pred, dif)
                            total_loss += loss.item()

                            pbar.update(local_batch_size)
                            pbar.set_description(f"Epoch {e+1} Validation Loss: {loss.item():.8f}")
                logger.log(f"Validation Loss at Epoch {e+1}: {total_loss/(len(validation_dataloader))}")
                scheduler.step(total_loss/(len(validation_dataloader)*local_batch_size))
                metadata["best_loss"], metadata["last_epoch"] = min(metadata["best_loss"], total_loss/(len(validation_dataloader)*local_batch_size)), e
                save_model(model, opt, model_type, checkpoint_path, model_config, metadata, num_classes-1)
            metadata["last_loss"], metadata["best_loss"], metadata["last_k"], metadata["last_epoch"] = metadata["last_loss"] + metadata["best_loss"], 1e4, k, -1
            save_model(model, opt, model_type, checkpoint_path, model_config, metadata, num_classes-1)
        metadata["m_losses"].append(metadata["last_loss"])
        metadata["last_loss"], metadata["last_m"], metadata["last_k"] = 0, m, -1
        save_model(model, opt, model_type, checkpoint_path, model_config, metadata, num_classes-1)
    logger.log(f"Cross Validation Loss for m={lambda_m} are {metadata['m_losses']}")

        

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default="/content/drive/MyDrive/checkpoints/validation", help="path to model config") #change default to checkpoints/validation
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/general.log", help="path to log file")
    ap.add_argument("--new_run", action="store_true")
    ap.add_argument("--seg_model_name", default="seg_v3_10layers", help="segment model name") 
    ap.add_argument("--seg_model_config", default="models/configs/fullmodels/segnet_v3.json", help="segment model config")
    ap.add_argument("--num_classes", type=int, default=2, help="number of classes")
    ap.add_argument("--K", type=int, default=5, help="number of folds")
    main(**ap.parse_args().__dict__)