import torch
import torch.nn as nn

from ray import tune
import ray.tune.schedulers as schedulers
from ray.tune.search.bayesopt import BayesOptSearch

from functools import partial
from tqdm import tqdm
import safetensors.torch as st

import argparse
import json
import os
from typing import Tuple, Any
import random
import hashlib
import shutil

from data.dataloader import get_kolomogrov_flow_data_loader
from models import buildModel
from log import Logger
from inference_testing_tuning import main as infer

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

def mask_from_classes(x:torch.Tensor, classes):
    binary_masks = [(x>classes[i]).to(torch.int64) for i in range(len(classes))]
    return torch.sum(torch.stack(binary_masks), dim=0)

def get_percentiles(num_tiles:10|25|100|1000=1000):
    with open("data/data_percentiles.json", "r") as f:
        data = json.load(f)
    return data[f"percentages_{num_tiles}"]

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

def load_metadata(name:str, checkpoint_path, new_run, cl):
    if f"{name}_{cl}_{hash_dict(name)}.json" in os.listdir(checkpoint_path) and not new_run:
        print(f"Metadata Found in {checkpoint_path}: {name}_{cl}_{hash_dict(name)}.json")
        with open(os.path.join(checkpoint_path, f"{name}_{cl}_{hash_dict(name)}.json"), "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"last_epoch":0, "metrics":{}, "percentile":cl, "best_loss":100}
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
    
    if f"{name}_{metadata['percentile']}_{hash_dict(config)}.safetensors" in os.listdir(checkpoint_path) and not new_run:
        print(f"Model Found in {checkpoint_path}: {name}_{metadata['percentile']}_{hash_dict(config)}.safetensors")
        model_path = os.path.join(checkpoint_path, f"{name}_{metadata['percentile']}_{hash_dict(config)}.safetensors")
        opt_path = os.path.join(checkpoint_path, f"{name}_{metadata['percentile']}_ADAM_{hash_dict(config)}.pt")
        model_weights = st.load_file(model_path)
        model_base.load_state_dict(model_weights)
        opt_state = torch.load(opt_path)
    else:
        opt_state = None
    return model_base, opt_state
        
def save_model(model:nn.Module, opt:torch.optim.Optimizer, model_type, checkpoint_path, model_config, metadata, groups=1):
    with open(model_config, "r") as f:
        config = json.load(f)
    if groups > 1:
        for i in config:
            i["group"] = groups
            i["structures"]["in_channels"] *= groups
            i["structures"]["out_channels"] *= groups
            i["structures"]["hidden_channels"] = [groups*j for j in i["structures"]["hidden_channels"]]

    metadata["model_config"] = config
    model_path = os.path.join(checkpoint_path, f"{model_type}_{metadata['percentile']}_{hash_dict(config)}.safetensors")
    opt_path = os.path.join(checkpoint_path, f"{model_type}_{metadata['percentile']}_ADAM_{hash_dict(config)}.pt")
    with open(os.path.join(checkpoint_path, f"{model_type}_{metadata['percentile']}_{hash_dict(model_type)}.json"), "w") as f:
        json.dump(metadata, f)
    st.save_model(model, model_path)
    torch.save(opt.state_dict(), opt_path)
    if "overview.json" in os.listdir(checkpoint_path):
        print("Overview file found")
        with open(os.path.join(checkpoint_path, f"overview.json"), "r") as f:
            overview = json.load(f)
        overview[f'{metadata["percentile"]}'] = metadata["best_loss"]
    else:
        overview = {f'{metadata["percentile"]}': metadata["best_loss"]}
    with open(os.path.join(checkpoint_path, f"overview.json"), "w") as f:
        json.dump(overview, f)
    tune_checkpoint = tune.Checkpoint.from_directory(checkpoint_path)
    tune.report(metadata["metrics"], checkpoint=tune_checkpoint)
    #print(opt.state_dict())

def get_mask(x, segment_model, classes:list):
    with torch.no_grad():
        out = mask_from_classes(segment_model(x), classes)
        return [out == i for i in range(1, len(classes) + 1)]

    
def train_model(config:dict, data_path, model_type, model_config, checkpoint_path, new_run, seg_model_name, seg_model_config, epochs:int, logger:Logger):
    #config["cl"] = round(config["cl"], 3)
    batchsize = 32
    gradient_accumulation_steps = 1
    local_batch_size = batchsize // gradient_accumulation_steps

    metadata = load_metadata(model_type, checkpoint_path, new_run, config["cl"])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seg_model = get_segment_model(seg_model_name, seg_model_config, checkpoint_path)
    seg_model.to(device)

    mask_stream = torch.cuda.Stream()
    model_stream = torch.cuda.Stream()

    train_dataloader = get_kolomogrov_flow_data_loader(data_path, batchsize=local_batch_size, train_only=True)

    model, opt_state = get_model(model_type, model_config, checkpoint_path, logger, new_run, metadata) #fix grouping
    model = model.to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    kwargs = {"model":model, "seg_model":seg_model, "device":device, "mask_stream":mask_stream, "model_stream":model_stream, "logger":logger}

    logger.log(f"Percentile: {round(config['cl'], 3)}")
    for e in range(metadata["last_epoch"], epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_dataloader)*local_batch_size,desc=f"Epoch {e+1} Training Loss: NaN") as pbar:
            for coarse, dif in train_dataloader:
                coarse, dif = coarse.to(device), dif.to(device)
                with torch.cuda.stream(mask_stream):
                    mask = get_mask(coarse, seg_model, [config["cl"]])
                with torch.cuda.stream(model_stream):
                    pred = model.forward(coarse.repeat(1, 1, 1, 1)) #fix grouping (1, group-1, 1, 1)
                torch.cuda.synchronize()
                pred = torch.cat([torch.masked_select(pred[:,2*i:2*(i+1)], mask[i]) for i in range(len(mask))], dim=0)
                dif = torch.cat([torch.masked_select(dif, mask[i]) for i in range(len(mask))], dim=0)
                loss = criterion.forward(pred, dif)
                loss.backward()
                total_loss += loss.item()
                opt.step()
                opt.zero_grad()
                pbar.update(local_batch_size)
                pbar.set_description(f"Epoch {e+1} Loss: {loss.item():.8f}")
        logger.log(f"Train Loss at Epoch {e+1}: {total_loss/(len(train_dataloader))}")

        model.eval()
        with torch.inference_mode():
            _, lc_error = infer(config, **kwargs)
        logger.log(f"LC Error at Epoch {e+1}: {lc_error}")
        scheduler.step(lc_error*local_batch_size)
        metadata["metrics"]["loss"], metadata["best_loss"] = lc_error, min(metadata["best_loss"], lc_error)
        save_model(model, opt, model_type, checkpoint_path, model_config, metadata) #fix grouping


def main(data_path, model_type, model_config, checkpoint_path, log_file, new_run, seg_model_name, seg_model_config, num_samples):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    random.seed(2025)
    
    logger = Logger(model_type, log_file)

    kwargs = {"data_path":data_path, "model_type":model_type, "model_config":model_config, "checkpoint_path":checkpoint_path, "new_run":new_run, "seg_model_name":seg_model_name, "seg_model_config":seg_model_config, "epochs":100, "logger":logger}
    config = {
        "cl":tune.quniform(0.001, 0.999, 0.001),        
    }
    #search = BayesOptSearch()#random_state=2025, random_search_steps=4, patience=5, points_to_evaluate=[{"cl": 0.667}]
    scheduler = schedulers.ASHAScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=kwargs["epochs"],
        grace_period=5,
        reduction_factor=2,
    )
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(partial(train_model, **kwargs)), resources={"gpu":1}),
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=num_samples),#search_alg=search, metric="loss", mode="min", 
        run_config=tune.RunConfig(name=model_type, storage_path="/tmp/ray_results"),
        param_space=config,
    )

    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    logger.log(f"Best trial config: {best_result.config}")
    logger.log(f"Best trial final LC Error: {best_result.metrics['loss']}")

    print("Copying results from /tmp/ray_results to Google Drive...")
    source_dir = "/tmp/ray_results"
    #destination_dir = "/content/drive/MyDrive/checkpoints/tuning2_2"
    shutil.copytree(source_dir, checkpoint_path)
    print("Copying complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--data_path", default="../data/data.safetensors", help="The path of the training data")
    ap.add_argument("--model_type", default="CNN", help="Model to train")
    ap.add_argument("--model_config", default="./model.config", help="path to model config")
    ap.add_argument("--checkpoint_path", default="/content/drive/MyDrive/checkpoints/tuning2_2", help="path to model config")
    ap.add_argument("--log_file", default="/content/drive/MyDrive/logs/cutoff_tuning.log", help="path to log file")
    ap.add_argument("--new_run", action="store_true")
    ap.add_argument("--seg_model_name", default="seg_v3_10layers", help="segment model name") 
    ap.add_argument("--seg_model_config", default="/content/ml-accelerated-simulation/src/models/configs/fullmodels/segnet_v3.json", help="segment model config")
    ap.add_argument("--num_samples", type=int, default=50, help="number of samples to run")
    main(**ap.parse_args().__dict__)