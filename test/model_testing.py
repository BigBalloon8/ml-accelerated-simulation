import torch
import hashlib
import json

def hash_dict(x:dict):
    formated_string = "".join(sorted(json.dumps(x, sort_keys=True)))
    return hashlib.sha1(formated_string.encode("utf‑8")).hexdigest()

def pt(name, config_file, groups):
    with open(config_file, "r") as f:
        config = json.load(f)
    
    if groups > 1:
        for i in config:
            i["group"] = groups
            i["structures"]["in_channels"] *= groups
            i["structures"]["out_channels"] *= groups
            i["structures"]["hidden_channels"] = [groups*j for j in i["structures"]["hidden_channels"]]
    print(config)
    print(f"{name}_{hash_dict(config)}.safetensors") #model_path =
    print(f"{name}_ADAM_{hash_dict(config)}.pt") #opt_path = 
    print(f"{name}_{hash_dict(config)}.json")

pt("BASELINE_4cl", "src/models/configs/fullmodels/baselines/MLACFD1.json", 3)