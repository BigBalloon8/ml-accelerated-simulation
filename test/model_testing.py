import torch
import hashlib
import json
import numpy as np

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

#pt("BASELINE_4cl", "src/models/configs/fullmodels/baselines/MLACFD1.json", 3)

def get_classes(classes:list):
    classes = torch.tensor(classes)
    cl_width = 1/classes
    out = []
    for i in range(len(classes)):
        out.append(torch.tensor([cl_width[i]*j for j in range(1, int(classes[i]))]))  
    return out

def mask_from_classes(x, classes):
    binary_masks = [(torch.norm(x, dim=1)>classes[i]).to(torch.int64) for i in range(len(classes))]
    return torch.sum(torch.stack(binary_masks), dim=0)

def class_accuracy(logits, dif_labels, classes=list(range(2, 6))):
    percentiles = get_classes(classes)
    acc = np.array([])
    for item in percentiles:
        logit = mask_from_classes(logits, item)
        dif_label = mask_from_classes(dif_labels, item)
        np.append(acc, (torch.argmax(logit, dim=1)==dif_label).sum()/torch.numel(dif_label))
    return acc

print(get_classes([2,3,4]))