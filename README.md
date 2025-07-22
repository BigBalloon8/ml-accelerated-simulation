# ml-accelerated-simulation

This is a repository that implements our updated version of learned correction


## Data Generation

First setup download the credentials from google drive to use CommandLineAuth in pydrive.

Enable the Drive API & create OAuth credentials

1. Go to [Google Cloud Console → APIs & Services → Library], enable Google Drive API. 

2. Then navigate to APIs & Services → Credentials → Create Credentials → OAuth client ID. Choose Web application (or Desktop), set http://localhost:8080/ as redirect URI. 

3. Download and rename your JSON file

4. After creation, download the file, which will have a name like client_secret_<long_id>.json.

5. Rename it exactly to client_secrets.json. 

6. Place it in the correct location. Copy client_secrets.json into the directory where the run your script (quickstart.py or similar). PyDrive2 looks there by default

Create a settings.yaml in your working directory:

```yaml
client_config_backend: file
client_config_file: /full/path/to/client_secrets.json

save_credentials: true
save_credentials_backend: file
save_credentials_file: /full/path/to/credentials.json

get_refresh_token: true

oauth_scope:
  - https://www.googleapis.com/auth/drive.file
```
finally run the data generation script

```bash
cd src/
python data_generation.py
```

## Train Segmentation Model
```bash 
cd ml-accelerated-simulation/src 
python segmentation_train.py --data_path ../../dataset --model_type segment_big \ 
                             --model_config models/configs/fullmodels/segnet.json\
                             --checkpoint_path /content/drive/MyDrive/checkpoints\
                             --log_file /content/drive/MyDrive/logs/segmentation.log
```

## Train LC Model
This is down with `train_v2.py`
```bash
python train_v2.py --help
usage: train_v2.py [-h] [--data_path DATA_PATH] [--model_type MODEL_TYPE] [--model_config MODEL_CONFIG] [--checkpoint_path CHECKPOINT_PATH] [--log_file LOG_FILE] [--new_run] [--seg_model_name SEG_MODEL_NAME]
                   [--seg_model_config SEG_MODEL_CONFIG]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        The path of the training data
  --model_type MODEL_TYPE
                        Model to train
  --model_config MODEL_CONFIG
                        path to model config
  --checkpoint_path CHECKPOINT_PATH
                        path to model config
  --log_file LOG_FILE   path to log file
  --new_run
  --seg_model_name SEG_MODEL_NAME
                        segment model name
  --seg_model_config SEG_MODEL_CONFIG
                        segment model config
```
Example of training baseline model
```bash
cd ml-accelerated-simulation/src 
python train_v2.py --data_path ../../dataset --model_type BASELINE \
                   --model_config models/configs/fullmodels/baselines/MLACFD.json \
                   --checkpoint_path /content/drive/MyDrive/checkpoints \ 
                   --log_file /content/drive/MyDrive/logs/baseline.log
```

Example of training U-Net
```
cd ml-accelerated-simulation/src 
python train_v2.py --data_path ../../dataset --model_type UNET \
                   --model_config models/configs/fullmodels/uNet1.json \
                   --checkpoint_path /content/drive/MyDrive/checkpoints \ 
                   --log_file /content/drive/MyDrive/logs/unet.log
```

## Inference Testing
```bash
python inference_testing.py --model_type BASELINE \
                   --model_config models/configs/fullmodels/baselines/MLACFD.json \
                   --checkpoint_path /content/drive/MyDrive/checkpoints \ 
                   --log_file /content/drive/MyDrive/logs/baseline.log
```

### Using Traditional LC

To train without segmentation use `train.py` instead of `train_v2.py`. To test at inference without segmentation pass the `--no_segment` to `inference_testing.py`.

# Model Configs

Model Configs are defined as following
```json
[
    {... Model Block 1 ...},
    {... Model Block 2...},
    ...,
    {... Model Block N}
]
```
Model configs are a list of predefined modules. To add custom modules add your block implementation to `tools.py/getModel`

