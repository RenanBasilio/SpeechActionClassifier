# This is the default configuration file for the script train.py
# In order to change this script, copy it and save it as "envconfig.py"

dataset_directory = "./data"
export_directory = "./export"

train_params = {
    'batch_size': 64,
    'shuffle': True,
    'max_workers': 8
}

val_params = {
    'batch_size': 64,
    'shuffle': True,
    'max_workers': 8
}