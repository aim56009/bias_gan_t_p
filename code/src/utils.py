from uuid import uuid1
from datetime import datetime
import os
import time
import pandas as pd
from IPython.display import display, HTML

import pytorch_lightning as pl
import numpy as np
import matplotlib as mpl
from pathlib import Path
import torch





def make_dict(path: str, data: dict):
    dir_names = path.split('/')[7:]
    if  len(dir_names) > 2:
        creation_time = os.path.getctime(path)
        creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
        model = dir_names[0]
        uuid = dir_names[2]
        
        data['Creation Date'].append(creation_time)
        data['Execution Date'].append(dir_names[1])
        data['UUID'].append(uuid)
        data['Path'].append(path)
        data['Model name'].append(model)
        
    return data


def show_checkpoints(path, model_name=None):
    paths = [x[0] for x in os.walk(path)]
    
    data =  {'Creation Date': [], 'Execution Date': [], 'UUID': [], 'Path': [], 'Model name': []}
    
    for path in paths:
        data = make_dict(path, data)
        
    df = pd.DataFrame(data=data)
    if model_name is not None:
        display(df.loc[df['Model name']==model_name].sort_values(by=['Creation Date'],ascending=False))
    else:
        display(df.sort_values(by=['Creation Date'],ascending=False))
    
    return df


def get_version(date,time):
    model_id = str(uuid1())
    version = f'{date}_{time}'

    return version


def get_checkpoint_path(config, version):

    model_name = config.model_name    
    checkpoint_path = config.checkpoint_path
    uuid_legth = 36
    date_legth = 10
    path = f'{config.checkpoint_path}{version}'

    return path


def save_config(config, version):
    import json
    
    uuid_legth = 36
    fname = f'{config.config_path}{version}/config_model.json'
    with open(fname, 'w') as file:
        file.write(json.dumps(vars(config))) 


def config_from_file(file_name):
    import json
    with open(file_name) as json_file:
        data = json.load(json_file)
    config = ClassFromDict(data)
    return config


def config_dict_from_file(file_name):
    import json
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data


class ClassFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
        setattr(self, 'flag', None)


def set_environment():
    pl.seed_everything(42)
    mpl.rcParams["axes.grid"     ] = False
    mpl.rcParams["figure.figsize"] = (8, 4)

def log_transform(x, epsilon):
    return np.log(x + epsilon) - np.log(epsilon)


def inv_log_transform(x, epsilon):
    return np.exp(x + np.log(epsilon)) - epsilon


def norm_transform(x, x_ref):
    return (x - x_ref.min())/(x_ref.max() - x_ref.min())


def inv_norm_transform(x, x_ref):
    return x * (x_ref.max() - x_ref.min()) + x_ref.min()


def norm_minus1_to_plus1_transform(x, x_ref):
    #print(f'norm_minus1_to_plus1 max:{x_ref.max()} min:{x_ref.min()}')
    results = (x - x_ref.min())/(x_ref.max() - x_ref.min())
    results = results*2 - 1
    return results


def inv_norm_minus1_to_plus1_transform(x, x_ref):
    #print(f'inv_norm_minus1_to_plus1 max:{x_ref.max()} min:{x_ref.min()}')
    x = (x + 1)/2
    results = x * (x_ref.max() - x_ref.min()) + x_ref.min()
    return results


def sqrt_transform(x):
    return torch.sqrt(x) 


def inv_sqrt_transform(x):
    return torch.sqrt(x) 
