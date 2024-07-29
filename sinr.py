"""
API for connecting web app with models.
To generate predictions call the method `generate_prediction`.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from . import utils, models, datasets

sys.path.append('../')


def get_cur_path():
    return str(Path(__file__).parent)


def load_taxa_metadata(file_path='web_app/taxa_02_08_2023_names.txt'):
    cur_path = get_cur_path()
    taxa_names_file = open(cur_path + file_path, "r")
    data = taxa_names_file.read().split("\n")
    data = [dd for dd in data if dd != '']
    taxa_ids = []
    taxa_names = []
    for tt in range(len(data)):
        id, nm = data[tt].split('\t')
        taxa_ids.append(int(id))
        taxa_names.append(nm)
    taxa_names_file.close()
    return dict(zip(taxa_names, taxa_ids))
 

def get_paths(path='/paths.json'):
    cur_path = get_cur_path()
    with open(cur_path + path, 'r') as f:
        return json.load(f)


def load_model(eval_params):
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    return train_params, model


def get_model_path(selected_model):
    cur_path = get_cur_path()
    if selected_model == 'AN_FULL_max_10':
        model_path = f'{cur_path}/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt'
    elif selected_model == 'AN_FULL_max_100':
        model_path = f'{cur_path}/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt'
    elif selected_model == 'AN_FULL_max_1000':
        model_path = f'{cur_path}/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
    elif selected_model == 'Distilled_env_model':
        model_path = f'{cur_path}/pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt'
    elif selected_model == 'fine-tuned':
        model_path = f'{cur_path}/pretrained_models/test_fine_tune.pt'
    return model_path


def generate_prediction(eval_params):
    eval_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_params['model_path'] = get_model_path(eval_params['model'])

    # load paths
    paths = get_paths()

    # load model
    train_params, model = load_model(eval_params)

    model.eval()

    # create input encoder
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env(norm=train_params['params']['env_norm'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    # load taxa of interest 
    class_of_interest = train_params['params']['class_to_taxa'].index(eval_params['taxa_id'])

    print(f'Loading taxa: {eval_params["taxa_id"]}')

    # load ocean mask
    cur_path = get_cur_path()
    mask = np.load(os.path.join(cur_path, paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    # generate input features
    locs = utils.coord_grid(mask.shape)

    # if need to disable ocean, remove all ocean coordinats
    if not eval_params['disable_ocean_mask']:
        locs = locs[mask_inds, :]

    ret_locs = locs.copy()

    locs = torch.from_numpy(locs)
    locs_enc = enc.encode(locs).to(eval_params['device'])

    # make prediction
    with torch.no_grad():
        preds = model(locs_enc, return_feats=False, class_of_interest=class_of_interest).cpu().numpy()

    return preds, ret_locs