"""
This file contains additional code related to fine tuning, here you will find functions that load annotation data and starts running fine-tuning on the selected model.
"""

import pandas as pd
import numpy as np
import torch
import torch.utils
import torch.utils.data
import json
import os

import datasets
import losses
import setup
import models
import train

def load_annotation_data(annotation_file, taxa_of_interest):
    """
    load_inat_data copy for loading annotation data extracted from the database.

    Parameters:
        - annotation_file: the filename for the csv that has the annotations wanted to fine tune
        - taxa_of_interest: list taxa that you want to filter annotations by
    
    Returns:
        - locs: location tuples of longitude and latitude
        - taxa: list of taxa values in data
        - hex_types: denotes absence or presence in a corresponding location
    """

    print('\nLoading ' + annotation_file)
    data = pd.read_csv(annotation_file, index_col=0)

    # remove outliers
    num_obs = data.shape[0]
    data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
    if (num_obs - data.shape[0]) > 0:
        print(num_obs - data.shape[0], 'items filtered due to invalid locations')

    num_obs_orig = data.shape[0]
    data = data.dropna()
    size_diff = num_obs_orig - data.shape[0]
    if size_diff > 0:
        print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

    # keep only taxa of interest:
    if taxa_of_interest is not None:
        num_obs_orig = data.shape[0]
        data = data[data['taxon_id'].isin(taxa_of_interest)]
        print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

    print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

    locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
    taxa = data['taxon_id'].values.astype(np.int64)
    hex_types = data['hex_type'].values.astype(np.int64)

    return locs, taxa, hex_types

def load_inat_taxa(taxa_file):
    """
    Reads meta data file for the taxas

    Parameters:
        - taxa_file: file path of metadata file
    Returns:
        - taxa: array of taxa in the metadata
    """

    with open(taxa_file, 'r') as f:
        metadata = json.load(f)
    
    taxa = []
    for obj in metadata:
        taxon_id = obj["taxon_id"]
        taxa.append(taxon_id)
    
    return np.asarray(taxa, np.int64)

def get_annotation_data(params):
    """
    Loads annotation csv into BinaryLocationDataset.

    Parameters:
        - params: training parameters set for the fine-tuning process
    Returns:
        - ds: BinaryLocationDataset, dataset that has an additional field to specify location annotation type (0 | 1) corresponds to absence | presence
    """

    with open('paths.json', 'r') as f:
        paths = json.load(f)
    
    data_dir = paths['train']
    annotation_dir = paths['annotation']
    train_dir = paths['train']
    annotation_file = os.path.join(annotation_dir, params['annotation_file'])
    train_file = os.path.join(train_dir, params['obs_file'])
    taxa_file = os.path.join(data_dir, params['taxa_file'])
    taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

    taxa_of_interest = datasets.get_taxa_of_interest(params['species_set'], params['num_aux_species'], params['aux_species_seed'], params['taxa_file'], taxa_file_snt)
    labels = load_inat_taxa(taxa_file)
    print('Loaded labels from metadata json')

    locs, _, hex_types = load_annotation_data(annotation_file, taxa_of_interest) # has only labels that are annotated, drop labels
    print('Loaded annotations')

    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    class_info_file = json.load(open(taxa_file, 'r'))
    class_names_file = [cc['latin_name'] for cc in class_info_file]
    taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    classes = dict(zip(taxa_ids_file, class_names_file))

    # idx_ss = datasets.get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'])
    locs = torch.from_numpy(np.array(locs)) # convert to Tensor
    labels = torch.from_numpy(np.array(class_ids))

    ds = datasets.BinaryLocationDataset(locs, labels, classes, hex_types, class_to_taxa, params['input_enc'], params['device']) # use labels loaded from metadata to avoid model dimension conflict

    return ds

class FineTuner():
    """
    Fine tuner suite used to mainly fine tune a provided geomodel. It additionally has an option to freeze locational embedder, further details in the Fine Tuning report.
    """

    def __init__(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, params: dict, freeze_loc_emb=True):
        self.model = model
        self.params = params
        self.loader = data_loader

        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = self.loader.dataset.enc.encode

        if freeze_loc_emb:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.class_emb.parameters():
                param.requires_grad = True
            
        # self.optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
        self.optimizer = torch.optim.Adam(self.model.class_emb.parameters(), params['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])
    
    def save_model(self):
        save_path = os.path.join(self.params['save_path'], str(self.params['model_name'] + '.pt'))
        print('Saving to: ', save_path)
        op_state = {'state_dict': self.model.state_dict(), 'params': self.params}
        torch.save(op_state, save_path)

    def train_one_epoch(self):
        self.model.train()

        running_loss = 0.0
        samples_processed = 0
        steps_trained = 0
        for _, batch in enumerate(self.loader):
            self.optimizer.zero_grad()

            batch_loss = self.compute_loss(batch, self.model, self.params, self.encode_location)
            batch_loss.backward()
            self.optimizer.step()

            running_loss += float(batch_loss.item())
            steps_trained += 1
            samples_processed += batch[0].shape[0]
            if steps_trained % self.params['log_frequency'] == 0:
                print(f'[{samples_processed}/{len(self.loader.dataset)})] loss: {np.around(running_loss / self.params["log_frequency"], 4)}')
                running_loss = 0
            
        self.lr_scheduler.step()

def launch_fine_tuning_run(ovr):
    """
    Launches fine tuning process.
    Uses fine tuning parameters to configure save location.
    It will overwrite existing model if in the same path.

    Parameters:
        - ovr: parameters for fine tuning
    Returns:
        - None: model will be saved to 'save_path'
    """

    params = setup.get_default_params_train(ovr)
    params['save_path'] = os.path.join(params['fine_tuned_save_base'], params['fine_tuned_experiment_name'])
    os.makedirs(params['save_path'], exist_ok=True)

    # data:
    train_dataset = get_annotation_data(params)
    params['input_dim'] = train_dataset.input_dim
    params['num_classes'] = train_dataset.num_classes
    params['class_to_taxa'] = train_dataset.class_to_taxa
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4)

    # model:
    pretrain_params = torch.load(ovr['pretrain_model_path'], map_location='cpu')
    model = models.get_model(pretrain_params['params'])
    model.load_state_dict(pretrain_params['state_dict'], strict=True)
    model = model.to(params['device'])
    print(model)

    myparams = {k: params[k] for k in set(list(params.keys())) - set(['class_to_taxa'])}
    print('Params: ', myparams)

    # train:
    trainer = FineTuner(model, train_loader, params, True) # True: freezes the locational embedder of the model with freeze it only trains class_emb the last layer, False: trains the whole network (predictions for unannotated species will also change)
    for epoch in range(0, params['num_epochs']):
        print(f'epoch {epoch+1}')
        trainer.train_one_epoch()
    trainer.save_model()
