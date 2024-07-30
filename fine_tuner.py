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

def load_annotation_data(annotation_file, taxa_of_interest):
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

    return locs, taxa

def get_annotation_data(params):
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

    _, labels, _, _, _, _ = datasets.load_inat_data(train_file, taxa_of_interest) # load labels from original training data to combat dimensional conflict
    locs, _ = load_annotation_data(annotation_file, taxa_of_interest) # has only 2 labels
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    class_info_file = json.load(open(taxa_file, 'r'))
    class_names_file = [cc['latin_name'] for cc in class_info_file]
    taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    classes = dict(zip(taxa_ids_file, class_names_file))

    # idx_ss = datasets.get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'])
    locs = torch.from_numpy(np.array(locs)) # convert to Tensor
    labels = torch.from_numpy(np.array(class_ids))

    ds = datasets.LocationDataset(locs, labels, classes, class_to_taxa, params['input_enc'], params['device'])

    return ds

class FineTuner():
    def __init__(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, params: dict):
        self.model = model
        self.params = params
        self.loader = data_loader

        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = self.loader.dataset.enc.encode

        self.optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])
    
    def save_model(self):
        save_path = os.path.join(self.params['save_path'], str(self.params['model_name'] + '.pt'))
        op_state = {'state_dict': self.model.state_dict(), 'params': self.params}
        torch.save(op_state, save_path)

        # save_path_json = os.path.join(self.params['save_path'], str(self.params['model_name'] + '.json'))
        # with open(save_path_json, "w") as f:
        #     json.dump(dict(self.params), f)

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
    params = setup.get_default_params_train(ovr)
    params['save_path'] = os.path.join(params['fine_tuned_save_base'], params['fine_tuned_experiment_name'])
    os.makedirs(params['save_path'], exist_ok=True)

    # model:
    pretrain_params = torch.load(ovr['pretrain_model_path'], map_location='cpu')
    model = models.get_model(pretrain_params['params'])
    model.load_state_dict(pretrain_params['state_dict'], strict=True)
    model = model.to(params['device'])

    # data:
    train_dataset = get_annotation_data(params)
    # params['input_dim'] = pretrain_params['input_dim']
    params['input_dim'] = train_dataset.input_dim
    # params['num_classes'] = pretrain_params['num_classes']
    params['num_classes'] = train_dataset.num_classes
    params['class_to_taxa'] = train_dataset.class_to_taxa
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4)

    # train:
    trainer = FineTuner(model, train_loader, params)
    for epoch in range(0, params['num_epochs']):
        print(f'epoch {epoch+1}')
        trainer.train_one_epoch()
    trainer.save_model()
