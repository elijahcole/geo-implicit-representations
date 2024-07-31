import os
import json

import fine_tune

train_params = {}

train_params['experiment_name'] = 'demo' # This will be the name of the directory where results for this run are saved.

'''
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds'
'''
train_params['species_set'] = 'all'

'''
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
'''
train_params['hard_cap_num_per_class'] = -1

'''
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
'''
train_params['num_aux_species'] = 0

'''
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env'
'''
train_params['input_enc'] = 'sin_cos'

'''
loss
- Which loss to use for training.
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me', 'bce' (for fine tuning), 'an_full_bce'
'''
train_params['loss'] = 'an_full_bce'

with open("paths.json", 'r') as f:
    paths = json.load(f)

pretrain_path = paths["pretrain"]

pre_trained_models = {
    "npc10": {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt",
        "cap": 10,
    },
    "npc100": {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt",
        "cap": 100,
    },
    "npc1000": {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt",
        "cap": 1000,
    }
}

train_params['log_frequency'] = 5
train_params['batch_size'] = 512

train_params['pretrain_model_path'] = os.path.join(pretrain_path, pre_trained_models['npc10']['path']) # you can choose a base model, refer to dictionary above
train_params['annotation_file'] = 'presence_absence2.csv' # enter the csv file you want to train on
train_params['model_name'] = 'ft_pa2' # name your output model, it will be saved in ./fine-tuned/${experiment_name}/${model_name}.pt

train_params['lr'] = 5e-5
train_params['lr_decay'] = 0.2

train_params['pos_weight'] = 128

if __name__ == '__main__':
    fine_tune.launch_fine_tuning_run(train_params)
