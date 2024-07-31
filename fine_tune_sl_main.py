import os
import json

import fine_tune_sl

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
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me'
'''
train_params['loss'] = 'an_full'

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

train_params['pretrain_model_path'] = os.path.join(pretrain_path, pre_trained_models['npc10']['path'])
train_params['annotation_file'] = 'example2.csv'
train_params['model_name'] = 'fine_tune_xsmall_lr'

train_params['lr'] = 0.0001
train_params['lr_decay'] = 0.8

if __name__ == '__main__':
    fine_tune_sl.launch_fine_tuning_run(train_params)
