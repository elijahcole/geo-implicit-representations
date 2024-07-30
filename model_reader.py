import models
import torch

eval_params = {
    'model_path': '/Users/ozelyilmazel/Documents/ds4cg2024-inaturalist/src/backend/sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt',
    'device': 'cpu'
}

def load_model(eval_params):
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    # print(train_params)
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    return train_params, model

params, model = load_model(eval_params)

del params['params']['class_to_taxa']

print(params['params'])
print(model)
