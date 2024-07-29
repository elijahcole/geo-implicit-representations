import models
import torch

eval_params = {
    'model_path': './pretrained_models/test_fine_tune.pt',
    'device': 'cpu'
}

def load_model(eval_params):
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    print(train_params)
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    return train_params, model

params, model = load_model(eval_params)

print(params)
print(model)