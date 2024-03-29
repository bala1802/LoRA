import torch
import torch.nn as nn

def initialize_loss():
    return nn.CrossEntropyLoss()

def initialize_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def extract_weights(model):
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.clone().detach()
    
    return weights_dict

def count_total_parameters(model_layers):
    original_parameters_count = 0
    for index, layer in enumerate(model_layers):
        original_parameters_count += layer.weight.nelement() + layer.bias.nelement()
        print(f'Layer {index+1}: W (Matrix): {layer.weight.shape} + Bias : {layer.bias.shape}')
    return original_parameters_count

def freeze_model_parameters(model, unfreeze_layer):
    for name, param in model.named_parameters():
        if unfreeze_layer not in name:
            print(f'Freezing Non LoRA layer parameter {name}')
            param.requires_grad = False
        else:
            print("Unfreeze Layer found : ", name)