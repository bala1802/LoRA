import torch
import torch.nn as nn

def initialize_loss():
    return nn.CrossEntropyLoss()

def initialize_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def extract_weights(model):
    weights_dict = {}
    for name, param in model.parameters():
        weights_dict[name] = param.clone().detach()
    
    return weights_dict