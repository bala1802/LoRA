import torch
import torch.nn as nn

def initialize_loss():
    return nn.CrossEntropyLoss()

def initialize_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
