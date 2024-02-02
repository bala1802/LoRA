import torch.nn.utils.parametrize as parametrize

from lora_model import LoRA_Parameterization
import config

def layer_parameterization(layer):
    d, k = layer.weight.shape
    return LoRA_Parameterization(
        d=d, k=k, rank=config.RANK, alpha=config.ALPHA
    )

def enable_or_disable_lora(model_layers, enabled):
    for layer in model_layers:
        layer.parametrizations["weight"][0].enabled=enabled