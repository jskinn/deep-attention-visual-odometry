from typing import Type
from functools import partial
import torch.nn as nn
from torch.nn.init import uniform_, kaiming_normal_


def init_weights_kaiming_normal(
    module: nn.Module, module_types: set[Type[nn.Module]], mode: str, nonlinearity: str
):
    if any(isinstance(module, module_type) for module_type in module_types):
        if hasattr(module, "weight"):
            kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, "bias"):
            uniform_(module.bias, 0.0)


def get_kaiming_normal_init_function(
    module_types: set[Type[nn.Module]], mode: str, nonlinearity: str
):
    return partial(
        init_weights_kaiming_normal,
        module_types=module_types,
        mode=mode,
        nonlinearity=nonlinearity,
    )
