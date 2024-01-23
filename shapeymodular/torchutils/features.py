import torch
from typing import Union, Tuple


def standardize_features(
    features: torch.Tensor,
    mean: Union[torch.Tensor, None] = None,
    std: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    # features: (n, l)
    # output: (n, l)
    if mean is None:
        mean = features.mean(dim=0, keepdim=True)
    if std is None:
        std = features.std(dim=0, keepdim=True)
    output = (features - mean) / std
    return output
