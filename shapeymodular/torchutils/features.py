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
    std_near_zero = torch.isclose(std, torch.tensor([0.0]))
    num_near_zero = torch.sum(std_near_zero)
    if num_near_zero > 0:
        print("found {} near zero standard deviation features".format(num_near_zero))
        std[torch.isclose(std, torch.tensor([0.0]))] = 1.0  # avoid division by zero
    output = (features - mean) / std
    return output
