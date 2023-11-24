import torch


def standardize_features(features: torch.Tensor) -> torch.Tensor:
    # features: (n, l)
    # output: (n, l)
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    output = (features - mean) / std
    return output
