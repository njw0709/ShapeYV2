import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import shapeymodular.data_loader as dl
import numpy as np
from tqdm import tqdm


class GetModelIntermediateLayer(nn.Module):
    def __init__(self, original_model: nn.Module, layerindex: int):
        super(GetModelIntermediateLayer, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:layerindex])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


@torch.no_grad()
def extract_feature_vectors(
    model: nn.Module,
    img_dataset: Dataset,
    batch_size: int = 10,
) -> np.ndarray:
    device = next(model.parameters()).device
    if model.training:
        model.eval()

    img_dataloader = DataLoader(
        img_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    features = []
    for img in tqdm(img_dataloader):
        img = img.to(device)
        feature_vector = model(img)
        output = feature_vector.view(feature_vector.size(0), -1)
        output_np = output.cpu().data.numpy()
        features.append(output_np)

    features = np.concatenate(features, axis=0)
    return features
