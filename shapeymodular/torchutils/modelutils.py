import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import shapeymodular.data_loader as dl
import numpy as np
from tqdm import tqdm
import copy


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
    batch_size: int = 30,
    timm: bool = False,
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
        if timm:
            feature_vector = model.forward_features(img)
        else:
            feature_vector = model(img)
        output = feature_vector.view(feature_vector.size(0), -1)
        output_np = output.cpu().data.numpy()
        features.append(output_np)

    features = np.concatenate(features, axis=0)
    return features


# Function to check if a certain batch size fits into GPU memory
def can_allocate_model(model, input_size, batch_size, device_id=0):
    try:
        device = torch.device("cuda:{}".format(device_id))

        # Generate dummy data with the current batch size
        input_tensor = torch.randn(batch_size, *input_size).to(device)
        model_copy = copy.deepcopy(model).to(device)
        model_copy.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            # Forward pass through the model copy
            output = model_copy(input_tensor)

        # Clean up to free GPU memory
        del model_copy
        del input_tensor
        torch.cuda.empty_cache()
        return True  # Allocation successful
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(e)
            del model_copy
            del input_tensor
            torch.cuda.empty_cache()
            return False  # Out of memory error
        else:
            del model_copy
            del input_tensor
            torch.cuda.empty_cache()
            raise e  # Any other error, re-raise


def batch_size_allocation_binary_search(
    model, input_size, low: int = 0, high: int = 10, device_id: int = 0
):
    # Binary search for the largest batch size
    low = 2**low
    high = 2**high

    while low < high:
        mid = (low + high + 1) // 2
        if can_allocate_model(model, input_size, mid, device_id=device_id):
            low = mid  # If it fits, try a larger batch size
        else:
            high = mid - 1  # If it doesn't fit, reduce the batch size
    return low
