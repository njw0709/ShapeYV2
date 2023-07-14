import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Callable

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_extension: str = ".png",
        transform: Callable = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(), normalize]
        ),
    ):
        self.root_dir = root_dir
        self.image_files = [
            img for img in os.listdir(root_dir) if img.endswith(file_extension)
        ]
        self.image_files.sort()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image
