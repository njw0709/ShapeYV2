import torch.nn as nn


class GetModelIntermediateLayer(nn.Module):
    def __init__(self, original_model, layerindex):
        super(GetModelIntermediateLayer, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:layerindex])

    def forward(self, x):
        x = self.features(x)
        return x
