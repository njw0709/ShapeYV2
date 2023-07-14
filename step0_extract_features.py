import shapeymodular.macros.extract_features as ef
import torchvision.models as models
import shapeymodular.torchutils as tu
import torch

torch.cuda.set_device(1)


model = models.resnet50(pretrained=True)
model_gap = tu.GetModelIntermediateLayer(model, -1)
savedir = "/home/namj/projects/shapeyv2results/resnet50/"

ef.extract_shapey200_features(savedir, model=model_gap, overwrite=True)
