#!/bin/sh

gdown https://drive.google.com/uc?id=1arDu0c9hYLHVMiB52j_a-e0gVnyQfuQV 
unzip ShapeY200.zip
mv ShapeY200 ./data

gdown https://drive.google.com/uc?id=1WXpNUVRn6D0F9T3IHruml2DcDCFRsix-
unzip ShapeY200CR.zip
mv ./dataset_postprocess_bw_reverse_shapenet_200/ ./data/ShapeY200CR