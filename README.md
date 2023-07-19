# ShapeY version 2

ShapeY is a benchmark that tests a vision system's shape recognition capacity. ShapeY currently consists of ~68k images of 200 3D objects taken from ShapeNet. Note that this benchmark is not meant to be used as a training dataset, but rather serves to validate that the visual object recogntion / classification under inspection has developed a capacity to perform well on our benchmarking tasks, which are designed to be hard if the system does not understand shape.

## Installing ShapeY
Requirements: Python 3.9, Cuda version 10.2 (prerequisite for cupy)

To install ShapeY, run the following command:
```
pip install ShapeYModular==2.0.3
```

## Step0: Download ShapeY200 dataset
Run `download.sh` to download the dataset. The script automatically unzips the images under `data/ShapeY200/`.
Downloading uses gdown, which is google drive command line tool. If it does not work, please just follow the two links down below to download the ShapeY200 / ShapeY200CR datasets.

ShapeY200:
https://drive.google.com/uc?id=1arDu0c9hYLHVMiB52j_a-e0gVnyQfuQV

ShapeY200CR:
https://drive.google.com/uc?id=1WXpNUVRn6D0F9T3IHruml2DcDCFRsix-

After downloading the two datasets, move each of them to the `data/` directory. For example, all of the images for ShapeY200 should be under `data/ShapeY200/dataset/`.

## Step1: Setup environment variable
Set the environment variable `SHAPEY_IMG_DIR` to the path of the ShapeY200 dataset. For example, if the dataset is under `/data/ShapeY200/dataset/`, then run the following command:
```
export SHAPEY_IMG_DIR=/data/ShapeY200/dataset/
```

