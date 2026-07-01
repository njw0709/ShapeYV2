# ShapeY version 2

ShapeY is a benchmark that tests a vision system's shape recognition capacity. ShapeY currently consists of ~68k images of 200 3D objects taken from ShapeNet. Note that this benchmark is not meant to be used as a training dataset, but rather serves to validate that the visual object recogntion / classification under inspection has developed a capacity to perform well on our benchmarking tasks, which are designed to be hard if the system does not understand shape.

## Installing ShapeY
Requirements: Python 3.9, Cuda version 10.2 (prerequisite for cupy)

To install ShapeY, run the following command:
```
pip install ShapeYModular==2.0.6
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

## Processing images into a nearest-neighbor matrix

The analysis pipeline turns raw images into a nearest-neighbor (exclusion) matrix in three
stages. Each stage operates on a single "feature directory" (referred to as `<DIR>` below),
which accumulates the intermediate files (`features_*.h5`, `distances-correlation.h5`,
`analysis_results.h5`) as you go. The reusable functions live in
[shapeymodular/macros/](shapeymodular/macros/); the analysis and graphing command-line entry
points live in [cmdtools/](cmdtools/).

The only difference between the **original** and **contrast-reversed (CR)** experiments is the
image dataset you feed into the feature-extraction step (`ShapeY200` vs. `ShapeY200CR`). The
distance / analysis steps are identical once the features exist. The dataset paths are
configured in `SHAPEY200_DATASET_PATH_DICT` in
[shapeymodular/utils/constants.py](shapeymodular/utils/constants.py) under the keys `"original"`
and `"contrast_reversed"` — point those at your local copies of `ShapeY200/dataset` and
`ShapeY200CR/dataset`.

### Step A — Extract features from images
Run a model over the ~68k images and save the per-image feature vectors to an HDF5 file.
Use `dataset_version="original"` for the original experiment and
`dataset_version="contrast_reversed"` for the CR experiment (see
[run_analysis/step0_extract_features.py](run_analysis/step0_extract_features.py) for a full example):

```python
import torch
import torchvision.models as models
import shapeymodular.torchutils as tu
import shapeymodular.macros.extract_features as ef

model = models.resnet50(pretrained=True)
model_gap = tu.GetModelIntermediateLayer(model, -1)
savedir = "/path/to/<DIR>/"

# Original experiment
ef.extract_shapey200_features(
    savedir, model=model_gap, overwrite=True, dataset_version="original"
)

# Contrast-reversed experiment (only this argument changes)
ef.extract_shapey200_features(
    savedir, model=model_gap, overwrite=True, dataset_version="contrast_reversed"
)
```
This writes `features_<model_name>_<dataset_version>.h5` into `savedir`.

### Step B — Compute the cosine-similarity matrix
Compute the pairwise cosine similarity between all image feature vectors. This is done with
`metric="correlation"`, which L2-normalizes each feature vector and then takes the dot product
(i.e. cosine similarity); the computation is segmented so it scales to the full ~68k × ~68k
matrix on the GPU. Use
[shapeymodular/macros/compute_distance_torch.py](shapeymodular/macros/compute_distance_torch.py):

```python
import h5py
import shapeymodular.macros.compute_distance_torch as compute_distance

# features_<model_name>_<dataset_version>.h5 from Step A
with h5py.File("/path/to/<DIR>/features_resnet50_original.h5", "r") as hf:
    features = hf["feature_output/output"][()]

compute_distance.compute_distance(
    output_dir="/path/to/<DIR>/",
    features=features,
    output_file="distances-correlation.h5",
    row_segment_size=1000,
    col_segment_size=1000,
    gpu_index=0,
    metric="correlation",
)
```
Writes `distances-correlation.h5` into `<DIR>`, containing a `"correlation"` dataset (the cosine
similarity matrix) that every downstream tool reads. Unlike binarized distances, cosine
similarity needs no separate thresholding step.

#### Contrast-reversed (CR) experiment — compute **two** matrices
The CR benchmark asks whether the system finds the correct shape even when the candidate images
have their contrast reversed. That requires **two** cosine-similarity matrices, computed from the
original and contrast-reversed features and fed to the analysis **in sequence**:

1. **Same-contrast matrix** — original features vs. original features (identical to the standard
   Step B above; probe and candidates share the original contrast polarity).
2. **Cross-contrast matrix** — original features vs. contrast-reversed features (probe is
   original, candidates are contrast-reversed). Compute it with `dataset_exclusion=True`, passing
   the two feature sets as a list `[row_features, col_features]` (row = original,
   col = contrast-reversed):

```python
import h5py
import shapeymodular.macros.compute_distance_torch as compute_distance

with h5py.File("/path/to/<DIR>/features_resnet50_original.h5", "r") as hf:
    features_orig = hf["feature_output/output"][()]
with h5py.File("/path/to/<DIR>/features_resnet50_contrast_reversed.h5", "r") as hf:
    features_cr = hf["feature_output/output"][()]

# (1) same-contrast matrix -> distances-correlation.h5   (same call as standard Step B)
compute_distance.compute_distance(
    output_dir="/path/to/<DIR>/", features=features_orig,
    output_file="distances-correlation.h5",
    row_segment_size=1000, col_segment_size=1000, gpu_index=0, metric="correlation",
)

# (2) cross-contrast matrix -> distances-correlation-cr.h5
compute_distance.compute_distance(
    output_dir="/path/to/<DIR>/", features=[features_orig, features_cr],
    output_file="distances-correlation-cr.h5",
    row_segment_size=1000, col_segment_size=1000, gpu_index=0, metric="correlation",
    dataset_exclusion=True,
)
```
Both files carry a `"correlation"` dataset. The order matters: the analysis reads the
same-contrast matrix first and the cross-contrast matrix second (see the `--distance_file`
tuple in Step C).

**Soft vs. hard mode** (`contrast_exclusion_mode` in the config — set it to `"soft"` or
`"hard"`). The positive / same-object match *always* uses the cross-contrast matrix, so the
correct answer is always a contrast-reversed view of the probe. The mode only changes which
matrix supplies the negative / other-object distractors:

- **`"soft"`** — the negative match also uses the cross-contrast matrix, so *both* positive and
  negative match candidates are contrast-reversed (they share the same reversed background).
- **`"hard"`** — the negative match uses the same-contrast matrix, so the distractors keep the
  original contrast polarity (same as the probe) while the positive candidate is contrast-reversed
  (different polarity). This is the harder condition: a shape-blind system can exploit the
  distractor's matching contrast to beat the correct, contrast-reversed match.

### Step C — Nearest-neighbor (exclusion) analysis
Runs the exclusion-distance nearest-neighbor analysis over the cosine-similarity matrix and
produces the final nearest-neighbor matrix / analysis results. Uses
[cmdtools/step2_nn_analysis.py](cmdtools/step2_nn_analysis.py):

```
cp shapeymodular/utils/imgnames_all.txt <DIR>/
python cmdtools/step2_nn_analysis.py \
    --dir <DIR> \
    --distance_file distances-correlation.h5 \
    --row_imgnames imgnames_all.txt \
    --col_imgnames imgnames_all.txt \
    --save_name analysis_results.h5
```
Writes `analysis_results.h5` into `<DIR>` — this is the nearest-neighbor result consumed by all
graphing tools below. Whether contrast exclusion is applied is controlled by the analysis config;
the config paths are defined in
[shapeymodular/utils/constants.py](shapeymodular/utils/constants.py) (`PATH_CONFIG_*_CR` for the
contrast-reversed experiment, `PATH_CONFIG_*_NO_CR` for the original). These configs read the
matrix from the `"correlation"` dataset written in Step B (the CR configs list it twice, one key
per matrix).

**Contrast-reversed (CR):** the CR run consumes the two matrices from Step B as an ordered pair.
The single-file `--distance_file` CLI flag can't express that, so run the analysis via the macro
directly, passing a `(same_contrast, cross_contrast)` tuple and a CR config
(`contrast_exclusion: true`, and `contrast_exclusion_mode` set to `"soft"` or `"hard"` — see the
soft-vs-hard explanation under Step B for the difference):

```python
import shapeymodular.macros.nn_batch as nn_batch
import shapeymodular.utils as utils

nn_batch.run_exclusion_analysis(
    "/path/to/<DIR>/",
    distance_file=("distances-correlation.h5", "distances-correlation-cr.h5"),  # order matters
    row_imgnames="imgnames_all.txt",
    col_imgnames="imgnames_all.txt",
    save_name="analysis_results.h5",
    config_filename=utils.PATH_CONFIG_ALL_CR,  # or PATH_CONFIG_PW_CR
)
```
The CR result file is suffixed automatically (e.g. `analysis_results_cr_soft.h5`).

> **Automating the pipeline:** [run_analysis/](run_analysis/) contains driver scripts that loop
> the above over many feature directories. [run_analysis/macro.py](run_analysis/macro.py) chains
> the stages end to end; edit the `all_features_directories` lists at the top of each `step*`
> script to point at your directories.

## Command-line tools (`cmdtools/`)

All tools are plain `argparse` scripts; run any with `-h` to see the full option list. Common
options: `--dir`/`--features_dir` selects the working feature directory, and the analysis-based
graphing tools accept `--analysis_file` (default `analysis_results.h5`), `--axes_choice`
(default `pw`), `--fig_save_dir` (default `figures`) and `--config_filename`.

| Tool | Purpose | Key example |
|------|---------|-------------|
| [step2_nn_analysis.py](cmdtools/step2_nn_analysis.py) | Nearest-neighbor exclusion analysis over the cosine-similarity matrix → `analysis_results.h5` | `python cmdtools/step2_nn_analysis.py --dir <DIR> --distance_file distances-correlation.h5` |
| [step3_graph_nn_classification_error.py](cmdtools/step3_graph_nn_classification_error.py) | Plot nearest-neighbor classification error curves (`--ylog`, `--fig_format`) | `python cmdtools/step3_graph_nn_classification_error.py --dir <DIR>` |
| [step3_graph_error_panels.py](cmdtools/step3_graph_error_panels.py) | Plot per-case error panels | `python cmdtools/step3_graph_error_panels.py --dir <DIR>` |
| [step3_graph_histogram.py](cmdtools/step3_graph_histogram.py) | Similarity histograms alongside the error graph | `python cmdtools/step3_graph_histogram.py --dir <DIR>` |
| [step3_graph_tuning_curves.py](cmdtools/step3_graph_tuning_curves.py) | Plot tuning curves from the similarity matrix | `python cmdtools/step3_graph_tuning_curves.py --dir <DIR> --distance_file distances-correlation.h5` |
| [step4_combine_error_graphs.py](cmdtools/step4_combine_error_graphs.py) | Combine error graphs across multiple directories | `python cmdtools/step4_combine_error_graphs.py -d <DIR1> <DIR2> -o <OUTDIR>` |

## Other graphing tools

Beyond the `step3_*` command-line graphers, [shapeymodular/visualization/](shapeymodular/visualization/)
exposes classes for building figures directly from the analysis / distance files. Two of the most
useful are below; see the notebooks in [notebooks/](notebooks/) for complete, runnable examples.

### Rank histograms — `shapeymodular.visualization.rank_histogram`
Shows, as a function of exclusion distance, how the correct object (or category) ranks among the
nearest neighbors. `RankHistogramSampler` pulls the rank matrix out of `analysis_results.h5`, and
`RankHistogramGraph.draw` renders it. Full example:
[notebooks/rank_histogram.ipynb](notebooks/rank_histogram.ipynb).

```python
import os, h5py, matplotlib.pyplot as plt
import shapeymodular.data_loader as dl
import shapeymodular.data_classes as dc
from shapeymodular.visualization.rank_histogram import (
    RankHistogramSampler,
    RankHistogramGraph,
)

feature_directory = "/path/to/<DIR>/"
config = dc.load_config(os.path.join(feature_directory, "analysis_config_all.json"))
analysis_hdf = h5py.File(os.path.join(feature_directory, "analysis_results.h5"), "r")
sampler = dl.Sampler(dl.HDFProcessor(), analysis_hdf, config)

ax = "pr"  # exclusion axis
obj_rank_mat = RankHistogramSampler.get_objrank_mat_all(ax, sampler, category=False)
cat_rank_mat = RankHistogramSampler.get_objrank_mat_all(ax, sampler, category=True)

last_xdist_to_show = 5
fig, axes = plt.subplots(2, last_xdist_to_show, figsize=(8, 4), constrained_layout=True)
RankHistogramGraph.draw(axes[0], obj_rank_mat, last_xdist_to_show=last_xdist_to_show, last_rank_to_show=10)
RankHistogramGraph.draw(axes[1], cat_rank_mat, last_xdist_to_show=last_xdist_to_show, last_rank_to_show=10, category=True)
```

### Similarity / correlation-dropoff histograms — `shapeymodular.visualization.histogram_correlation.SimilarityHistogramGraph`
Plots the distribution of positive-match-candidate (PMC) vs. negative-match-candidate (NMC)
similarity scores as a function of exclusion distance, straight from the cosine-similarity matrix.
`SimilarityHistogramSampler` gathers the scores and `SimilarityHistogramGraph` draws the histogram,
correlation fall-off, and image panels. Full example:
[notebooks/correlation_dropoff_histogram.ipynb](notebooks/correlation_dropoff_histogram.ipynb).

```python
import h5py
from shapeymodular.visualization.histogram_correlation import (
    SimilarityHistogramGraph,
    SimilarityHistogramSampler,
)
import shapeymodular.utils as utils
from shapeymodular.utils.constants import SHAPEY200_OBJS

with h5py.File("/path/to/<DIR>/distances-correlation.h5", "r") as hf:
    cmat = hf["correlation"][()]  # cosine-similarity matrix from Step B

imgname = SHAPEY200_OBJS[3] + "-pr03.png"
parsed = utils.ImageNameHelper.parse_imgname(imgname)
(
    (pmc_corrvals_with_xdist, top1_pmc_cval_with_xdist, top1_pmc_imgnames_with_xdist),
    (nmc_corrvals, top1_nmc_cval, top1_nmc_imgname, top_per_obj_nmc_cvals),
) = SimilarityHistogramSampler.get_pmc_nmc_scores_with_xdist(imgname, cmat, category=False)

fig, (grid_ax, cval_ax) = SimilarityHistogramGraph.create_axis_grid()
SimilarityHistogramGraph.draw_similarity_histogram(
    cval_ax, pmc_corrvals_with_xdist, nmc_corrvals, top_per_obj_nmc_cvals, sample_size=800
)
SimilarityHistogramGraph.draw_correlation_fall_off(cval_ax, top1_pmc_cval_with_xdist)
SimilarityHistogramGraph.format_similarity_histogram(cval_ax, parsed)
```

Other classes in the same package include `DistanceHistogram`
([histogram.py](shapeymodular/visualization/histogram.py)),
tuning-curve helpers ([tuning_curve.py](shapeymodular/visualization/tuning_curve.py)),
and error-panel / image utilities ([image.py](shapeymodular/visualization/image.py)).

