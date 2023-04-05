from dataclasses import dataclass
from typing import List, Optional

# Dataset Parameters
@dataclass
class DatasetConfig:
    project_dir: str
    data_dir: str
    cr: bool
    cr_data_dir: Optional[str]


# Network inference parameters
@dataclass
class NetworkParams:
    name: str
    input_size: int
    batch_size: int
    num_workers: int
    gpu_num: int


@dataclass
class DistanceParams:
    metric: str


@dataclass
class AnalysisParams:
    match_mode: str
    cr_mode: Optional[str]


@dataclass
class PipelineParams:
    step1_input: str
    step1_output: str
    step1_recompute: bool
    step2_input: str
    step3_input: str
    step3_output: str
    step4_input: str
    step4_output: str


@dataclass
class ShapeYConfig:
    data: DatasetConfig
    network: NetworkParams
    graph: AnalysisParams
    pipeline: PipelineParams
    distance: DistanceParams
