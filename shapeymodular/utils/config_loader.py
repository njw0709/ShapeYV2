from dacite import from_dict
import json
import shapeymodular.data_classes as dc


def load_config(json_path: str) -> dc.NNAnalysisConfig:
    with open(json_path, "r") as f:
        config_dict = json.load(f)
        config = from_dict(data_class=dc.NNAnalysisConfig, data=config_dict)
    return config
