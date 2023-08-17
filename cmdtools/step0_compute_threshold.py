import argparse
import shapeymodular.data_loader as dl
from shapeymodular.macros.compute_threshold import compute_threshold_subsample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        type=str,
    )
    parser.add_argument(
        "--variable_name",
        type=str,
        default="l2pool",
    )
    args = parser.parse_args()

    data_loader = dl.FeatureDirMatProcessor()
    compute_threshold_subsample(
        args.features_dir, data_loader, variable_name=args.variable_name
    )
    print("Done")
