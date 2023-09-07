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
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--orientation_axis",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--threshold_level",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--dtype",
        type=type,
        default=None,
    )

    args = parser.parse_args()

    data_loader = dl.FeatureDirMatProcessor()
    compute_threshold_subsample(
        args.features_dir,
        data_loader,
        variable_name=args.variable_name,
        save_dir=args.save_dir,
        file_name=args.file_name,
        sample_size=args.sample_size,
        orientation_axis=args.orientation_axis,
        threshold_level=args.threshold_level,
        dtype=args.dtype,
    )
    print("Done")
