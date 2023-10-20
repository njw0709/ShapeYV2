import os
import shapeymodular.data_loader as dl
import shapeymodular.visualization as vis
import matplotlib.pyplot as plt
import shapeymodular.macros.graphing as graphing
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine error graphs from different cases"
    )
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="+",
        help="<Required> Directories to combine the error graphs",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="<Required> Output directory to save the combined error graphs",
        required=True,
    )
    parser.add_argument(
        "--analysis_file",
        type=str,
        default="analysis_results.h5",
    )
    parser.add_argument(
        "--axes_choice",
        type=str,
        default="pw",
    )
    parser.add_argument(
        "--fig_save_dir",
        type=str,
        default="figures",
    )
    parser.add_argument(
        "--config_filename",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ylog",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--fig_format",
        type=str,
        default="png",
    )
    parser.add_argument(
        "--marker_size",
        type=str,
        default="normal",
    )
    parser.add_argument(
        "--legends",
        nargs="+",
        default=[],
    )
    args = parser.parse_args()
    graphing.combine_nn_classification_error_graphs(
        args.dirs,
        args.output,
        analysis_file=args.analysis_file,
        axes_choice=args.axes_choice,
        fig_save_dir=args.fig_save_dir,
        config_filename=args.config_filename,
        log_scale=args.ylog,
        fig_format=args.fig_format,
        marker_size=args.marker_size,
        legends=args.legends,
    )
