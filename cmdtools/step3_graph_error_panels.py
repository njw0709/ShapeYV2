import argparse
import os
import shapeymodular.macros.graphing as graphing

#     feature_directory: str,
# analysis_file: str = "analysis_results.h5",
# axes_choice: str = "pw",
# fig_save_dir: str = "figures",
# config_filename: Union[None, str] = None,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph results per case")
    parser.add_argument(
        "--dir",
        type=str,
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
        "--distance_file",
        type=str,
        default="distances-Jaccard.mat",
    )
    parser.add_argument(
        "--thresholds_file",
        type=str,
        default="thresholds.mat",
    )
    parser.add_argument(
        "--row_descriptions",
        type=str,
        default="imgnames_all.txt",
    )
    parser.add_argument(
        "--col_descriptions",
        type=str,
        default="imgnames_all.txt",
    )
    args = parser.parse_args()

    graphing.plot_error_panels(
        args.dir,
        analysis_file=args.analysis_file,
        distances_file=args.distance_file,
        thresholds_file=args.thresholds_file,
        axes_choice=args.axes_choice,
        fig_save_dir=args.fig_save_dir,
        config_filename=args.config_filename,
        row_descriptions=args.row_descriptions,
        col_descriptions=args.col_descriptions,
    )
