import argparse
import shapeymodular.macros.graphing as graphing

# feature_directory: str,
#     distances_file: str = "distances-Jaccard.mat",
#     axes_choice: str = "pw",
#     fig_save_dir: str = "figures",
#     config_filename: Union[None, str] = None,
#     row_descriptions: Union[None, str] = None,
#     col_descriptions: Union[None, str] = None,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph results per case")
    parser.add_argument(
        "--dir",
        type=str,
    )

    parser.add_argument(
        "--axes_choice",
        type=str,
        default="pw",
    )
    parser.add_argument(
        "--distances_file",
        type=str,
        default="distances-Jaccard.mat",
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

    graphing.plot_tuning_curves(
        args.dir,
        distances_file=args.distances_file,
        axes_choice=args.axes_choice,
        fig_save_dir=args.fig_save_dir,
        config_filename=args.config_filename,
        row_descriptions=args.row_descriptions,
        col_descriptions=args.col_descriptions,
    )
