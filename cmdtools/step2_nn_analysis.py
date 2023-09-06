import os
import argparse
import shapeymodular.macros.nn_batch as nn_batch

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exclusion analysis")
    parser.add_argument(
        "--dir",
        type=str,
    )

    parser.add_argument(
        "--distance_file",
        type=str,
        default="distance-Jaccard.mat",
    )

    parser.add_argument(
        "--row_imgnames",
        type=str,
        default="imgnames_pw_series.txt",
    )

    parser.add_argument(
        "--col_imgnames",
        type=str,
        default="imgnames_all.txt",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="analysis_results.h5",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default="pw",
    )

    args = parser.parse_args()

    nn_batch.run_exclusion_analysis(
        args.dir,
        distance_file=args.distance_file,
        row_imgnames=args.row_imgnames,
        col_imgnames=args.col_imgnames,
        save_name=args.save_name,
        axes=args.axes,
    )
