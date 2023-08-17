import argparse
import os
import shapeymodular.macros.graphing as graphing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph results per case")
    parser.add_argument(
        "--dir",
        type=str,
    )
    args = parser.parse_args()

    graphing.plot_tuning_curves(args.dir)
    graphing.plot_error_panels(args.dir)
    graphing.plot_histogram_with_error_graph(args.dir)
    graphing.plot_nn_classification_error_graph(args.dir)
