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
    args = parser.parse_args()

    nn_batch.run_exclusion_analysis(args.dir)
