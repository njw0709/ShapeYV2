import argparse
import os
import shapeymodular.macros.compute_distance_torch as compute_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute distances between representation vectors for images in ShapeY"
    )
    parser.add_argument(
        "--dir",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="distances-Jaccard.mat",
    )
    parser.add_argument(
        "--replace",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--imgnames",
        type=str,
        default="imgnames_all.txt",
    )
    parser.add_argument(
        "--threshold_file",
        type=str,
        default="thresholds.mat",
    )
    parser.add_argument(
        "--save_thresholded_features",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--thresholded_features_save_name",
        type=str,
        default="thresholded_features.h5",
    )
    parser.add_argument(
        "--row_segment_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--col_segment_size",
        type=int,
        default=1000,
    )

    args = parser.parse_args()
    # compute distances
    if os.path.exists(os.path.join(args.dir, args.output_name)) and args.replace:
        print("distances file already exists in {}. Removing...".format(dir))
        os.remove(os.path.join(args.dir, args.output_name))
    elif os.path.exists(os.path.join(args.dir, args.output_name)) and not args.replace:
        print("distances file already exists in {}. Skipping...".format(dir))
        exit()
    else:
        print("Computing distances...")

    # check if necessary files exist
    compute_distance.check_and_prep_for_distance_computation(
        args.dir,
        args.imgnames,
        args.imgnames,
        threshold_file=args.threshold_file,
    )
    # load and threshold features
    imgnames, features = compute_distance.get_thresholded_features(
        args.dir,
        threshold_file=args.threshold_file,
        save_thresholded_features=args.save_thresholded_features,
        save_name=args.thresholded_features_save_name,
    )

    # compute jaccard distance
    compute_distance.compute_jaccard_distance(
        args.dir,
        features,
        args.output_name,
        args.row_segment_size,
        args.col_segment_size,
        gpu_index=args.gpu,
    )
