import argparse
import os
import shapeymodular.macros.compute_distance as compute_distance


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
        "--lsh",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--lsh_num_band",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lsh_num_hash",
        type=int,
        default=300,
    )
    args = parser.parse_args()
    # compute distances
    if os.path.exists(os.path.join(args.dir, args.output_name)) and args.replace:
        print("distances file already exists in {}. Removing...".format(dir))
        os.remove(os.path.join(args.dir, "distances-Jaccard.mat"))
    elif os.path.exists(os.path.join(args.dir, args.output_name)) and not args.replace:
        print("distances file already exists in {}. Skipping...".format(dir))
        exit()
    else:
        print("Computing distances...")

    if args.lsh:
        lsh_configs = {
            "lshHashName": "minHash",
            "lshNumBands": args.lsh_num_band,
            "lshNumHashes": args.lsh_num_hash,
        }
        compute_distance.check_and_prep_for_distance_computation(
            args.dir, lsh_configs=lsh_configs
        )
        distance_configs = {
            "lsh": True,
            "neighbors-dist-in": "imgnames_pw_series.txt",
            "neighbors-dist-out": "lsh-distances-Jaccard.mat",
            "distance-name": "Jaccard",
            "lsh-neighbors-out": "lsh-neighbors.txt",
        }
        compute_distance.compute_distance(
            args.dir, gpunum=args.gpu, distance_configs=distance_configs
        )
    else:
        compute_distance.check_and_prep_for_distance_computation(args.dir)
        distance_configs = {
            "lsh": False,
            "pairwise-dist-in": "imgnames_pw_series.txt",
            "pairwise-dist-out": "distances-Jaccard.mat",
            "distance-name": "Jaccard",
        }
        compute_distance.compute_distance(
            args.dir, gpunum=args.gpu, distance_configs=distance_configs
        )
