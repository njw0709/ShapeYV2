import os
import shapeymodular.macros.compute_distance as compute_distance


# get all directories to run
all_features_directories = [
    "/home/francis/nineCasesToRun/kernels12_poolingMap1Left1Right/features-results-l2p0,1",
    "/home/francis/nineCasesToRun/kernels16_poolingMap0Left1Right/features-results-l2p0,1",
    "/home/francis/nineCasesToRun/kernels16_poolingMap1Left1Right/features-results-l2p0,1",
]
# base_dir = "/home/francis/nineCasesToRun/"
# datadirs = os.listdir(base_dir)
# datadirs.sort()
# datadirs = datadirs[:5]
# print(datadirs)
# for dir in datadirs:
#     features_dir = [
#         os.path.join(base_dir, dir, fd)
#         for fd in os.listdir(os.path.join(base_dir, dir))
#         if "features-results-" in fd
#     ]
#     all_features_directories.extend(features_dir)


# compute distances
for i, dir in enumerate(all_features_directories):
    if os.path.exists(os.path.join(dir, "distances-Jaccard.mat")):
        print("distances file already exists in {}. Removing...".format(dir))
        os.remove(os.path.join(dir, "distances-Jaccard.mat"))
    compute_distance.check_and_prep_for_distance_computation(dir)
    compute_distance.compute_distance(dir)
