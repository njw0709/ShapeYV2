import torch


def jaccard_distance_mm(mat1, mat2):
    mat1 = mat1.float()
    mat2 = mat2.float()

    # Intersection
    intersection = torch.mm(mat1, mat2.T)

    # Sum of rows for mat1 and mat2
    sum_mat1 = mat1.sum(dim=1, keepdim=True)
    sum_mat2 = mat2.sum(dim=1, keepdim=True)

    # Union
    union = (sum_mat1 + sum_mat2.T) - intersection

    # Avoid division by zero
    union = union + (union == 0).float()

    # Jaccard Similarity
    jaccard_similarity = intersection / union

    return jaccard_similarity  # (n, m)


# new function for computing log ratios
def log_ratio_jaccard_per_feature_type(mat1, mat2, density_threshold=0.005):

    mat1 = mat1.float()  # N (number of images) x # features
    mat2 = mat2.float()  # M (number of images) x # features

    # Intersection
    intersection = torch.mm(mat1, mat2.T)  # N X M

    # Sum of rows for mat1 and mat2
    sum_mat1 = mat1.sum(dim=1, keepdim=True)  # N x 1
    sum_mat2 = mat2.sum(dim=1, keepdim=True)  # M x 1

    # if density1 or density2 is zero, return just zero
    density_mat1 = sum_mat1 / mat1.shape[1]  # N x 1
    density_mat2 = sum_mat1 / mat1.shape[1]  # M x 1

    # Union
    union = (sum_mat1 + sum_mat2.T) - intersection  # N x M

    # Avoid division by zero
    union = union + (union == 0).float()

    # Jaccard Similarity
    jaccard_similarity = intersection / union  # N X M

    # compute expected jaccard distances given random two densities
    expected_intersection = torch.mm(density_mat1, density_mat2.T)  # N x M
    expected_union = density_mat1 + density_mat2.T - expected_intersection  # N X M
    expected_jaccard = expected_intersection / expected_union  # N x M

    # log of ratios
    log_of_ratios = torch.log2(jaccard_similarity) - torch.log2(
        expected_jaccard
    )  # N x M

    # if density is zero, make it zero
    mat1_lower_than_threshold = density_mat1 < density_threshold  # N x 1
    mat2_lower_than_threshold = density_mat2 < density_threshold  # M x 1

    log_of_ratios[mat1_lower_than_threshold, :] = 0.0
    log_of_ratios[:, mat2_lower_than_threshold] = 0.0

    return log_of_ratios


def avg_log_ratios_all_feature_type(
    mat1_all_feat_type, mat2_all_feat_type, density_threshold=0.005
):
    # mat1_all_feat_type: # N x num_feat x num_feature_type
    # mat2_all_feat_type: # M x num_feat x num_feature_type

    num_feature_type = mat1_all_feat_type.shape[-1]
    assert num_feature_type == mat2_all_feat_type.shape[-1]

    mat_log_of_ratios = torch.zeros(
        (mat1_all_feat_type.shape[0], mat2_all_feat_type[0], num_feature_type), device="cpu"
    )
    for feat_type_idx in range(num_feature_type):
        log_of_ratio = log_ratio_jaccard_per_feature_type(
            mat1_all_feat_type[:, :, feat_type_idx],
            mat1_all_feat_type[:, :, feat_type_idx],
        )
        mat_log_of_ratios[:, :, feat_type_idx] = log_of_ratio.cpu()

    return mat_log_of_ratios.mean(dim=2)


def weighted_jaccard_distance_mm(mat1, mat2, weights):
    mat1 = mat1.float()
    mat2 = mat2.float()
    weights = weights.float()
    # Weighted Intersection (Sum_i(w_i*e1_i*e2_i))
    intersection = torch.mm((mat1 * weights), mat2.T)

    # Weighted Sum of rows for mat1 and mat2
    sum_mat1 = (mat1 * weights).sum(dim=1, keepdim=True)
    # if applying the same weight across all feats
    if weights.shape[0] == 1:
        sum_mat2 = (mat2 * weights).sum(dim=1, keepdim=True)
        union = (sum_mat1 + sum_mat2.T) - intersection
    # if applying adaptive weights according to the reference feature vector (row)
    else:
        sum_mat2 = torch.mm(weights, mat2.T)
        union = (sum_mat1 + sum_mat2) - intersection

    # Avoid division by zero
    union = union + (union == 0).float()

    # Jaccard Similarity
    jaccard_similarity = intersection / union

    return jaccard_similarity  # (n, m)


def correlation_prenormalized(
    mat1_normalized: torch.Tensor, mat2_normalized: torch.Tensor
) -> torch.Tensor:
    # assume mat1_normalized and mat2_normalized are standardized
    # mat1_normalized: (n, l)
    # mat2_normalized: (m, l)
    # output shape: (n, m)
    output = torch.mm(mat1_normalized, mat2_normalized.T)
    return output


def l2_distance(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    # mat1: (n, l)
    # mat2: (m, l)
    # output shape: (n, m)
    output = torch.cdist(mat1, mat2, p=2)
    return output


def l1_distance(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    # mat1: (n, l)
    # mat2: (m, l)
    # output shape: (n, m)
    output = torch.cdist(mat1, mat2, p=1)
    return output


def cosine_similarity(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    # divide by the norms
    norm1 = torch.norm(mat1, dim=1, keepdim=True)
    norm2 = torch.norm(mat2, dim=1, keepdim=True)
    mat1_normalized = mat1 / norm1
    mat2_normalized = mat2 / norm2
    # compute the dot product
    output = torch.mm(mat1_normalized, mat2_normalized.T)
    return output
