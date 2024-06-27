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


def cosine_similarity(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    # divide by the norms
    norm1 = torch.norm(mat1, dim=1, keepdim=True)
    norm2 = torch.norm(mat2, dim=1, keepdim=True)
    mat1_normalized = mat1 / norm1
    mat2_normalized = mat2 / norm2
    # compute the dot product
    output = torch.mm(mat1_normalized, mat2_normalized.T)
    return output
