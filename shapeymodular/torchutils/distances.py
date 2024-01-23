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
