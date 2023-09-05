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
