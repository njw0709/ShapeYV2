import pytest
import torch
import shapeymodular.torchutils as torchutils


def test_jaccard_distance_mm():
    # Test Case 1: Square matrices
    mat1 = torch.Tensor([[1, 1], [0, 1]])
    mat2 = torch.Tensor([[1, 0], [1, 1]])
    expected_output = torch.Tensor([[0.5, 1.0], [0.0, 0.5]])
    output = torchutils.jaccard_distance_mm(mat1, mat2)
    torch.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    # Test Case 2: Rectangular matrices
    mat1 = torch.Tensor([[1, 1, 1], [0, 0, 1]])
    mat2 = torch.Tensor([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    expected_output = torch.Tensor([[0.6667, 1.0, 0.3333], [0.5, 0.3333, 0.0]])
    output = torchutils.jaccard_distance_mm(mat1, mat2)
    torch.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    # Test Case 3: Zero vector handling
    mat1 = torch.Tensor([[0, 0], [0, 0]])
    mat2 = torch.Tensor([[0, 0], [0, 0]])
    expected_output = torch.Tensor(
        [[0.0, 0.0], [0.0, 0.0]]
    )  # Union and intersection both are zero
    output = torchutils.jaccard_distance_mm(mat1, mat2)
    torch.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    # Test Case 4: Single element vectors
    mat1 = torch.Tensor([[1]])
    mat2 = torch.Tensor([[1]])
    expected_output = torch.Tensor([[1.0]])  # Union and intersection both are 1
    output = torchutils.jaccard_distance_mm(mat1, mat2)
    torch.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
