import h5py
import numpy as np
import torch
from typing import Sequence, Union, Tuple
import typing


class H5ConcatHandler:
    def __init__(
        self, file_paths, dataset_name, to_tensor=False, device=None, return_dtype=None
    ):
        """Initialize by opening HDF5 files and retrieving dataset shapes."""
        self.file_paths: Sequence[str] = file_paths
        self.dataset_name: str = dataset_name
        self.to_tensor: bool = to_tensor
        self.datasets: Sequence = []
        self.dataset_sizes: Sequence = []
        self.total_size: int = 0
        self.dataset_shape: Union[Tuple[int], Sequence[int]] = None  # type: ignore
        self.concatenated_shape: Tuple = None  # type: ignore
        self.transposed = False
        self.axes: Union[None, Sequence[int]] = (
            None  # For custom transpose with specific axes
        )
        self.device: Union[None, str] = device
        self.return_dtype: Union[None, torch.dtype] = None

        if self.to_tensor:
            if self.device is None:
                self.device = "cpu"

        # Open files and get datasets
        for file_path in file_paths:
            file = h5py.File(file_path, "r")
            dataset = file[dataset_name]
            assert isinstance(dataset, h5py.Dataset), "key: {} must load a h5py Dataset"
            self.datasets.append(dataset)
            self.dataset_sizes.append(
                len(dataset)
            )  # Assuming the 1st dimension is the number of rows
            self.total_size += len(dataset)

            # Ensure all datasets have the same shape except for the first axis
            if self.dataset_shape is None:
                self.dataset_shape = dataset.shape
            else:
                if dataset.shape[1:] != self.dataset_shape[1:]:
                    raise ValueError(
                        "All datasets must have the same shape except for the first axis."
                    )
        self.concatenated_shape = (self.total_size,) + tuple(self.dataset_shape[1:])

    @property
    def shape(self):
        """Return the virtual concatenated shape of all datasets."""
        if self.transposed and self.axes:
            # Adjust the shape based on the custom axes permutation
            return tuple(self.concatenated_shape[ax] for ax in self.axes)
        elif self.transposed:
            # Return the shape with all axes reversed
            return tuple(reversed(self.concatenated_shape))
        else:
            # Original shape with total_size as the first dimension
            return self.concatenated_shape

    @property
    def T(self):
        """Return a transposed view of the dataset (transpose all axes)."""
        handler = H5ConcatHandler(
            self.file_paths, self.dataset_name, to_tensor=self.to_tensor
        )
        handler.transposed = True
        return handler

    def transpose(self, *axes):
        """Return a view of the dataset with the specified axes transposed."""
        if len(axes) == 0:
            axes = tuple(range(len(self.concatenated_shape)))[
                ::-1
            ]  # Default to reversing axes
        elif len(axes) != len(self.concatenated_shape):
            raise ValueError("Axes don't match dataset dimensions.")

        handler = H5ConcatHandler(
            self.file_paths, self.dataset_name, to_tensor=self.to_tensor
        )
        handler.transposed = True
        handler.axes = axes
        return handler

    def __len__(self):
        """Return the total size of the first axis (after potential transpose)."""
        if self.transposed and self.axes:
            return self.concatenated_shape[self.axes[0]]  # type: ignore
        elif self.transposed:
            return self.concatenated_shape[
                -1
            ]  # After transpose, the first axis size is from the last axis
        else:
            return self.total_size

    def _find_dataset(self, idx):
        """Find which dataset and local index to use for the given global index."""
        cumulative_size = 0
        for i, size in enumerate(self.dataset_sizes):
            if cumulative_size <= idx < cumulative_size + size:
                return self.datasets[i], idx - cumulative_size
            cumulative_size += size
        raise IndexError("Index out of bounds")

    def __getitem__(self, idx):
        """Lazy load the data from the appropriate dataset based on index or list/slice of indices."""
        if isinstance(idx, tuple):
            # Handle multi-indexing like [1, 5:10]
            return self._handle_multi_indexing(idx)
        elif isinstance(idx, (int, slice)):
            # Handle simple indexing or slicing
            return self._handle_simple_indexing(idx)
        elif isinstance(idx, list):
            # Handle list of indices
            return self._handle_list_indexing(idx)
        else:
            raise TypeError("Invalid index type")

    def _handle_simple_indexing(self, idx):
        """Handle simple integer or slice indexing."""
        if isinstance(idx, slice):
            return self._handle_slice(idx)

        if idx < 0:
            idx += self.total_size
        if idx >= self.total_size or idx < 0:
            raise IndexError("Index out of range")

        dataset, local_idx = self._find_dataset(idx)
        data = dataset[local_idx]
        data = typing.cast(np.ndarray, data)
        # Apply transpose if necessary
        if self.transposed:
            data = data.T if self.axes is None else np.transpose(data, self.axes)

        # Convert to PyTorch tensor if needed
        if self.to_tensor:
            if self.return_dtype is not None:
                data = torch.tensor(data, device=self.device, dtype=self.return_dtype)
            else:
                data = torch.tensor(data, device=self.device)
        return data

    def _handle_slice(self, s):
        """Handle slicing and concatenate results from different datasets."""
        indices = range(*s.indices(self.total_size))
        concatenated = np.concatenate([self[idx] for idx in indices])

        # Apply transpose if necessary
        if self.transposed:
            concatenated = (
                concatenated.T
                if self.axes is None
                else np.transpose(concatenated, self.axes)
            )

        # Convert to PyTorch tensor if needed
        if self.to_tensor:
            if self.return_dtype is not None:
                return torch.tensor(
                    concatenated, device=self.device, dtype=self.return_dtype
                )
            else:
                return torch.tensor(concatenated, device=self.device)

        return concatenated

    def _handle_list_indexing(self, idx_list):
        """Handle list of indices, like [1, 5, 7]."""
        results = [self._handle_simple_indexing(idx) for idx in idx_list]
        concatenated = np.concatenate(results)

        # Apply transpose if necessary
        if self.transposed:
            concatenated = (
                concatenated.T
                if self.axes is None
                else np.transpose(concatenated, self.axes)
            )

        # Convert to PyTorch tensor if needed
        if self.to_tensor:
            if self.return_dtype is not None:
                return torch.tensor(
                    concatenated, device=self.device, dtype=self.return_dtype
                )
            else:
                return torch.tensor(concatenated, device=self.device)

        return concatenated

    def _handle_multi_indexing(self, idx_tuple):
        """Handle multiple indices like [1, 5:10]."""
        results = []
        for idx in idx_tuple:
            if isinstance(idx, int):
                results.append(self._handle_simple_indexing(idx))
            elif isinstance(idx, slice):
                results.append(self._handle_slice(idx))
            elif isinstance(idx, list):
                results.append(self._handle_list_indexing(idx))
            else:
                raise TypeError("Invalid index type")

        # Concatenate the results
        concatenated = np.concatenate(results)

        # Apply transpose if necessary
        if self.transposed:
            concatenated = (
                concatenated.T
                if self.axes is None
                else np.transpose(concatenated, self.axes)
            )

        # Convert to PyTorch tensor if needed
        if self.to_tensor:
            if self.return_dtype is not None:
                return torch.tensor(
                    concatenated, device=self.device, dtype=self.return_dtype
                )
            else:
                return torch.tensor(concatenated, device=self.device)

        return concatenated

    def close(self):
        """Close all open HDF5 files."""
        for dataset in self.datasets:
            dataset.file.close()

    def __del__(self):
        """Ensure all files are closed when the object is destroyed."""
        self.close()


# # Example Usage
# file_paths = ['file1.h5', 'file2.h5', 'file3.h5']

# # Option to return NumPy arrays
# handler_numpy = H5ConcatHandler(file_paths, 'dataset_name', to_tensor=False)
# print(handler_numpy[0])  # Output: NumPy array

# # Option to return PyTorch tensors
# handler_tensor = H5ConcatHandler(file_paths, 'dataset_name', to_tensor=True)
# print(handler_tensor[0])  # Output: PyTorch tensor

# # Get the virtual shape of the concatenated dataset
# print("Shape:", handler_tensor.shape)

# handler_tensor.close()  # Close files manually (or when object is deleted)
