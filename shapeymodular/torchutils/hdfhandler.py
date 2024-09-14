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
        self.axes_ordering = tuple(range(len(self.concatenated_shape)))

    @property
    def shape(self):
        """Return the virtual concatenated shape of all datasets."""
        if self.transposed:
            # Adjust the shape based on the custom axes permutation
            return tuple(self.concatenated_shape[ax] for ax in self.axes_ordering)
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
        handler.axes_ordering = self.axes_ordering[::-1]
        return handler

    def transpose(self, *axes):
        """Return a view of the dataset with the specified axes transposed."""
        if len(axes) == 0:
            axes = self.axes_ordering[::-1]  # Default to reversing axes
        elif len(axes) != len(self.concatenated_shape):
            raise ValueError("Axes don't match dataset dimensions.")

        handler = H5ConcatHandler(
            self.file_paths, self.dataset_name, to_tensor=self.to_tensor
        )
        handler.transposed = True
        axes = [self.axes_ordering[i] for i in axes]
        handler.axes_ordering = axes
        return handler

    def __len__(self):
        """Return the total size of the first axis (after potential transpose)."""
        if self.transposed:
            return self.concatenated_shape[self.axes_ordering[0]]  # type: ignore
        else:
            return self.total_size

    def _find_dataset(self, axis_zero_idx):
        """Find which dataset and local index to use for the given global index."""
        cumulative_size = 0
        for i, size in enumerate(self.dataset_sizes):
            if cumulative_size <= axis_zero_idx < cumulative_size + size:
                return self.datasets[i], axis_zero_idx - cumulative_size
            cumulative_size += size
        raise IndexError("Index out of bounds")

    def __getitem__(self, idx):
        """Lazy load the data from the appropriate dataset based on index or list/slice of indices."""
        if isinstance(idx, tuple):
            # Handle multi-indexing like [1, 5:10]
            return self._handle_multi_indexing(idx)
        elif isinstance(idx, (int, slice)):
            # Handle simple indexing or slicing
            return self._handle_simple_indexing(idx, 0)
        elif isinstance(idx, list):
            # Handle list of indices
            return self._handle_list_indexing(idx, 0)
        else:
            raise TypeError("Invalid index type")

    def _handle_simple_indexing(self, idx, current_naxis):
        """Handle simple integer or slice indexing."""
        original_naxis = self.axes_ordering[current_naxis]
        axis_size = self.concatenated_shape[original_naxis]
        if isinstance(idx, slice):
            return self._handle_slice(idx, current_naxis)
        if idx < 0:
            idx += axis_size
        if idx >= axis_size or idx < 0:
            raise IndexError("Index out of range")
        if original_naxis == 0:
            dataset, local_idx = self._find_dataset(idx)
            data = dataset[local_idx]
            data = typing.cast(np.ndarray, data)
        else:
            # in case axis = 1
            data_list = []
            indexing = []
            for ax_idx in range(len(self.concatenated_shape)):
                if ax_idx == original_naxis:  # original axis ordering
                    indexing.append(idx)
                else:
                    indexing.append(slice(None))
            indexing = tuple(indexing)
            for dataset in self.datasets:
                data_list.append(dataset[indexing])
            data = np.concatenate(data_list)  # concatenate results

        # Apply transpose if necessary
        if self.transposed:
            data = np.transpose(data, self.axes_ordering)

        # Convert to PyTorch tensor if needed
        if self.to_tensor:
            if self.return_dtype is not None:
                data = torch.tensor(data, device=self.device, dtype=self.return_dtype)
            else:
                data = torch.tensor(data, device=self.device)
        return data

    def _handle_slice(self, s, current_naxis):
        """Handle slicing and concatenate results from different datasets."""
        original_naxis = self.axes_ordering[current_naxis]
        axis_size = self.concatenated_shape[original_naxis]

        indices = range(*s.indices(axis_size))

        concatenated = np.concatenate(
            [
                self._handle_simple_indexing(idx, current_naxis) for idx in indices
            ],  # returns in transposed axis
            axis=current_naxis,
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

    def _handle_list_indexing(self, idx_list, current_naxis):
        """Handle list of indices, like [1, 5, 7]."""
        results = [
            self._handle_simple_indexing(idx, current_naxis) for idx in idx_list
        ]  # returns in transposed axis
        concatenated = np.concatenate(results, axis=current_naxis)

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
        idx_tuple_rearranged = [
            idx_tuple[naxis] for naxis in self.axes_ordering
        ]  # rearranged to original order
        idx_tuple_common = idx_tuple_rearranged[1:]

        # convert the first axis to local index
        idx_first = idx_tuple_rearranged[0]
        if isinstance(idx_first, slice):
            idx_first = list(range(*idx_first.indices(self.concatenated_shape[0])))

        if isinstance(idx_first, list):
            results = []
            for idx in idx_first:
                dataset, local_idx = self._find_dataset(idx)
                idx_tuple_local = (local_idx,) + tuple(idx_tuple_common)
                results.append(dataset[idx_tuple_local])
        # Concatenate the results
        concatenated = np.concatenate(results)

        # Apply transpose if necessary
        if self.transposed:
            concatenated = np.transpose(concatenated, self.axes_ordering)

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
