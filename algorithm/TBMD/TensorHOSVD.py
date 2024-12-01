import tensorly as tl
import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from typing import Union, List, Dict, Optional


class TuckerDecomposer:
    """
    A class to perform Tucker decomposition on tensors or collections of tensors using TensorLy.
    """
    def __init__(self, tensors: Union[tl.tensor, Dict[str, tl.tensor]],
                 ranks: Optional[Union[int, List[int]]] = None,
                 epsilon: float = 1e-2,
                 random_state: Optional[int] = None):
        """
        Initialize the TuckerDecomposer.

        Parameters:
        - tensors (tl.tensor or dict): Input tensor(s) for decomposition.
        - ranks (int, list, or None): Ranks for Tucker decomposition. Defaults to None (50% per mode).
        - epsilon (float): Tolerance for reconstruction error. Defaults to 1e-2.
        - random_state (int or None): Random seed for reproducibility. Defaults to None.
        """
        self.epsilon = epsilon
        self.random_state = random_state
        self.cores = None
        self.factors = None
        self.reconstructed_tensors = None
        self.reconstruction_errors = None
        self.ranks = ranks

        if isinstance(tensors, dict):
            self.is_collection = True
            self.tensors = {key: tl.tensor(tensor, dtype=tl.float32) for key, tensor in tensors.items()}
        elif isinstance(tensors, np.ndarray) or isinstance(tensors, tl.tensor):
            self.is_collection = False
            self.tensors = tl.tensor(tensors, dtype=tl.float32)
        else:
            raise ValueError("Tensors must be a TensorLy tensor, numpy array, or a dictionary of tensors.")

    def _determine_ranks(self, tensor_shape: List[int]) -> List[int]:
        """
        Determine ranks for Tucker decomposition.

        Parameters:
        - tensor_shape (list): Shape of the input tensor.

        Returns:
        - ranks (list): Ranks for each mode of the tensor.
        """
        if self.ranks is None:
            ranks = [np.min(tensor_shape)] * len(tensor_shape)
        elif isinstance(self.ranks, int):
            ranks = [self.ranks] * len(tensor_shape)
        elif isinstance(self.ranks, list):
            if len(self.ranks) != len(tensor_shape):
                raise ValueError("Ranks list must have the same length as tensor modes.")
            ranks = self.ranks
        else:
            raise ValueError("Ranks must be None, an integer, or a list of integers.")
        return ranks

    def decompose(self) -> None:
        """
        Perform Tucker decomposition on the tensor or collection of tensors.
        """
        if self.is_collection:
            self.cores = {}
            self.factors = {}
            for key, tensor in self.tensors.items():
                tensor_shape = tensor.shape
                ranks = self._determine_ranks(list(tensor_shape))
                core, factors = tucker(tensor, rank=ranks, random_state=self.random_state)
                self.cores[key] = core
                self.factors[key] = factors
        else:
            tensor_shape = self.tensors.shape
            ranks = self._determine_ranks(list(tensor_shape))
            self.core, self.factors = tucker(self.tensors, rank=ranks, random_state=self.random_state)

    def reconstruct(self) -> None:
        """
        Reconstruct tensors from their Tucker decomposition and calculate reconstruction errors.
        """
        if self.is_collection:
            self.reconstructed_tensors = {}
            self.reconstruction_errors = {}
            for key in self.tensors.keys():
                reconstructed = tucker_to_tensor((self.cores[key], self.factors[key]))
                error = tl.norm(self.tensors[key] - reconstructed) / tl.norm(self.tensors[key])
                self.reconstructed_tensors[key] = reconstructed
                self.reconstruction_errors[key] = float(error)
        else:
            self.reconstructed_tensor = tucker_to_tensor((self.core, self.factors))
            self.reconstruction_error = float(
                tl.norm(self.tensors - self.reconstructed_tensor) / tl.norm(self.tensors)
            )

    def visualize(self, subjects: Optional[List[str]] = None) -> None:
        """
        Visualize the original and reconstructed tensors.
        For collections, specify the `subjects` to visualize.
        """
        if self.is_collection:
            if subjects is None:
                subjects = self.tensors.keys()
            for subject in subjects:
                if len(self.tensors[subject].shape) != 3:
                    raise ValueError(f"Tensor for subject {subject} is not 3D and cannot be visualized.")
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(self.tensors[subject][:, :, self.tensors[subject].shape[2] // 2], cmap="gray")
                plt.title(f"Original (Subject {subject})")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(self.reconstructed_tensors[subject][:, :, self.reconstructed_tensors[subject].shape[2] // 2], cmap="gray")
                plt.title(f"Reconstructed (Subject {subject})")
                plt.axis("off")
                plt.show()
        else:
            if len(self.tensors.shape) != 3:
                raise ValueError("Tensor is not 3D and cannot be visualized.")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(self.tensors[:, :, self.tensors.shape[2] // 2], cmap="gray")
            plt.title("Original Tensor")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(self.reconstructed_tensor[:, :, self.reconstructed_tensor.shape[2] // 2], cmap="gray")
            plt.title("Reconstructed Tensor")
            plt.axis("off")
            plt.show()

    def get_cores(self) -> Union[tl.tensor, Dict[str, tl.tensor]]:
        """
        Returns the core tensor(s) from the Tucker decomposition.
        """
        if self.is_collection:
            if not self.cores:
                raise ValueError("Decomposition not performed. Call `decompose()` first.")
            return self.cores
        if self.core is None:
            raise ValueError("Decomposition not performed. Call `decompose()` first.")
        return self.core

    def get_factors(self) -> Union[List[tl.tensor], Dict[str, List[tl.tensor]]]:
        """
        Returns the factor matrices from the Tucker decomposition.
        """
        if self.is_collection:
            if not self.factors:
                raise ValueError("Decomposition not performed. Call `decompose()` first.")
            return self.factors
        if self.factors is None:
            raise ValueError("Decomposition not performed. Call `decompose()` first.")
        return self.factors

    def get_reconstruction_errors(self) -> Dict[str, float]:
        """
        Returns reconstruction errors for all tensors in a collection.
        """
        if not self.is_collection:
            raise ValueError("Reconstruction errors are available only for tensor collections.")
        if not self.reconstruction_errors:
            raise ValueError("Reconstruction not performed. Call `reconstruct()` first.")
        return self.reconstruction_errors

    def get_reconstruction_error(self) -> float:
        """
        Returns the reconstruction error for a single tensor.
        """
        if self.is_collection:
            raise ValueError("Use `get_reconstruction_errors` for tensor collections.")
        if self.reconstruction_error is None:
            raise ValueError("Reconstruction not performed. Call `reconstruct()` first.")
        return self.reconstruction_error

    def set_ranks(self, ranks: Union[int, List[int]]) -> None:
        """
        Sets the ranks for Tucker decomposition.
        """
        if not (isinstance(ranks, (int, list)) or ranks is None):
            raise ValueError("Ranks must be an integer, a list, or None.")
        self.ranks = ranks
