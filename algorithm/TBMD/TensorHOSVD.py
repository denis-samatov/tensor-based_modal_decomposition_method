import numpy as np
import matplotlib.pyplot as plt

from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor




class TuckerDecomposer:
    def __init__(self, tensors, ranks=None, epsilon=1e-2, random_state=None):
        """
        Initialize the TuckerDecomposer.

        Parameters:
        tensors (numpy.ndarray or dict): Input tensor(s) for decomposition.
        ranks (int, list, or None): Ranks for Tucker decomposition. Defaults to None (50% per mode).
        epsilon (float): Tolerance for reconstruction error. Defaults to 1e-2.
        random_state (int or None): Random seed for reproducibility. Defaults to None.
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
            self.tensors = tensors
            for tensor in self.tensors.values():
                if not isinstance(tensor, np.ndarray):
                    raise ValueError("All tensors in the dictionary must be numpy arrays.")
        elif isinstance(tensors, np.ndarray):
            self.is_collection = False
            self.tensors = tensors
        else:
            raise ValueError("Tensors must be a numpy array or a dictionary of numpy arrays.")
    
    def _determine_ranks(self, tensor_shape):
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

    
    def decompose(self):
        """
        Perform Tucker decomposition on the tensor or collection of tensors.
        """
        try:
            if self.is_collection:
                self.cores = {}
                self.factors = {}
                for key, tensor in self.tensors.items():
                    tensor_shape = tensor.shape
                    ranks = self._determine_ranks(tensor_shape)
                    core, factors = tucker(tensor, rank=ranks, n_iter_max=100, random_state=self.random_state)
                    self.cores[key] = core
                    self.factors[key] = factors
            else:
                tensor_shape = self.tensors.shape
                ranks = self._determine_ranks(tensor_shape)
                self.core, self.factors = tucker(self.tensors, rank=ranks, n_iter_max=100, random_state=self.random_state)
        except Exception as e:
            raise RuntimeError(f"Error during decomposition: {e}")

    
    def reconstruct(self):
        """
        Reconstruct tensors from their Tucker decomposition and calculate reconstruction errors.
        """
        try:
            if self.is_collection:
                self.reconstructed_tensors = {}
                self.reconstruction_errors = {}
                for key in self.tensors.keys():
                    reconstructed = tucker_to_tensor((self.cores[key], self.factors[key]))
                    error = np.linalg.norm(self.tensors[key] - reconstructed) / np.linalg.norm(self.tensors[key])
                    self.reconstructed_tensors[key] = reconstructed
                    self.reconstruction_errors[key] = error
            else:
                self.reconstructed_tensor = tucker_to_tensor((self.core, self.factors))
                self.reconstruction_error = np.linalg.norm(self.tensors - self.reconstructed_tensor) / np.linalg.norm(self.tensors)
        except Exception as e:
            raise RuntimeError(f"Error during reconstruction: {e}")

    
    def visualize(self, subjects=None):
        """
        Visualize the original and reconstructed tensors.
        For collections, specify the `subjects` to visualize.
        """
        try:
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
        except Exception as e:
            raise RuntimeError(f"Error during visualization: {e}")

    
    def get_cores(self):
        """
        Returns the core tensor(s) from the Tucker decomposition.
        For a single tensor, returns the core tensor directly.
        For collections, returns a dictionary of core tensors.
        """
        if self.is_collection:
            if not self.cores:
                raise ValueError("Decomposition not performed. Call `decompose()` first.")
            return self.cores
        if self.core is None:
            raise ValueError("Decomposition not performed. Call `decompose()` first.")
        return self.core

    def get_factors(self):
        """
        Returns the factor matrices from the Tucker decomposition.
        For a single tensor, returns the factor matrices directly.
        For collections, returns a dictionary of factor matrices.
        """
        if self.is_collection:
            if not self.factors:
                raise ValueError("Decomposition not performed. Call `decompose()` first.")
            return self.factors
        if self.factors is None:
            raise ValueError("Decomposition not performed. Call `decompose()` first.")
        return self.factors

    def get_reconstruction_errors(self):
        """
        Returns reconstruction errors for all tensors in a collection.
        """
        if not self.is_collection:
            raise ValueError("Reconstruction errors are available only for tensor collections.")
        if not self.reconstruction_errors:
            raise ValueError("Reconstruction not performed. Call `reconstruct()` first.")
        return self.reconstruction_errors

    def get_reconstruction_error(self):
        """
        Returns the reconstruction error for a single tensor.
        """
        if self.is_collection:
            raise ValueError("Use `get_reconstruction_errors` for tensor collections.")
        if self.reconstruction_error is None:
            raise ValueError("Reconstruction not performed. Call `reconstruct()` first.")
        return self.reconstruction_error

    def get_total_reconstruction_error(self):
        """
        Returns the total reconstruction error for all tensors in a collection
        or the reconstruction error for a single tensor.
        """
        if self.is_collection:
            if not self.reconstruction_errors:
                raise ValueError("Reconstruction not performed. Call `reconstruct()` first.")
            return sum(self.reconstruction_errors.values())
        if self.reconstruction_error is None:
            raise ValueError("Reconstruction not performed. Call `reconstruct()` first.")
        return self.reconstruction_error

    def set_ranks(self, ranks):
        """
        Sets the ranks for Tucker decomposition.
        """
        if not (isinstance(ranks, (int, list)) or ranks is None):
            raise ValueError("Ranks must be an integer, a list, or None.")
        self.ranks = ranks

