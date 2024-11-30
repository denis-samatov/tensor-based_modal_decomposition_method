import numpy as np




class TensorOperations:
    """
    A class to perform various tensor operations.
    """
    
    def __init__(self, tensor):
        """
        Initialize the TensorOperations with a tensor.
        
        Parameters:
        - tensor: numpy ndarray of shape (I1, I2, ..., IN)
        """
        if not isinstance(tensor, np.ndarray):
            raise ValueError("Tensor must be a numpy ndarray.")
        if tensor.ndim < 3:
            raise ValueError("Tensor must be at least 3-dimensional.")
        self.tensor = tensor
    
    def mode_n_product(self, matrix, mode):
        """
        Computes the mode-n product of the tensor with a matrix along the specified mode.
        
        Parameters:
        - matrix: numpy ndarray of shape (J, In)
        - mode: int, the mode along which to compute the product (0-based index)
        
        Returns:
        - result_tensor: numpy ndarray of shape (I1, ..., I_{n-1}, J, I_{n+1}, ..., IN)
        """
        N = self.tensor.ndim
        if mode < 0 or mode >= N:
            raise ValueError(f"Mode should be between 0 and {N-1}, got mode={mode}")
        tensor_perm = np.moveaxis(self.tensor, mode, 0)
        In = tensor_perm.shape[0]
        remaining_shape = tensor_perm.shape[1:]
        tensor_mat = tensor_perm.reshape(In, -1)
        result_mat = matrix @ tensor_mat
        result_shape = (matrix.shape[0],) + remaining_shape
        result_tensor = result_mat.reshape(result_shape)
        result_tensor = np.moveaxis(result_tensor, 0, mode)
        return result_tensor
    
    def tube_fiber_wise_dot_product(self):
        """
        Computes the tube fiber-wise dot product of the tensor with itself.
        
        Returns:
        - C: numpy ndarray of shape (I, J), where C[i, j] = sum_k tensor[i, j, k]^2
        """
        if self.tensor.ndim != 3:
            raise ValueError("Tube fiber-wise dot product requires a 3D tensor.")
        C = np.sum(self.tensor * self.tensor, axis=2)
        return C
    
    def mode_3_fiberwise_product(self, B):
        """
        Performs the mode-3 fiber-wise product of the tensor with matrix B.
        
        Parameters:
        - B: numpy ndarray of shape (K, K)
        
        Returns:
        - result: numpy ndarray of shape (I, J, K)
        """
        I, J, K = self.tensor.shape
        if B.shape != (K, K):
            raise ValueError(f"Matrix B must have shape ({K}, {K})")
        X_reshaped = self.tensor.reshape(-1, K)
        result_reshaped = X_reshaped @ B.T
        result = result_reshaped.reshape(I, J, K)
        return result
    
    def frontal_slicewise_product(self, v):
        """
        Performs the frontal slice-wise product of the tensor with vector v.
        
        Parameters:
        - v: numpy ndarray of shape (K,)
        
        Returns:
        - result: numpy ndarray of shape (I, J)
        """
        I, J, K = self.tensor.shape
        if v.shape[0] != K:
            raise ValueError(f"Vector v must have length {K}")
        result = np.zeros((I, J))
        for k in range(K):
            result += self.tensor[:, :, k] * v[k]
        return result
    
    def frontal_slicewise_transpose(self):
        """
        Performs the frontal slice-wise transpose of the tensor.
        
        Returns:
        - result: numpy ndarray of shape (I, J, K)
        """
        result = self.tensor[:, :, ::-1]
        return result
    
    def __str__(self):
        """
        Returns a string representation of the tensor shape.
        """
        return f"TensorOperations instance with tensor of shape {self.tensor.shape}"