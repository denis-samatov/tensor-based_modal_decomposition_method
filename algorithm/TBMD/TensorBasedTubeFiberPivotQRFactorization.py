import numpy as np
import matplotlib.pyplot as plt




class TensorTubeQRDecomposition:
    """
    A class to implement Tensor-based Tube Fiber-pivot QR Factorization 
    and visualize sensor placements.
    """

    def __init__(self, tensor, N):
        """
        Initialize the TensorTubeQR object.

        Parameters:
        - tensor: Input 3D tensor (numpy array) of shape (n1, n2, m).
        - N: Number of iterations (sensors to select).
        """
        if not isinstance(tensor, np.ndarray) or tensor.ndim != 3:
            raise ValueError("tensor must be a 3D numpy array.")
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        
        self.n1, self.n2, self.m = tensor.shape
        
        if N > self.m:
            raise ValueError("N (number of sensors to select) must be less than the size of the third dimension (m).")
        
        self.tensor = tensor
        self.N = N
        self.P = None
        self.Q = None
        self.R = None

    def _compute_householder_vector(self, t):
        """
        Computes the Householder vector to reflect t to a multiple of e1.

        Parameters:
        - t: numpy array, the input vector.
        - d: Index of the column being processed.

        Returns:
        - u: Householder vector as a column vector (numpy array).
        """
        sigma = np.linalg.norm(t, ord=2)
        e1 = np.zeros_like(t)
        e1[0] = 1
        
        sign_t1 = np.sign(t[0]) if t[0] != 0 else 1
        u = t + sign_t1 * sigma * e1
        
        denominator = np.sqrt(2 * sigma * (sigma + np.abs(t[0])))
        if np.abs(denominator) < 1e-10:  # Handle degenerate case
            return np.zeros_like(t)
        
        u = u / denominator
        return u.reshape(-1, 1)

    def factorize(self):
        """
        Perform the Tensor-based Tube Fiber-pivot QR Factorization.

        Returns:
        - P: Permutation matrix of shape (n1 x n2).
        - Q: Orthogonal matrix of shape (m x m).
        - R: Updated tensor after applying Householder transformations.
        """
        P = np.zeros((self.n1, self.n2), dtype=int)
        Q = np.eye(self.m)
        R = self.tensor.copy()
        rejection_domain = set()

        for d in range(self.N):
            # Compute tube norms (l1 norms across the third dimension)
            tube_norms = np.linalg.norm(R.reshape(self.n1, self.n2, -1), axis=2, ord=1)

            # Find the position with the maximum tube norm
            x, y = np.unravel_index(np.argmax(tube_norms), tube_norms.shape)

            # Handle potential duplicate sensor placement
            while (x, y) in rejection_domain and tube_norms[x, y] > 0:
                tube_norms[x, y] = 0
                x, y = np.unravel_index(np.argmax(tube_norms), tube_norms.shape)

            # Update permutation matrix and rejection domain
            P[x, y] = 1
            rejection_domain.add((x, y))

            # Extract the tube fiber
            t = R[x, y, d:]
            if t.size == 0:
                raise ValueError("Tube fiber size is zero, check tensor dimensions or N.")

            # Compute the Householder vector
            u = self._compute_householder_vector(t)

            # Update R and Q
            R[x, y, d:] -= 2 * u.flatten() * (u.flatten() @ R[x, y, d:])
            Q[:, d:] -= 2 * (Q[:, d:] @ u) @ u.T


        self.P = P
        self.Q = Q
        self.R = R

        return P, Q, R

    def visualize_sensor_placement(self):
        """
        Visualize the sensor placement using the permutation matrix P.
        """
        if self.P is None:
            raise ValueError("Sensor placement has not been computed. Run `factorize()` first.")
        
        sensor_positions = np.argwhere(self.P == 1)
        grid_shape = self.P.shape
        
        # Create the plot with the same aspect ratio as the tensor
        fig, ax = plt.subplots(figsize=(grid_shape[1] / 10, grid_shape[0] / 10))  # Adjust the size based on tensor shape
        ax.set_facecolor("black")
        ax.imshow(np.zeros(grid_shape), cmap="gray", origin="upper")  # Background grid
        ax.scatter(sensor_positions[:, 1], sensor_positions[:, 0], c="red", s=20, label="Sensors")  # Plot sensors
        ax.set_title("Sensor Placement", color="white", fontsize=10)
        ax.axis("off")
        ax.set_aspect("auto")  # Ensures the aspect ratio matches the data
        plt.show()

