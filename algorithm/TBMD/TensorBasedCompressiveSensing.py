import numpy as np
from typing import Union

class TensorCompressiveSensing:
    """
    A class to implement the tensor-based compressive sensing algorithm.
    """
    def __init__(self, A: np.ndarray, P: np.ndarray, Y: np.ndarray,
                 max_iter: int, epsilon: float, lambd: float, delta_0: float, delta_max: float):
        """
        Initialize the TensorCompressiveSensing class.

        Parameters:
        - A: Tensor dictionary of shape (I, J, W)
        - P: Sensor selection matrix of shape (I, J)
        - Y: Measurement matrix of shape (I, J)
        - max_iter: Maximum number of iterations
        - epsilon: Threshold value for the soft-thresholding operation
        - lambd: Regularization parameter λ
        - delta_0: Initial value for the penalty parameter δ
        - delta_max: Maximum value for δ
        """
        self.A = np.array(A, dtype=np.float64)
        self.P = np.array(P, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.lambd = lambd
        self.delta_0 = delta_0
        self.delta_max = delta_max

        # Validate input dimensions
        self.I, self.J, self.W = self.A.shape
        assert self.P.shape == (self.I, self.J), "P must have dimensions (I, J)"
        assert self.Y.shape == (self.I, self.J), "Y must have dimensions (I, J)"

        # Initialize variables
        self.x_n = np.zeros((self.W, 1), dtype=np.float64)
        self.d_n = np.zeros((self.W, 1), dtype=np.float64)
        self.p_n = np.zeros((self.W, 1), dtype=np.float64)
        self.delta_n = self.delta_0

        # Preprocess inputs
        self.A_sensors, self.Y_sensors = self._process_inputs()

        # Precompute matrices
        self.A_T_A = self.A_sensors.T @ self.A_sensors  # Shape: (W, W)
        self.A_T_Y = self.A_sensors.T @ self.Y_sensors  # Shape: (W, 1)
        self.I_W = np.eye(self.W, dtype=np.float64)     # Identity matrix of shape (W, W)

    def _process_inputs(self):
        """
        Apply the sensor selection matrix P to tensor A and matrix Y.
        Returns the data corresponding to the sensors.

        Returns:
        - A_sensors: Selected sensor matrix from tensor A, shape (N, W)
        - Y_sensors: Measurements corresponding to the sensors, shape (N, 1)
        """
        P_expanded = self.P[:, :, np.newaxis]  # Expand P to shape (I, J, 1)
        A_masked = self.A * P_expanded         # Shape: (I, J, W)
        Y_masked = self.Y * self.P             # Shape: (I, J)

        # Reshape tensors for processing
        A_unfold = A_masked.reshape(-1, self.W)  # Shape: (I * J, W)
        Y_unfold = Y_masked.reshape(-1, 1)      # Shape: (I * J, 1)
        P_vec = self.P.flatten()                # Shape: (I * J,)

        # Find indices of active sensors
        sensor_indices = np.where(P_vec != 0)[0]

        # Select rows corresponding to sensors
        A_sensors = A_unfold[sensor_indices, :]  # Shape: (N, W)
        Y_sensors = Y_unfold[sensor_indices]     # Shape: (N, 1)
        return A_sensors, Y_sensors

    def solve(self) -> np.ndarray:
        """
        Perform the iterative process to recover the sparse coefficient vector.

        Returns:
        - x_hat: Recovered sparse coefficient vector of shape (W,)
        """
        for n in range(self.max_iter):
            # Update x_n
            left_matrix = self.A_T_A + self.delta_n * self.I_W  # Shape: (W, W)
            right_vector = self.A_T_Y + self.delta_n * (self.d_n - self.p_n)  # Shape: (W, 1)
            self.x_n = np.linalg.solve(left_matrix, right_vector)  # Shape: (W, 1)

            # Update x_hat
            x_hat = self.lambd * self.x_n + (1 - self.lambd) * self.d_n  # Shape: (W, 1)

            # Update d_n using soft-thresholding
            temp = x_hat + self.p_n
            threshold = self.epsilon / self.delta_n
            self.d_n = np.sign(temp) * np.maximum(np.abs(temp) - threshold, 0)  # Shape: (W, 1)

            # Update p_n
            self.p_n += x_hat - self.d_n  # Shape: (W, 1)

            # Update delta_n
            self.delta_n = min(self.delta_n, self.delta_max)

        return self.x_n.flatten()
