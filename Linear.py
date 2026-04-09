import numpy as np
from Module import Module


class Linear(Module):
    """
    A fully connected linear layer: y = x @ W.T + b

    Implements the Perceptron Learning Rule in backward() for direct
    weight updates without a separate Optimizer.
    """

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features (int): Number of input features (e.g., 784 for MNIST).
            out_features (int): Number of output neurons (e.g., 10 for digits).
        """
        super().__init__()
        self.W = np.random.randn(out_features, in_features).astype(np.float32) * 0.01
        self.b = np.zeros((1, out_features), dtype=np.float32)
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    # end method

    def forward(self, x):
        """
        Forward pass: y = x @ W.T + b (batch-first notation).

        Args:
            x (numpy.ndarray): Shape (batch_size, in_features).
        Returns:
            numpy.ndarray: Shape (batch_size, out_features).
        """
        self.x = x
        return x @ self.W.T + self.b
    # end method

    def backward(self, error):
        """
        Applies the Perceptron Learning Rule.

        For each sample, update:
            W += error.T @ x
            b += error

        Args:
            error (numpy.ndarray): Error signal (target - prediction),
                                   shape (batch_size, out_features).
        Returns:
            numpy.ndarray: Gradient w.r.t. input (for deeper networks).
        """
        self.dW = error.T @ self.x
        self.db = np.sum(error, axis=0, keepdims=True)

        # Directly update weights (no optimizer needed for perceptron)
        self.W += self.dW
        self.b += self.db

        return error @ self.W
    # end method

    def parameters(self):
        return [self.W, self.b]
    # end method

    def grads(self):
        return [self.dW, self.db]
    # end method

# end class
