import numpy as np
from Module import Module


class Linear(Module):
    """
    A fully connected linear layer: y = x @ W.T + b

    Stores parameter gradients during backward() for Optimizer-based updates.

    Note: Updated this one for optimizer based updates
    """

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features (int): Number of input features (e.g., 784 for MNIST).
            out_features (int): Number of output neurons (e.g., 10 for digits).
        """
        super().__init__()
        scale = np.sqrt(1.0 / in_features)
        self.W = (np.random.randn(out_features, in_features).astype(np.float32) * scale)
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
        Backward pass for a linear layer.

        Args:
            error (numpy.ndarray): Gradient w.r.t. output,
                                   shape (batch_size, out_features).
        Returns:
            numpy.ndarray: Gradient w.r.t. input (for deeper networks).
        """
        self.dW = error.T @ self.x
        self.db = np.sum(error, axis=0, keepdims=True)
        return error @ self.W
    # end method
    #added these lines
    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)
    # end method

    def parameters(self):
        return [self.W, self.b]
    # end method

    def grads(self):
        return [self.dW, self.db]
    # end method

# end class
