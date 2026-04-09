import numpy as np
from Loss import Loss


class Error(Loss):
    """
    Simple error loss for the Perceptron Learning Rule.

    Computes element-wise error: target - prediction.
    The scalar loss returned is the mean absolute error for tracking.
    """

    def __init__(self):
        super().__init__()
    # end method

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        error = targets - predictions
        return np.mean(np.abs(error))
    # end method

    def backward(self):
        return self.targets - self.predictions
    # end method

# end class
