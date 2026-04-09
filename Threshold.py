import numpy as np
from Activation import Activation


class Threshold(Activation):
    """
    Threshold (step) activation function for perceptrons.

    Outputs 1.0 where input >= threshold, 0.0 otherwise.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    # end method

    def forward(self, x):
        self.x = x
        return (x >= self.threshold).astype(np.float32)
    # end method

    def backward(self, grad_output):
        # Threshold has no meaningful gradient; pass through
        return grad_output
    # end method

# end class
