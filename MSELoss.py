import numpy as np

from Loss import Loss


class MSELoss(Loss):
    """
    Mean Squared Error loss we use during training.
    """

    def __init__(self):
        super().__init__()
    # end method

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return float(np.mean((predictions - targets) ** 2))
    # end method

    def backward(self):
        n = self.predictions.size
        return (2.0 / n) * (self.predictions - self.targets)
    # end method

# end class
