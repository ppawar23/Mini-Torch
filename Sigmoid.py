import numpy as np
try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))
# end try

from Activation import Activation

try:
    import numexpr as ne
except ImportError:
    ne = None
# end try


class Sigmoid(Activation):
    """
    Sigmoid activation layer for our MLP blocks.

    Applies sigma(x) = 1 / (1 + exp(-x)) element-wise.
    """

    def __init__(self):
        super().__init__()
        self.y = None
    # end method

    def forward(self, x):
        self.x = x
        self.y = expit(x)
        if self.y.dtype != x.dtype:
            self.y = self.y.astype(x.dtype)
        # end if
        return self.y
    # end method

    def backward(self, grad_output):
        if ne is not None:
            local_grad = ne.evaluate("y * (1.0 - y)", local_dict={"y": self.y})
        else:
            local_grad = self.y * (1.0 - self.y)
        # end if

        return grad_output * local_grad
    # end method

# end class
