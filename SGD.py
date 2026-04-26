from Optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, modules, lr=0.01):
        super().__init__(modules=modules, lr=lr)
    # end method

    def step(self):
        for module in self.modules:
            params = module.parameters()
            grads = module.grads()
            for param, grad in zip(params, grads):
                param -= self.lr * grad
            # end for
        # end for
    # end method

# end class
