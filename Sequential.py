from Module import Module


class Sequential(Module):
    """
    Container that lets us chain modules in order.
    """

    def __init__(self, modules):
        super().__init__()
        self.modules = list(modules)
    # end method

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        # end for
        return x
    # end method

    def backward(self, grad_output):
        for module in reversed(self.modules):
            grad_output = module.backward(grad_output)
        # end for
        return grad_output
    # end method

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        # end for
        return params
    # end method

    def grads(self):
        gradients = []
        for module in self.modules:
            gradients.extend(module.grads())
        # end for
        return gradients
    # end method

    def zero_grad(self):
        for module in self.modules:
            if hasattr(module, "zero_grad"):
                module.zero_grad()
            # end if
        # end for
    # end method

# end class
