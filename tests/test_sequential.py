import os
import sys
import unittest

import numpy as np


# Ensure project root is importable when running tests via discovery.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Module import Module
from Sequential import Sequential


class ScaleShift(Module):
    def __init__(self, scale, shift):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.scale * x + self.shift

    def backward(self, grad_output):
        # d/dx (scale * x + shift) = scale
        return grad_output * self.scale


class TestSequential(unittest.TestCase):
    def test_forward_order_and_backward_reverse_order(self):
        seq = Sequential([
            ScaleShift(scale=2.0, shift=1.0),
            ScaleShift(scale=-3.0, shift=5.0),
        ])

        x = np.array([[2.0]], dtype=np.float32)
        out = seq.forward(x)
        # ((2*x + 1) * -3 + 5)
        expected_out = ((2.0 * x + 1.0) * -3.0) + 5.0
        np.testing.assert_allclose(out, expected_out, rtol=1e-6, atol=1e-6)

        grad_output = np.array([[1.0]], dtype=np.float32)
        grad_input = seq.backward(grad_output)
        # Chain rule: grad * (-3) * (2) = -6
        expected_grad_input = np.array([[-6.0]], dtype=np.float32)
        np.testing.assert_allclose(grad_input, expected_grad_input, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
