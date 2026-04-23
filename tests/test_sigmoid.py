import os
import sys
import unittest

import numpy as np


# Ensure project root is importable when running tests via unittest directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):
    def test_forward_matches_expected(self):
        layer = Sigmoid()
        x = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)

        out = layer.forward(x)
        expected = 1.0 / (1.0 + np.exp(-x))

        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(out.dtype, x.dtype)

    def test_backward_uses_sigmoid_derivative(self):
        layer = Sigmoid()
        x = np.array([[-1.0, 1.0]], dtype=np.float32)
        y = layer.forward(x)

        grad_output = np.array([[0.5, -0.25]], dtype=np.float32)
        grad_input = layer.backward(grad_output)

        expected = grad_output * y * (1.0 - y)
        np.testing.assert_allclose(grad_input, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
