import os
import sys
import unittest
import numpy as np


# Ensure project root is importable when running tests via discovery.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Linear import Linear
from MSELoss import MSELoss
from SGD import SGD
from Sequential import Sequential


class TestSGD(unittest.TestCase):
    def test_step_updates_parameters(self):
        layer = Linear(in_features=2, out_features=1)
        layer.W[...] = np.array([[1.0, -2.0]], dtype=np.float32)
        layer.b[...] = np.array([[0.5]], dtype=np.float32)
        layer.dW[...] = np.array([[0.2, -0.4]], dtype=np.float32)
        layer.db[...] = np.array([[0.1]], dtype=np.float32)

        opt = SGD(modules=[layer], lr=0.5)
        opt.step()

        expected_W = np.array([[0.9, -1.8]], dtype=np.float32)
        expected_b = np.array([[0.45]], dtype=np.float32)

        np.testing.assert_allclose(layer.W, expected_W, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(layer.b, expected_b, rtol=1e-6, atol=1e-6)

    def test_zero_grad_clears_nested_module_grads(self):
        model = Sequential([
            Linear(2, 3),
            Linear(3, 1),
        ])
        loss_fn = MSELoss()
        opt = SGD(modules=[model], lr=0.1)

        x = np.array([[1.0, 2.0]], dtype=np.float32)
        y = np.array([[0.0]], dtype=np.float32)

        out = model.forward(x)
        _ = loss_fn.forward(out, y)
        grad = loss_fn.backward()
        _ = model.backward(grad)

        # Verify grads are non-zero before zeroing.
        has_nonzero_before = any(np.any(g != 0.0) for g in model.grads())
        self.assertTrue(has_nonzero_before)

        opt.zero_grad()

        has_nonzero_after = any(np.any(g != 0.0) for g in model.grads())
        self.assertFalse(has_nonzero_after)


if __name__ == "__main__":
    unittest.main()
