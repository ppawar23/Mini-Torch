import os
import sys
import unittest
import numpy as np


# Ensure project root is importable when running tests via unittest directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MSELoss import MSELoss


class TestMSELoss(unittest.TestCase):
    def test_forward_and_backward_match_definition(self):
        loss_fn = MSELoss()

        predictions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targets = np.array([[1.5, 2.5], [2.0, 5.0]], dtype=np.float32)

        loss = loss_fn.forward(predictions, targets)
        grad = loss_fn.backward()

        expected_loss = float(np.mean((predictions - targets) ** 2))
        expected_grad = (2.0 / predictions.size) * (predictions - targets)

        self.assertAlmostEqual(loss, expected_loss, places=7)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
