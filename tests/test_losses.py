import unittest

import torch
from torch.nn import functional as F

from bigw_rlv import relative_local_variance_error
from bigw_rlv.losses import _weighted_score


class LossChecks(unittest.TestCase):
    def test_rlv_matches_direct_window_variance(self):
        def variance(image):
            windows = F.pad(image, (1, 1, 1, 1), mode="reflect")
            return windows.unfold(2, 3, 1).unfold(3, 3, 1).var(
                dim=(-1, -2), unbiased=False
            )

        generator = torch.Generator().manual_seed(123)
        infrared = torch.rand(2, 1, 9, 11, generator=generator, dtype=torch.float64)
        visible = torch.rand(infrared.shape, generator=generator, dtype=infrared.dtype)
        fused = ((infrared + visible) * 0.5).requires_grad_()
        target = torch.maximum(variance(infrared), variance(visible))
        dims = (1, 2, 3)
        expected = (variance(fused) - target).abs().mean(dim=dims) / (target.mean(dim=dims) + 1e-4)
        actual = relative_local_variance_error(infrared, visible, fused)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(actual.mean(), fused)[0]).all().item())

    def test_flat_target(self):
        flat = torch.zeros(2, 1, 5, 5)
        torch.testing.assert_close(relative_local_variance_error(flat, flat, flat), torch.zeros(2))

    def test_bigw_anchor_weights_are_independent(self):
        def anchor_derivative(energy_ir, energy_vi):
            score_ir = torch.tensor([0.4, 0.7], requires_grad=True)
            score_vi = torch.tensor([0.8, 0.1])
            score = _weighted_score(torch.tensor(energy_ir), torch.tensor(energy_vi), score_ir, score_vi)
            return torch.autograd.grad(score, score_ir)[0][0]

        torch.testing.assert_close(
            anchor_derivative([4.0, 8.0], [1.0, 2.0]),
            anchor_derivative([4.0, 2.0], [1.0, 8.0]),
        )


if __name__ == "__main__":
    unittest.main()
