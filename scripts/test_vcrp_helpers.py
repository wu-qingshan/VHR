import unittest

import torch

from vcrp import project_ffn_to_visual_halfspace


class VcrpProjectionTests(unittest.TestCase):
    def test_aligned_ffn_is_unchanged(self):
        ffn = torch.tensor([[2.0, 1.0]])
        visual = torch.tensor([[1.0, 0.0]])
        corrected, stats = project_ffn_to_visual_halfspace(ffn, visual)
        self.assertTrue(torch.allclose(corrected, ffn))
        self.assertFalse(stats["conflict"].item())

    def test_opposing_component_is_minimally_removed(self):
        ffn = torch.tensor([[-2.0, 3.0]])
        visual = torch.tensor([[1.0, 0.0]])
        corrected, stats = project_ffn_to_visual_halfspace(ffn, visual)
        self.assertTrue(torch.allclose(corrected, torch.tensor([[0.0, 3.0]]), atol=1e-5))
        self.assertTrue(stats["conflict"].item())
        self.assertAlmostEqual(float((corrected * visual).sum().item()), 0.0, places=5)

    def test_zero_visual_residual_is_safe(self):
        ffn = torch.tensor([[-2.0, 3.0]])
        visual = torch.zeros_like(ffn)
        corrected, stats = project_ffn_to_visual_halfspace(ffn, visual)
        self.assertTrue(torch.allclose(corrected, ffn))
        self.assertFalse(stats["conflict"].item())
        self.assertEqual(stats["correction_norm"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
