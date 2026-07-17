import unittest

import torch

from generation import summarize_teacher_target_logits


class CirdTeacherHelperTests(unittest.TestCase):
    def test_target_nll_and_deltas(self):
        base = torch.tensor([[0.0, 1.0, 2.0]])
        vhr = torch.tensor([[0.0, 2.0, 2.0]])
        sc = torch.tensor([[0.0, 3.0, 2.0]])
        stats = summarize_teacher_target_logits(base, vhr, sc, 1)
        self.assertTrue(stats["sc_target_plausible"])
        self.assertAlmostEqual(stats["vhr_minus_base_target_logit"], 1.0)
        self.assertAlmostEqual(stats["sc_minus_vhr_target_logit"], 1.0)
        self.assertLess(stats["sc_nll"], stats["vhr_nll"])

    def test_masked_teacher_target_is_explicit(self):
        base = torch.zeros(1, 3)
        vhr = torch.zeros(1, 3)
        sc = torch.tensor([[0.0, -float("inf"), 0.0]])
        stats = summarize_teacher_target_logits(base, vhr, sc, 1)
        self.assertFalse(stats["sc_target_plausible"])
        self.assertIsNone(stats["sc_nll"])
        self.assertIsNone(stats["sc_minus_vhr_target_logit"])


if __name__ == "__main__":
    unittest.main()
