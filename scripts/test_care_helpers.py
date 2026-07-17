import unittest

import torch

from generation import (
    apply_intervention_stability_guard,
    candidate_completes_exact_cycle,
    compute_counterfactual_aligned_logits,
)


class CareHelperTests(unittest.TestCase):
    def test_counterfactual_alignment_keeps_only_matching_signs(self):
        intervention = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
        base = torch.tensor([[2.0, 2.0, 1.0, 1.0]])
        text = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
        logits, aligned, intervention_delta, visual_delta = (
            compute_counterfactual_aligned_logits(intervention, base, text, alpha=0.5)
        )
        self.assertTrue(torch.equal(aligned, torch.tensor([[True, True, False, False]])))
        self.assertTrue(torch.allclose(logits, torch.tensor([[3.5, 0.5, 2.0, 0.0]])))
        self.assertTrue(torch.equal(intervention_delta, torch.tensor([[1.0, -1.0, 1.0, -1.0]])))
        self.assertTrue(torch.equal(visual_delta, torch.tensor([[1.0, -1.0, -1.0, 1.0]])))

    def test_exact_cycle_detects_candidate_completion(self):
        history = torch.tensor([[1, 2, 1, 2, 1]])
        self.assertTrue(candidate_completes_exact_cycle(history, torch.tensor([2])).item())
        self.assertFalse(candidate_completes_exact_cycle(history, torch.tensor([3])).item())

    def test_stability_guard_falls_back_only_for_new_intervention_cycle(self):
        history = torch.tensor([[1, 2, 1, 2, 1]])
        intervention_logits = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
        candidate_logits = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
        routed, guarded, candidate_cycle, intervention_cycle = (
            apply_intervention_stability_guard(
                intervention_logits,
                candidate_logits,
                history,
            )
        )
        self.assertTrue(guarded.item())
        self.assertTrue(candidate_cycle.item())
        self.assertFalse(intervention_cycle.item())
        self.assertTrue(torch.equal(routed, intervention_logits))


if __name__ == "__main__":
    unittest.main()
