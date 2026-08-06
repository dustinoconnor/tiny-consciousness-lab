import unittest

import numpy as np

from causal_dsl_lora import build_curriculum
from variable_binding_probe import (
    ProbeCase,
    ablate_attention_head,
    binary_auc,
    find_attention_output_projection,
    leave_group_out_curve,
    load_frozen_cases,
    rank_attention_heads,
    select_balanced_cases,
    target_row,
)


class VariableBindingProbeTests(unittest.TestCase):
    def test_frozen_artifact_aligns_with_reconstructed_curriculum(self):
        _artifact, cases = load_frozen_cases()
        self.assertEqual(len(cases), 128)
        self.assertEqual(sum(not case.generation_correct for case in cases), 7)

    def test_selection_retains_failures_and_balances_rows(self):
        _artifact, cases = load_frozen_cases()
        selected = select_balanced_cases(cases, maximum=32)
        self.assertEqual(len(selected), 32)
        self.assertEqual(sum(not case.generation_correct for case in selected), 7)
        row_counts = [sum(target_row(case.example) == row for case in selected) for row in (0, 1)]
        self.assertLessEqual(abs(row_counts[0] - row_counts[1]), 1)
        self.assertEqual(len({case.index for case in selected}), 32)

    def test_leave_group_out_probe_generalizes_decodable_signal(self):
        labels = np.asarray([index % 2 for index in range(32)])
        groups = np.asarray([f"pair-{index % 4}" for index in range(32)])
        activations = np.zeros((32, 3, 6), dtype=np.float64)
        activations[:, 0, 0] = labels * 2 - 1
        activations[:, 1, 1] = labels * 4 - 2
        activations[:, 2, 2] = labels * 6 - 3
        result = leave_group_out_curve(activations, labels, groups)
        self.assertEqual(result["accuracy_by_layer"], [1.0, 1.0, 1.0])

    def test_auc_and_attention_split_are_deterministic(self):
        self.assertEqual(binary_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        scores = np.zeros((8, 2, 3), dtype=np.float64)
        scores[::2, 1, 2] = 0.75
        ranking = rank_attention_heads(scores, list(range(8)))
        self.assertEqual(ranking["top_binding_head"], {
            "layer": 1,
            "head": 2,
            "mean_target_minus_control_attention": 0.75,
        })
        self.assertEqual(ranking["confirmation_cases"], 4)

    def test_head_ablation_zeros_only_selected_slice(self):
        import torch

        layer = torch.nn.Linear(8, 2, bias=False)
        observed = []

        def capture(_module, arguments):
            observed.append(arguments[0].detach().clone())

        values = torch.ones(1, 1, 8)
        with ablate_attention_head(layer, head=1, head_dim=2):
            capture_handle = layer.register_forward_pre_hook(capture)
            layer(values)
            capture_handle.remove()
        self.assertTrue(torch.equal(observed[0][..., :2], torch.ones(1, 1, 2)))
        self.assertTrue(torch.equal(observed[0][..., 2:4], torch.zeros(1, 1, 2)))
        self.assertTrue(torch.equal(observed[0][..., 4:], torch.ones(1, 1, 4)))

    def test_output_projection_lookup_is_layer_specific(self):
        import torch

        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = torch.nn.Linear(4, 4)

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

        model = FakeModel()
        self.assertIs(
            find_attention_output_projection(model, 1),
            model.layers[1].self_attn.o_proj,
        )


if __name__ == "__main__":
    unittest.main()
