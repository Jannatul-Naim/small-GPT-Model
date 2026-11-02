import unittest
from unittest import mock

import torch

from model.modern_gpt import ModernGPT, apply_rope


def tiny_model():
    return ModernGPT(
        vocab_size=300,
        block_size=32,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
    ).eval()


class ModelTests(unittest.TestCase):
    def test_attention_is_causal(self):
        model = tiny_model()
        x = torch.randint(0, 300, (1, 32))
        changed = x.clone()
        changed[0, 20] = (changed[0, 20] + 1) % 300

        with torch.no_grad():
            before, _ = model(x)
            after, _ = model(changed)

        self.assertEqual((before[0, :20] - after[0, :20]).abs().max().item(), 0.0)
        self.assertGreater((before[0, 20:] - after[0, 20:]).abs().max().item(), 0.0)

    def test_rope_depends_on_relative_position(self):
        q = torch.randn(1, 1, 1, 16).repeat(1, 1, 8, 1)
        k = torch.randn(1, 1, 1, 16).repeat(1, 1, 8, 1)
        q, k = apply_rope(q, k)

        self.assertAlmostEqual(
            (q[0, 0, 1] * k[0, 0, 3]).sum().item(),
            (q[0, 0, 4] * k[0, 0, 6]).sum().item(),
            places=4,
        )

    def test_embedding_is_tied_to_output_head(self):
        model = tiny_model()
        self.assertIs(model.lm_head.weight, model.embedding.weight)

    def test_generate_stops_at_eos(self):
        model = tiny_model()

        def always_token_three(idx, targets=None):
            logits = torch.zeros(*idx.shape, 300)
            logits[..., 3] = 100.0
            return logits, None

        prompt = torch.tensor([[5, 6]])

        with mock.patch.object(model, "forward", always_token_three):
            stopped = model.generate(prompt, 10, top_k=0, eos_id=3)
            full = model.generate(prompt, 10, top_k=0)

        self.assertEqual(stopped.tolist(), [[5, 6, 3]])
        self.assertEqual(full.shape[1], 12)

    def test_generate_blocks_eos_until_min_new_tokens(self):
        model = tiny_model()

        def prefers_eos_then_four(idx, targets=None):
            logits = torch.zeros(*idx.shape, 300)
            logits[..., 3] = 100.0
            logits[..., 4] = 50.0
            return logits, None

        with mock.patch.object(model, "forward", prefers_eos_then_four):
            out = model.generate(
                torch.tensor([[5, 6]]),
                10,
                top_k=0,
                eos_id=3,
                min_new_tokens=2,
            )

        self.assertEqual(out.tolist(), [[5, 6, 4, 4, 3]])


if __name__ == "__main__":
    unittest.main()
