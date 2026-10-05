# pyright: reportOptionalCall=false
"""Fused attention projections are derived caches and must never be saved."""

from __future__ import annotations

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from demucs_mlx.mlx_transformer import FastMultiHeadAttention


class FusedAttentionStateTest(unittest.TestCase):
    def test_forward_does_not_change_saved_keys(self) -> None:
        attn = FastMultiHeadAttention(16, 4)
        before = dict(tree_flatten(attn.parameters()))
        x = mx.random.normal((1, 5, 16))
        mx.eval(attn(x, x, x), attn(x, x + 1, x + 1))
        after = dict(tree_flatten(attn.parameters()))
        self.assertEqual(sorted(before), sorted(after))
        self.assertFalse(any("qkv_proj" in key or "kv_proj" in key for key in after))


if __name__ == "__main__":
    unittest.main()
