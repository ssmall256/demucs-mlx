"""Unit tests for dynamic batch size tail fitting in hardware.py and apply_mlx.py."""
from demucs_mlx.hardware import fit_batch_size


def test_fit_batch_size_small():
    # When chunks are fewer than target, run exactly that count in 1 batch
    assert fit_batch_size(3, 8) == 3
    assert fit_batch_size(5, 8) == 5
    assert fit_batch_size(8, 8) == 8


def test_fit_batch_size_even_split():
    # Keep the batch count the target implies and spread chunks evenly
    assert fit_batch_size(21, 8) == 7  # 21 = 7 * 3 (120s benchmark!)
    assert fit_batch_size(14, 8) == 7  # 14 = 7 * 2
    assert fit_batch_size(16, 8) == 8  # 16 = 8 * 2
    assert fit_batch_size(24, 8) == 8  # 24 = 8 * 3
    assert fit_batch_size(30, 8) == 8  # 4 batches either way; 6 would need 5


def test_fit_batch_size_near_divisors():
    # When no exact divisor, keep total batch count minimal while maximizing tail batch occupancy
    assert fit_batch_size(31, 8) == 8  # 8*3 + 7 = 31 (tail is 7/8 full)
    assert fit_batch_size(11, 8) == 6  # 6*1 + 5 = 11 (tail is 5/6 full, 2 batches)


def test_fit_batch_size_pro_topology():
    # For M4 Pro (target_b = 4)
    assert fit_batch_size(2, 4) == 2
    assert fit_batch_size(4, 4) == 4
    assert fit_batch_size(6, 4) == 3  # 6 = 3 * 2
    assert fit_batch_size(9, 4) == 3  # 9 = 3 * 3
    assert fit_batch_size(12, 4) == 4  # 12 = 4 * 3


def test_fit_batch_size_never_adds_batches():
    # 38 chunks at target 4: ten batches. The old divisor search chose 2 (19 batches).
    for chunks in range(1, 200):
        for target in (2, 3, 4, 6, 8):
            b = fit_batch_size(chunks, target)
            assert 1 <= b <= target
            assert -(-chunks // b) == -(-chunks // target)
    assert fit_batch_size(38, 4) == 4
    assert fit_batch_size(38, 3) == 3
