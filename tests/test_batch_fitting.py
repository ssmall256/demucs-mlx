"""Unit tests for dynamic batch size tail fitting in hardware.py and apply_mlx.py."""
from demucs_mlx.hardware import fit_batch_size


def test_fit_batch_size_small():
    # When chunks are fewer than target, run exactly that count in 1 batch
    assert fit_batch_size(3, 8) == 3
    assert fit_batch_size(5, 8) == 5
    assert fit_batch_size(8, 8) == 8


def test_fit_batch_size_exact_divisors():
    # When exact divisors exist, pick the exact divisor to eliminate tail remainder
    assert fit_batch_size(21, 8) == 7  # 21 = 7 * 3 (120s benchmark!)
    assert fit_batch_size(14, 8) == 7  # 14 = 7 * 2
    assert fit_batch_size(16, 8) == 8  # 16 = 8 * 2
    assert fit_batch_size(24, 8) == 8  # 24 = 8 * 3
    assert fit_batch_size(30, 8) == 6  # 30 = 6 * 5


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
