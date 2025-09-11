"""Tests for core manifest utilities."""
from polypdb.manifests.core import pick_k, hmac_rank


def test_pick_k_same_seed_same_order():
    items = ["a", "b", "c", "d"]
    seed = 42
    expected = ["b", "c", "a", "d"]
    result1 = pick_k(items, seed, k=len(items), key=lambda x: x)
    result2 = pick_k(items, seed, k=len(items), key=lambda x: x)
    assert result1 == expected
    assert result1 == result2


def test_pick_k_different_seed_different_order():
    items = ["a", "b", "c", "d"]
    seed1 = 42
    seed2 = 43
    result1 = pick_k(items, seed1, k=len(items), key=lambda x: x)
    result2 = pick_k(items, seed2, k=len(items), key=lambda x: x)
    assert result1 != result2


def test_hmac_rank_stable():
    expected = 104605240965717557086117454196918639815776956194521525658296304952388907657516
    assert hmac_rank(42, "example") == expected
