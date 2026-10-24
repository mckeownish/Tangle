"""Checks for wiggle.fingerprint. Run with:  python -m pytest tests/test_fingerprint.py"""
import numpy as np
import pytest

from wiggle import fingerprint as fp


def random_walk(n=40, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n - 1, 3))
    steps *= 3.8 / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.vstack([np.zeros(3), np.cumsum(steps, 0)])


def helix(n=40, radius=2.3, rise=1.5, turn_deg=100):
    k = np.arange(n)
    th = np.deg2rad(turn_deg) * k
    return np.c_[radius * np.cos(th), radius * np.sin(th), rise * k]


def test_fingerprint_matches_brute_force():
    X = random_walk()
    W = fp.writhe_matrix(X)
    F = fp.fingerprint(X)
    assert np.isclose(fp.global_writhe(F), np.triu(W, 1).sum())
    for i, j in [(0, 10), (5, 30), (12, 38)]:
        brute = sum(W[a, b] for a in range(i, j + 1) for b in range(a + 1, j + 1))
        assert np.isclose(F[i, j], brute)


def test_batch_equals_single():
    X = np.stack([random_walk(seed=s) for s in range(3)])
    Fb = fp.fingerprint(X, m=2)
    for t in range(3):
        assert np.allclose(Fb[t], fp.fingerprint(X[t], m=2), atol=1e-5)


def test_smoothing_matches_binomial_average():
    X = random_walk()
    m = 4
    w = np.array([1, 4, 6, 4, 1]) / 16
    expected = np.array([w @ X[i:i + m + 1] for i in range(len(X) - m)])
    assert np.allclose(fp.smooth(X, m), expected)


def test_pyramid_geometry():
    X = random_walk()
    F = fp.fingerprint(X)
    assert np.all(np.isnan(fp.pyramid(F, F))[np.isnan(fp.pyramid(F))])          # same mask
    assert np.nanmax(np.abs(fp.pyramid(F, F))) == 0
    S = fp.pyramid(F, per_length=False)
    n = F.shape[0]
    ell, k = 10, 20
    assert np.isclose(S[ell - 1, k], F[k - ell // 2, k - ell // 2 + ell])
    assert np.isnan(S[n - 2, 0])                                                 # window off the chain


def test_helix_writhe_positive_and_mirror_negates():
    H = helix()
    wr = fp.global_writhe(fp.fingerprint(H))
    assert wr > 3
    assert np.isclose(fp.global_writhe(fp.fingerprint(H * [1, 1, -1])), -wr, atol=1e-4)


def test_smoothing_does_not_break_helix():
    counts, _ = fp.smoothing_passages(helix(60), m_max=10)
    assert counts.sum() == 0


def test_strand_passage_detected():
    # segment 0-1 along x at z=0; segment 2-3 along y moving from z=+1 to z=-1 through it
    base = np.array([[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0]], float)
    X0, X1 = base + [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]], base + [[0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, -1]]
    # insert a far-away linker point so the two segments are non-adjacent (segments 0 and 3)
    X0 = np.vstack([X0[:2], [[5, 5, 5]], X0[2:]])
    X1 = np.vstack([X1[:2], [[5, 5, 5]], X1[2:]])
    hits = fp.strand_passages(X0, X1)
    assert [(i, j) for i, j, _, _ in hits] == [(0, 3)]
    assert abs(hits[0][2] - 0.5) < 0.02                     # halfway along the path
    W0, W1 = fp.writhe_matrix(X0), fp.writhe_matrix(X1)
    assert np.sign(W0[0, 3]) == -np.sign(W1[0, 3])           # the pair's handedness flips


def test_compare_with_arrays():
    A = random_walk(seed=1)
    B = A + np.random.default_rng(2).normal(scale=0.5, size=A.shape)
    cmp = fp.compare(A, B, m=2, max_step=None, verbose=False, check_smoothing=False)
    n_seg = len(A) - 1 - 2                                  # m = 2 removes two points
    assert cmp.S.shape == (n_seg - 1, n_seg)
    v, res, ell = cmp.peak()
    assert ell >= fp.L_MIN
    col = cmp.cross_section(8)
    assert col.shape == (len(A),)
