"""Property-based and edge-case test suite (@QA & @TestGen) for scikit-hopfield."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from scikit_hopfield import (
    HopfieldBAM,
    HopfieldMetastableCluster,
    HopfieldOOD,
    attention_weights,
    hopfield_energy,
    l2_normalize,
    logsumexp_scaled,
    reclag_score,
)


# ---------------------------------------------------------------------------
# 1. Hypothesis Property-Based Tests: Softmax Attention Simplex & Stability
# ---------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(
    n_memories=st.integers(min_value=1, max_value=25),
    n_queries=st.integers(min_value=1, max_value=15),
    dim=st.integers(min_value=2, max_value=16),
    beta=st.floats(min_value=1e-3, max_value=500.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_property_attention_weights_simplex_and_stability(
    n_memories: int,
    n_queries: int,
    dim: int,
    beta: float,
    seed: int,
) -> None:
    """Property: attention_weights(X, Q, beta) is non-negative, finite, and sums to 1.0 row-wise."""
    rng = np.random.default_rng(seed)
    X = l2_normalize(rng.normal(size=(n_memories, dim)))
    Q = l2_normalize(rng.normal(size=(n_queries, dim)))

    weights = attention_weights(X, Q, beta=beta)

    assert weights.shape == (n_queries, n_memories)
    assert np.all(np.isfinite(weights)), "NaN or Inf detected in attention_weights"
    assert np.all(weights >= 0.0), "Negative probability detected in attention_weights"
    assert np.all(np.isclose(weights.sum(axis=1), 1.0, atol=1e-6)), "Row sums deviate from 1.0"


@settings(max_examples=50, deadline=None)
@given(
    X=hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 12), st.integers(2, 8)),
        elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    ),
    Q=hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 8), st.integers(2, 8)),
        elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    ),
    beta=st.floats(min_value=1e-3, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_property_arbitrary_matrices_attention_and_energy_finite(
    X: np.ndarray,
    Q: np.ndarray,
    beta: float,
) -> None:
    """Property: Random matrices yield non-negative weights summing to 1.0."""
    d = min(X.shape[1], Q.shape[1])
    X_aligned = l2_normalize(X[:, :d])
    Q_aligned = l2_normalize(Q[:, :d])

    weights = attention_weights(X_aligned, Q_aligned, beta=beta)
    energies = hopfield_energy(X_aligned, Q_aligned, beta=beta)
    lse = logsumexp_scaled(Q_aligned @ X_aligned.T, beta=beta)

    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.all(np.isclose(weights.sum(axis=1), 1.0, atol=1e-6))
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(lse))


# ---------------------------------------------------------------------------
# 2. Extreme Beta (1e-3 to 500.0) & Zero-Norm Vector Edge Cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 250.0, 500.0])
def test_extreme_beta_and_zero_norm_vectors_numerical_stability(beta: float) -> None:
    """Verifies no NaN/Inf across estimators and energy functions for extreme beta."""
    X_zeros = np.zeros((5, 8), dtype=float)
    Q_zeros = np.zeros((3, 8), dtype=float)

    # 1D and 2D zero-norm normalization
    norm_1d = l2_normalize(np.zeros(8, dtype=float))
    norm_2d = l2_normalize(X_zeros)
    assert norm_1d.shape == (1, 8)
    assert np.all(norm_2d == 0.0)
    assert np.all(np.isfinite(norm_2d))

    # Energy primitives on zero-norm matrices
    w = attention_weights(norm_2d, l2_normalize(Q_zeros), beta=beta)
    e = hopfield_energy(norm_2d, l2_normalize(Q_zeros), beta=beta)
    r = reclag_score(norm_2d, l2_normalize(Q_zeros), beta=beta, tau_rect=0.5)
    assert np.all(np.isfinite(w))
    assert np.all(np.isclose(w.sum(axis=1), 1.0))
    assert np.all(np.isfinite(e))
    assert np.all(r == 0.0)

    # Mixed normal + zero-norm inputs across HopfieldOOD, HopfieldMetastableCluster, HopfieldBAM
    rng = np.random.default_rng(123)
    X_mixed = np.vstack([rng.normal(size=(8, 6)), np.zeros((2, 6))])
    Q_mixed = np.vstack([rng.normal(size=(3, 6)), np.zeros((1, 6))])

    ood = HopfieldOOD(beta=beta, mode="full").fit(X_mixed)
    scores = ood.decision_function(Q_mixed)
    preds = ood.predict(Q_mixed[0])  # 1D query edge case
    assert np.all(np.isfinite(scores))
    assert preds.shape == (1,)

    clust = HopfieldMetastableCluster(beta=beta, max_iter=10).fit(X_mixed)
    assert np.all(np.isfinite(clust.cluster_centers_))
    assert clust.labels_.shape == (10,)

    bam = HopfieldBAM(beta=beta, max_iter=5).fit(X_mixed, X_mixed)
    res = bam.resonate(Q_mixed[0])  # 1D query edge case
    gap = bam.detect_knowledge_gap(Q_mixed)
    assert np.all(np.isfinite(res["resonant_b"]))
    assert np.all(np.isfinite(gap["energy"]))


# ---------------------------------------------------------------------------
# 3. HopfieldBAM Domain Shape Mismatch & Unnormalized / Max-Iter Branches
# ---------------------------------------------------------------------------
def test_bam_domain_shape_mismatch_raises_value_error() -> None:
    """Domain A and Domain B with different row counts must raise ValueError."""
    X_a = np.ones((5, 4), dtype=float)
    Y_b = np.ones((3, 4), dtype=float)
    bam = HopfieldBAM(beta=10.0)
    with pytest.raises(ValueError, match="same number of paired/aligned associative slots"):
        bam.fit(X_a, Y_b)


def test_bam_unnormalized_and_single_step_convergence() -> None:
    """Tests HopfieldBAM with normalize=False and max_iter=1 non-convergence branch."""
    X_a = np.eye(4, 6, dtype=float)
    Y_b = np.eye(4, 6, k=1, dtype=float)
    bam = HopfieldBAM(beta=0.5, max_iter=1, tol=1e-12, normalize=False).fit(X_a, Y_b)
    res = bam.resonate(X_a[0])
    assert res["iterations"][0] == 1
    assert np.all(np.isfinite(res["top_b_weights"]))


# ---------------------------------------------------------------------------
# 4. HopfieldOOD Modes: KMeans & Unlabeled/Single-Class SHE
# ---------------------------------------------------------------------------
def test_ood_kmeans_and_unlabeled_or_single_class_she() -> None:
    """Verifies HopfieldOOD(mode='kmeans') and HopfieldOOD(mode='she', y=None / single-class)."""
    rng = np.random.default_rng(99)
    X_in = rng.normal(loc=2.0, scale=0.1, size=(25, 8))
    X_out = rng.normal(loc=-2.0, scale=0.1, size=(5, 8))

    # 1. mode='kmeans' with n_prototypes < N and n_prototypes > N (single-sample bank)
    det_km = HopfieldOOD(beta=10.0, mode="kmeans", n_prototypes=4).fit(X_in)
    assert det_km.memories_.shape == (4, 8)
    scores_km = det_km.decision_function(X_out)
    assert np.all(np.isfinite(scores_km))
    assert np.all(det_km.predict(X_out) == -1)

    det_km_single = HopfieldOOD(beta=10.0, mode="kmeans", n_prototypes=32).fit(X_in[:1])
    assert det_km_single.memories_.shape == (1, 8)

    # 2. mode='she' with y=None (unlabeled global prototype)
    det_she_unlabeled = HopfieldOOD(beta=10.0, mode="she").fit(X_in, y=None)
    assert det_she_unlabeled.memories_.shape == (1, 8)
    scores_she_unlabeled = det_she_unlabeled.decision_function(X_out)
    assert np.all(np.isfinite(scores_she_unlabeled))
    assert np.all(det_she_unlabeled.predict(X_out) == -1)

    # 3. mode='she' with single-class labels y=np.zeros(N)
    y_single = np.zeros(X_in.shape[0], dtype=int)
    det_she_single = HopfieldOOD(beta=10.0, mode="she").fit(X_in, y=y_single)
    assert det_she_single.memories_.shape == (1, 8)
    assert np.allclose(det_she_unlabeled.memories_, det_she_single.memories_)


# ---------------------------------------------------------------------------
# 5. HopfieldMetastableCluster Helper Methods (fit_predict, bifurcation_spectrum)
# ---------------------------------------------------------------------------
def test_clustering_fit_predict_and_bifurcation_spectrum() -> None:
    """Verifies fit_predict and bifurcation_spectrum across beta schedule."""
    rng = np.random.default_rng(7)
    X = np.vstack(
        [
            rng.normal(loc=[1.0, 0.0, 0.0], scale=0.03, size=(10, 3)),
            rng.normal(loc=[0.0, 1.0, 0.0], scale=0.03, size=(10, 3)),
        ]
    )
    clust = HopfieldMetastableCluster(beta=12.0, merge_tol=1e-2)
    labels = clust.fit_predict(X)
    assert labels.shape == (20,)
    assert len(np.unique(labels)) == 2

    spectrum = clust.bifurcation_spectrum(X, betas=[0.01, 12.0])
    assert spectrum[0.01] == 1
    assert spectrum[12.0] == 2
