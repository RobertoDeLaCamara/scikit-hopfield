"""Regression tests for HopfieldOOD LOO threshold calibration and HopfieldBAM gap_margin."""

from __future__ import annotations

import numpy as np
import pytest

from scikit_hopfield import HopfieldBAM, HopfieldOOD


def _sample_id_clusters(
    rng: np.random.Generator,
    n_per_cluster: int,
    dim: int = 16,
    scale: float = 0.25,
) -> np.ndarray:
    centers = np.zeros((3, dim), dtype=float)
    centers[0, 0] = 1.0
    centers[1, 1] = 1.0
    centers[2, 2] = 1.0
    return np.vstack([rng.normal(loc=c, scale=scale, size=(n_per_cluster, dim)) for c in centers])


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("full", {}),
        ("reclag", {"tau_rect": 0.4}),
    ],
)
def test_ood_unseen_id_false_positive_rate_matches_contamination(
    mode: str,
    kwargs: dict[str, float],
) -> None:
    """FPR on unseen in-distribution samples must track contamination (no self-match bias)."""
    rng = np.random.default_rng(2025)
    X_train = _sample_id_clusters(rng, n_per_cluster=200)
    X_test_id = _sample_id_clusters(rng, n_per_cluster=600)

    contamination = 0.05
    det = HopfieldOOD(
        beta=10.0,
        mode=mode,
        contamination=contamination,
        **kwargs,
    ).fit(X_train)

    preds = det.predict(X_test_id)
    fpr = float(np.mean(preds == -1))

    assert 0.025 <= fpr <= 0.085, (
        f"mode={mode!r} produced unseen-ID FPR={fpr:.3f} for contamination={contamination}; "
        "expected calibration near contamination via leave-one-out."
    )


def test_ood_full_with_outliers_unseen_id_calibration() -> None:
    """LOO calibration must preserve contamination target when X_outliers is provided."""
    rng = np.random.default_rng(2026)
    X_train = _sample_id_clusters(rng, n_per_cluster=200)
    X_test_id = _sample_id_clusters(rng, n_per_cluster=600)
    X_outliers = rng.normal(loc=[0.0] * 10 + [1.0] * 6, scale=0.2, size=(20, 16))

    contamination = 0.05
    det = HopfieldOOD(
        beta=10.0,
        mode="full",
        contamination=contamination,
    ).fit(X_train, X_outliers=X_outliers)

    fpr = float(np.mean(det.predict(X_test_id) == -1))
    assert 0.025 <= fpr <= 0.085, (
        f"full with X_outliers produced unseen-ID FPR={fpr:.3f} "
        f"for contamination={contamination}"
    )


def test_ood_removed_boosting_mode_and_reclag_outliers_rejected() -> None:
    """Removed 'boosting' mode and unsupported X_outliers in 'reclag' must raise ValueError."""
    rng = np.random.default_rng(7)
    X_id = rng.normal(size=(10, 4))
    X_out = rng.normal(loc=3.0, size=(3, 4))

    with pytest.raises(ValueError, match="Unsupported mode"):
        HopfieldOOD(mode="boosting").fit(X_id)

    with pytest.raises(ValueError, match="X_outliers is not supported"):
        HopfieldOOD(mode="reclag").fit(X_id, X_outliers=X_out)


@pytest.mark.parametrize("mode", ["full", "reclag"])
def test_ood_single_sample_fit_handles_loo_gracefully(mode: str) -> None:
    """Fitting with a single training sample must not fail or yield NaN/Inf thresholds."""
    X_single = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    det = HopfieldOOD(beta=8.0, mode=mode, tau_rect=0.5).fit(X_single)
    assert np.isfinite(det.threshold_)
    scores = det.decision_function(X_single)
    assert np.all(np.isfinite(scores))
    assert det.predict(X_single).shape == (1,)


def test_bam_gap_margin_parameter_is_explicit_and_validated() -> None:
    """HopfieldBAM exposes gap_margin as a validated parameter shifting gap_threshold_."""
    X_notes = np.eye(4, 8, dtype=float)
    Y_books = np.eye(4, 8, k=2, dtype=float)

    bam_default = HopfieldBAM(beta=15.0, gap_quantile=0.99).fit(X_notes, Y_books)
    bam_wider = HopfieldBAM(beta=15.0, gap_quantile=0.99, gap_margin=0.20).fit(X_notes, Y_books)

    assert bam_default.get_params()["gap_margin"] == pytest.approx(0.05)
    assert bam_wider.get_params()["gap_margin"] == pytest.approx(0.20)
    assert bam_wider.gap_threshold_ - bam_default.gap_threshold_ == pytest.approx(0.15)

    # Orthogonal singleton topics remain correctly separated (no blind LOO regression)
    known_gap = bam_default.detect_knowledge_gap(X_notes[0:1])
    unknown_query = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    unknown_gap = bam_default.detect_knowledge_gap(unknown_query)
    assert not bool(known_gap["is_gap"][0])
    assert bool(unknown_gap["is_gap"][0])

    # Validation happens in fit() per scikit-learn BaseEstimator contract
    with pytest.raises(ValueError, match="gap_margin"):
        HopfieldBAM(gap_margin=-0.01).fit(X_notes, Y_books)
    with pytest.raises(ValueError, match="gap_margin"):
        HopfieldBAM(gap_margin=float("nan")).fit(X_notes, Y_books)
    with pytest.raises(ValueError, match="gap_margin"):
        bam_default.set_params(gap_margin=-0.5).fit(X_notes, Y_books)
