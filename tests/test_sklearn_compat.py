"""Scikit-learn compatibility, check_estimator, KMeans random_state, and clustering invariance."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from scikit_hopfield import HopfieldBAM, HopfieldMetastableCluster, HopfieldOOD


def test_sklearn_check_estimator_metastable_cluster() -> None:
    check_estimator(HopfieldMetastableCluster())


def test_sklearn_check_estimator_bam() -> None:
    check_estimator(HopfieldBAM())


def test_sklearn_check_estimator_ood_kmeans_and_she() -> None:
    check_estimator(HopfieldOOD(mode="kmeans", n_prototypes=8, random_state=0))
    check_estimator(HopfieldOOD(mode="she"))


def test_sklearn_check_estimator_ood_full_loo() -> None:
    """
    `HopfieldOOD(mode='full')` calibrates `threshold_` via leave-one-out so unseen
    in-distribution FPR matches `contamination`, intentionally differing from
    self-matched training-set predictions checked by sklearn's two contamination checks.
    """
    reason = (
        "LOO threshold calibration targets unseen-ID contamination rather than "
        "self-matched training predictions."
    )
    check_estimator(
        HopfieldOOD(mode="full"),
        expected_failed_checks={
            "check_outliers_train": reason,
            "check_outliers_fit_predict": reason,
        },
    )


def test_set_params_rejects_fitted_attributes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))

    ood = HopfieldOOD().fit(X)
    with pytest.raises(ValueError, match="Invalid parameter"):
        ood.set_params(memories_=X)

    clust = HopfieldMetastableCluster().fit(X)
    with pytest.raises(ValueError, match="Invalid parameter"):
        clust.set_params(labels_=np.zeros(12))

    bam = HopfieldBAM().fit(X, X)
    with pytest.raises(ValueError, match="Invalid parameter"):
        bam.set_params(X_a_=X)


def test_ood_kmeans_random_state_and_pipeline_fit_predict() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 8))

    km1 = HopfieldOOD(mode="kmeans", n_prototypes=6, random_state=42).fit(X)
    km2 = HopfieldOOD(mode="kmeans", n_prototypes=6, random_state=42).fit(X)
    np.testing.assert_allclose(km1.memories_, km2.memories_)
    np.testing.assert_allclose(km1.score_samples(X), km2.score_samples(X))

    preds_fp = HopfieldOOD(mode="kmeans", n_prototypes=6, random_state=42).fit_predict(X)
    np.testing.assert_array_equal(preds_fp, km1.predict(X))


def test_clustering_attractor_merge_is_order_independent() -> None:
    """Shuffling input rows must produce the exact same cluster partition."""
    rng = np.random.default_rng(123)
    X = np.vstack(
        [
            rng.normal(loc=[1.0, 0.0, 0.0, 0.0], scale=0.08, size=(15, 4)),
            rng.normal(loc=[0.0, 1.0, 0.0, 0.0], scale=0.08, size=(15, 4)),
            rng.normal(loc=[0.0, 0.0, 1.0, 0.0], scale=0.08, size=(15, 4)),
        ]
    )
    perm = rng.permutation(X.shape[0])

    c_orig = HopfieldMetastableCluster(beta=8.0, merge_tol=1e-2).fit(X)
    c_shuf = HopfieldMetastableCluster(beta=8.0, merge_tol=1e-2).fit(X[perm])

    assert c_orig.n_clusters_ == c_shuf.n_clusters_
    assert adjusted_rand_score(c_orig.labels_[perm], c_shuf.labels_) == pytest.approx(1.0)


def test_bam_transform_in_sklearn_pipeline() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(16, 6))
    pipe = make_pipeline(StandardScaler(), HopfieldBAM(beta=10.0))
    Z = pipe.fit_transform(X)
    assert Z.shape == X.shape
    assert np.all(np.isfinite(Z))
