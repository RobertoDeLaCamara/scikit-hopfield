"""
BOLT-002: Metastable Bifurcation Clustering via Continuous Hopfield Dynamics.
Evolves data points along the Modern Hopfield update rule until they coalesce
into metastable attractors at inverse temperature `beta`.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import validate_data

from .energy import hopfield_update_step, l2_normalize, validate_beta


class HopfieldMetastableCluster(ClusterMixin, BaseEstimator):
    """
    Clustering estimator based on Metastable States of Modern Hopfield Networks.
    """

    def __init__(
        self,
        beta: float = 8.0,
        max_iter: int = 50,
        tol: float = 1e-6,
        merge_tol: float = 1e-2,
        normalize: bool = True,
    ) -> None:
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.merge_tol = merge_tol
        self.normalize = normalize

    def _validate_hyperparams(self) -> tuple[float, int, float, float]:
        beta = validate_beta(self.beta)
        max_iter = int(self.max_iter)
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter!r}.")
        tol = float(self.tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError(f"tol must be finite and >= 0.0, got {self.tol!r}.")
        merge_tol = float(self.merge_tol)
        if not np.isfinite(merge_tol) or merge_tol < 0.0:
            raise ValueError(f"merge_tol must be finite and >= 0.0, got {self.merge_tol!r}.")
        return beta, max_iter, tol, merge_tol

    def _evolve_states(
        self,
        memories: np.ndarray,
        states: np.ndarray,
        beta: float,
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, int]:
        curr = states.copy()
        n_iter = 0
        for step in range(1, max_iter + 1):
            n_iter = step
            nxt = hopfield_update_step(memories, curr, beta=beta)
            if self.normalize:
                nxt = l2_normalize(nxt)
            if np.max(np.linalg.norm(nxt - curr, axis=1)) < tol:
                curr = nxt
                break
            curr = nxt
        return curr, n_iter

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HopfieldMetastableCluster:
        beta, max_iter, tol, merge_tol = self._validate_hyperparams()
        X_val = validate_data(
            self, X, dtype=np.float64, ensure_min_samples=1, ensure_min_features=1
        )
        X_proc = l2_normalize(X_val) if self.normalize else X_val
        self.memories_ = X_proc

        attractors, n_iter = self._evolve_states(
            self.memories_, X_proc, beta=beta, max_iter=max_iter, tol=tol
        )
        self.n_iter_ = n_iter
        self.sample_attractors_ = attractors

        if attractors.shape[0] == 1:
            labels = np.array([0], dtype=int)
            centers = attractors.copy()
        else:
            # Order-independent hierarchical merging of converged attractors.
            threshold = max(merge_tol, float(np.finfo(float).eps))
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage="complete",
            )
            labels = agg.fit_predict(attractors).astype(int)
            n_clusters = int(np.max(labels)) + 1
            centers = np.vstack(
                [np.mean(attractors[labels == c], axis=0) for c in range(n_clusters)]
            )

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters_ = int(centers.shape[0])
        return self

    def bifurcation_spectrum(self, X: np.ndarray, betas: list[float]) -> dict[float, int]:
        """Computes the number of emergent metastable clusters across a spectrum of beta values."""
        spectrum: dict[float, int] = {}
        for b in betas:
            clone = HopfieldMetastableCluster(
                beta=b,
                max_iter=self.max_iter,
                tol=self.tol,
                merge_tol=self.merge_tol,
                normalize=self.normalize,
            ).fit(X)
            spectrum[float(b)] = clone.n_clusters_
        return spectrum
