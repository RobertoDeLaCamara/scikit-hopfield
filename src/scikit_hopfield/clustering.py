"""
BOLT-002: Metastable Bifurcation Clustering via Continuous Hopfield Dynamics.
Evolves data points along the Modern Hopfield update rule until they coalesce
into metastable attractors at inverse temperature `beta`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .energy import hopfield_update_step, l2_normalize, validate_array, validate_beta


class HopfieldMetastableCluster:
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
        self.beta = validate_beta(beta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.merge_tol = float(merge_tol)
        self.normalize = bool(normalize)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (scikit-learn BaseEstimator protocol)."""
        return {
            "beta": self.beta,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "merge_tol": self.merge_tol,
            "normalize": self.normalize,
        }

    def set_params(self, **params: Any) -> HopfieldMetastableCluster:
        """Set parameters for this estimator."""
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k!r} for HopfieldMetastableCluster.")
            setattr(self, k, v)
        self.beta = validate_beta(self.beta)
        return self

    def _evolve_states(self, memories: np.ndarray, states: np.ndarray, beta: float) -> np.ndarray:
        curr = states.copy()
        for _ in range(self.max_iter):
            nxt = hopfield_update_step(memories, curr, beta=beta)
            if self.normalize:
                nxt = l2_normalize(nxt)
            if np.max(np.linalg.norm(nxt - curr, axis=1)) < self.tol:
                curr = nxt
                break
            curr = nxt
        return curr

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HopfieldMetastableCluster:
        X_proc = l2_normalize(X) if self.normalize else validate_array(X)
        self.memories_ = X_proc

        attractors = self._evolve_states(self.memories_, X_proc, beta=self.beta)
        self.sample_attractors_ = attractors

        centers: list[np.ndarray] = []
        labels = np.full(X_proc.shape[0], -1, dtype=int)

        for i, vec in enumerate(attractors):
            assigned = False
            for c_idx, center in enumerate(centers):
                if np.linalg.norm(vec - center) <= self.merge_tol:
                    labels[i] = c_idx
                    assigned = True
                    break
            if not assigned:
                labels[i] = len(centers)
                centers.append(vec.copy())

        self.cluster_centers_ = np.vstack(centers)
        self.labels_ = labels
        self.n_clusters_ = len(centers)
        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).labels_

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
