"""
BOLT-001: Modern Hopfield Energy Out-of-Distribution & Novelty Detector (`HopfieldOOD`).
Compatible with scikit-learn (BaseEstimator, OutlierMixin) and PyOD conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .energy import (
    attention_weights,
    l2_normalize,
    logsumexp_scaled,
    reclag_score,
    validate_array,
    validate_beta,
)

VALID_OOD_MODES = {"full", "she", "kmeans", "reclag", "boosting"}


class HopfieldOOD:
    """
    Out-of-Distribution (OOD) and Novelty Detector based on Modern Hopfield Energy.
    """

    def __init__(
        self,
        beta: float = 8.0,
        mode: str = "full",
        tau_rect: float = 0.5,
        n_prototypes: int = 32,
        contamination: float = 0.05,
        normalize: bool = True,
    ) -> None:
        self.beta = validate_beta(beta)
        mode_clean = mode.lower()
        if mode_clean not in VALID_OOD_MODES:
            modes = sorted(VALID_OOD_MODES)
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {modes}.")
        if not (0.0 < float(contamination) < 0.5):
            raise ValueError(f"contamination must be in (0.0, 0.5), got {contamination!r}.")
        self.mode = mode_clean
        self.tau_rect = float(tau_rect)
        self.n_prototypes = int(n_prototypes)
        self.contamination = float(contamination)
        self.normalize = bool(normalize)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (scikit-learn BaseEstimator protocol)."""
        return {
            "beta": self.beta,
            "mode": self.mode,
            "tau_rect": self.tau_rect,
            "n_prototypes": self.n_prototypes,
            "contamination": self.contamination,
            "normalize": self.normalize,
        }

    def set_params(self, **params: Any) -> HopfieldOOD:
        """Set the parameters of this estimator (scikit-learn BaseEstimator protocol)."""
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k!r} for HopfieldOOD.")
            setattr(self, k, v)
        self.beta = validate_beta(self.beta)
        if self.mode not in VALID_OOD_MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}.")
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "memories_"):
            raise ValueError("This HopfieldOOD instance is not fitted yet. Call 'fit' first.")

    def _prep(self, X: np.ndarray) -> np.ndarray:
        arr = validate_array(X)
        return l2_normalize(arr) if self.normalize else arr

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        X_outliers: np.ndarray | None = None,
    ) -> HopfieldOOD:
        X_proc = self._prep(X)

        if self.mode == "she":
            if y is None:
                prototypes = np.mean(X_proc, axis=0, keepdims=True)
            else:
                labels = np.unique(y)
                prototypes = np.vstack([np.mean(X_proc[y == c], axis=0) for c in labels])
            self.memories_ = l2_normalize(prototypes) if self.normalize else prototypes
        elif self.mode == "kmeans":
            k = min(self.n_prototypes, X_proc.shape[0])
            rng = np.random.default_rng(42)
            idx = rng.choice(X_proc.shape[0], size=k, replace=False)
            centers = X_proc[idx].copy()
            for _ in range(15):
                dists = np.linalg.norm(X_proc[:, None, :] - centers[None, :, :], axis=2)
                assign = np.argmin(dists, axis=1)
                for j in range(k):
                    mask = assign == j
                    if np.any(mask):
                        centers[j] = np.mean(X_proc[mask], axis=0)
            self.memories_ = l2_normalize(centers) if self.normalize else centers
        else:
            self.memories_ = X_proc

        self.outlier_memories_: np.ndarray | None = None
        if X_outliers is not None:
            self.outlier_memories_ = self._prep(X_outliers)

        train_scores = self.decision_function(X)
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Computes anomaly scores for X (higher = more anomalous / higher energy)."""
        self._check_is_fitted()
        Q = self._prep(X)
        if self.mode == "reclag":
            return reclag_score(self.memories_, Q, beta=self.beta, tau_rect=self.tau_rect)

        sim_id = Q @ self.memories_.T
        score_id = -logsumexp_scaled(sim_id, beta=self.beta)

        if self.outlier_memories_ is not None:
            sim_ood = Q @ self.outlier_memories_.T
            score_ood = logsumexp_scaled(sim_ood, beta=self.beta)
            return score_id + score_ood

        return score_id

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts inlier (`1`) vs outlier (`-1`) following scikit-learn OutlierMixin."""
        scores = self.decision_function(X)
        return np.where(scores > self.threshold_, -1, 1)

    def explain(self, X: np.ndarray, top_k: int = 3) -> dict[str, Any]:
        """Returns associative memory attribution (top-k indices and weights)."""
        self._check_is_fitted()
        Q = self._prep(X)
        weights = attention_weights(self.memories_, Q, beta=self.beta)
        k = min(top_k, self.memories_.shape[0])
        top_idx = np.argsort(-weights, axis=1)[:, :k]
        top_w = np.take_along_axis(weights, top_idx, axis=1)
        scores = self.decision_function(X)
        return {
            "anomaly_scores": scores,
            "top_indices": top_idx,
            "top_weights": top_w,
            "full_weights_sum": np.sum(weights, axis=1),
        }
