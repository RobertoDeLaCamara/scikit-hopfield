"""
BOLT-001: Modern Hopfield Energy Out-of-Distribution & Novelty Detector (`HopfieldOOD`).
Compatible with scikit-learn (`BaseEstimator`, `OutlierMixin`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted, validate_data

from .energy import (
    attention_weights,
    l2_normalize,
    logsumexp_scaled,
    reclag_score_from_sim,
    validate_beta,
)

VALID_OOD_MODES = {"full", "she", "kmeans", "reclag"}
_SELF_MEMORY_MODES = {"full", "reclag"}


class HopfieldOOD(OutlierMixin, BaseEstimator):
    """
    Out-of-Distribution (OOD) and Novelty Detector based on Modern Hopfield Energy.

    Scoring conventions follow `scikit-learn` (`OutlierMixin`):
    - `energy_score(X)`: raw Hopfield anomaly energy (higher = more anomalous / OOD).
    - `score_samples(X)`: opposite of anomaly energy (`-energy_score(X)`, higher = more normal).
    - `decision_function(X)`: shifted normality score (`score_samples(X) - offset_`;
      `>= 0` inlier, `< 0` outlier).
    - `predict(X)` / `fit_predict(X)`: `1` for inliers, `-1` for outliers.
    """

    def __init__(
        self,
        beta: float = 8.0,
        mode: str = "full",
        tau_rect: float = 0.5,
        n_prototypes: int = 32,
        contamination: float = 0.05,
        normalize: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.beta = beta
        self.mode = mode
        self.tau_rect = tau_rect
        self.n_prototypes = n_prototypes
        self.contamination = contamination
        self.normalize = normalize
        self.random_state = random_state

    def _validate_hyperparams(self) -> tuple[float, str, float, int, float]:
        beta = validate_beta(self.beta)
        if not isinstance(self.mode, str) or self.mode.lower() not in VALID_OOD_MODES:
            modes = sorted(VALID_OOD_MODES)
            raise ValueError(f"Unsupported mode {self.mode!r}; expected one of {modes}.")
        mode = self.mode.lower()

        contamination = float(self.contamination)
        if not np.isfinite(contamination) or not (0.0 < contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0.0, 0.5), got {self.contamination!r}."
            )

        n_prototypes = int(self.n_prototypes)
        if n_prototypes < 1:
            raise ValueError(f"n_prototypes must be >= 1, got {self.n_prototypes!r}.")

        tau_rect = float(self.tau_rect)
        if not np.isfinite(tau_rect):
            raise ValueError(f"tau_rect must be finite, got {self.tau_rect!r}.")

        return beta, mode, tau_rect, n_prototypes, contamination

    def _normalize_if_enabled(self, X: np.ndarray) -> np.ndarray:
        return l2_normalize(X) if self.normalize else X

    def _energy_from_proc(self, Q: np.ndarray) -> np.ndarray:
        if self._mode_ == "reclag":
            sim = Q @ self.memories_.T
            return reclag_score_from_sim(sim, beta=self._beta_, tau_rect=self._tau_rect_)

        sim_id = Q @ self.memories_.T
        score_id = -logsumexp_scaled(sim_id, beta=self._beta_)

        if self.outlier_memories_ is not None:
            sim_ood = Q @ self.outlier_memories_.T
            score_ood = logsumexp_scaled(sim_ood, beta=self._beta_)
            return score_id + score_ood

        return score_id

    def _calibration_energies(self, X_proc: np.ndarray) -> np.ndarray:
        """
        Computes in-distribution anomaly energies for threshold calibration.
        For modes storing raw training patterns (`full`, `reclag`), uses exact
        leave-one-out (excluding self-similarity on the diagonal) so the threshold
        is not biased downward by exact self-matches.
        """
        n_samples = X_proc.shape[0]
        if self._mode_ not in _SELF_MEMORY_MODES or n_samples <= 1:
            return self._energy_from_proc(X_proc)

        sim_id = X_proc @ self.memories_.T
        off_diag = ~np.eye(n_samples, dtype=bool)
        sim_id_loo = sim_id[off_diag].reshape(n_samples, n_samples - 1)

        if self._mode_ == "reclag":
            return reclag_score_from_sim(
                sim_id_loo, beta=self._beta_, tau_rect=self._tau_rect_
            )

        score_id = -logsumexp_scaled(sim_id_loo, beta=self._beta_)
        if self.outlier_memories_ is not None:
            sim_ood = X_proc @ self.outlier_memories_.T
            score_ood = logsumexp_scaled(sim_ood, beta=self._beta_)
            return score_id + score_ood
        return score_id

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        X_outliers: np.ndarray | None = None,
    ) -> HopfieldOOD:
        beta, mode, tau_rect, n_prototypes, contamination = self._validate_hyperparams()
        X_val = validate_data(
            self, X, dtype=np.float64, ensure_min_samples=1, ensure_min_features=1
        )
        X_proc = self._normalize_if_enabled(X_val)

        self._beta_ = beta
        self._mode_ = mode
        self._tau_rect_ = tau_rect

        if mode == "she":
            if y is None:
                prototypes = np.mean(X_proc, axis=0, keepdims=True)
            else:
                y_arr = np.asarray(y)
                if y_arr.shape[0] != X_proc.shape[0]:
                    raise ValueError(
                        f"y length ({y_arr.shape[0]}) must match X rows ({X_proc.shape[0]})."
                    )
                labels = np.unique(y_arr)
                prototypes = np.vstack([np.mean(X_proc[y_arr == c], axis=0) for c in labels])
            self.memories_ = self._normalize_if_enabled(prototypes)
        elif mode == "kmeans":
            n_distinct = np.unique(X_proc, axis=0).shape[0]
            k = min(n_prototypes, X_proc.shape[0], n_distinct)
            if k == 1:
                centers = np.mean(X_proc, axis=0, keepdims=True)
            else:
                km = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init="auto",
                    random_state=self.random_state,
                )
                km.fit(X_proc)
                centers = km.cluster_centers_
            self.memories_ = self._normalize_if_enabled(centers)
        else:
            self.memories_ = X_proc

        self.outlier_memories_: np.ndarray | None = None
        if X_outliers is not None:
            if mode == "reclag":
                raise ValueError(
                    "X_outliers is not supported when mode='reclag'; "
                    "use mode='full', 'she', or 'kmeans'."
                )
            X_out_val = validate_data(
                self, X_outliers, dtype=np.float64, reset=False, ensure_min_samples=1
            )
            self.outlier_memories_ = self._normalize_if_enabled(X_out_val)

        train_energies = self._calibration_energies(X_proc)
        self.threshold_ = float(np.quantile(train_energies, 1.0 - contamination))
        self.offset_ = -self.threshold_
        return self

    def energy_score(self, X: np.ndarray) -> np.ndarray:
        """Computes raw Hopfield anomaly energy for X (higher = more anomalous / OOD)."""
        check_is_fitted(self, ["memories_", "threshold_", "offset_"])
        Q = validate_data(self, X, dtype=np.float64, reset=False, ensure_min_samples=1)
        Q_proc = self._normalize_if_enabled(Q)
        return self._energy_from_proc(Q_proc)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Opposite of anomaly energy (higher = more normal, scikit-learn OutlierMixin)."""
        return -self.energy_score(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Shifted normality score (`score_samples(X) - offset_`).
        Negative scores indicate outliers; non-negative scores indicate inliers.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts inlier (`1`) vs outlier (`-1`) following scikit-learn OutlierMixin."""
        return np.where(self.decision_function(X) < 0.0, -1, 1)

    def explain(self, X: np.ndarray, top_k: int = 3) -> dict[str, Any]:
        """Returns associative memory attribution (top-k indices, weights, and energy)."""
        check_is_fitted(self, ["memories_", "threshold_", "offset_"])
        Q = validate_data(self, X, dtype=np.float64, reset=False, ensure_min_samples=1)
        Q_proc = self._normalize_if_enabled(Q)
        weights = attention_weights(self.memories_, Q_proc, beta=self._beta_)
        k = min(top_k, self.memories_.shape[0])
        top_idx = np.argsort(-weights, axis=1)[:, :k]
        top_w = np.take_along_axis(weights, top_idx, axis=1)
        energies = self._energy_from_proc(Q_proc)
        return {
            "anomaly_scores": energies,
            "top_indices": top_idx,
            "top_weights": top_w,
            "full_weights_sum": np.sum(weights, axis=1),
        }
