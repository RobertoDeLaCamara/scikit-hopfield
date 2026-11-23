"""
BOLT-003: Continuous Hetero-Associative Bidirectional Associative Memory (`HopfieldBAM`).
Bridges two distinct knowledge spaces (e.g., Obsidian Vault Notes <-> Technical Book Chunks)
via iterative cross-attention reverberation and detects unindexed Knowledge Gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .energy import l2_normalize, logsumexp_scaled, validate_array, validate_beta


def _validate_gap_margin(gap_margin: float) -> float:
    margin = float(gap_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(f"gap_margin must be finite and >= 0.0, got {gap_margin!r}.")
    return margin


class HopfieldBAM(TransformerMixin, BaseEstimator):
    """
    Modern Continuous Bidirectional Associative Memory (BAM).
    """

    def __init__(
        self,
        beta: float = 12.0,
        max_iter: int = 15,
        tol: float = 1e-6,
        gap_quantile: float = 0.98,
        gap_margin: float = 0.05,
        normalize: bool = True,
    ) -> None:
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.gap_quantile = gap_quantile
        self.gap_margin = gap_margin
        self.normalize = normalize

    def _validate_hyperparams(self) -> tuple[float, int, float, float, float]:
        beta = validate_beta(self.beta)
        max_iter = int(self.max_iter)
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter!r}.")
        tol = float(self.tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError(f"tol must be finite and >= 0.0, got {self.tol!r}.")
        gap_quantile = float(self.gap_quantile)
        if not np.isfinite(gap_quantile) or not (0.0 < gap_quantile <= 1.0):
            raise ValueError(
                f"gap_quantile must be in (0.0, 1.0], got {self.gap_quantile!r}."
            )
        gap_margin = _validate_gap_margin(self.gap_margin)
        return beta, max_iter, tol, gap_quantile, gap_margin

    def _prep_query(self, M: np.ndarray) -> np.ndarray:
        arr = validate_array(M)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but HopfieldBAM is expecting "
                f"{self.n_features_in_} features as input."
            )
        return l2_normalize(arr) if self.normalize else arr

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HopfieldBAM:
        """
        Fits the bidirectional memory bank pairing Domain A (`X`) with Domain B (`y`).
        When `y` is None (or a 1D target passed by a scikit-learn Pipeline), defaults to
        auto-associative Domain B = Domain A so `transform(X)` remains well-defined.
        """
        beta, max_iter, tol, gap_quantile, gap_margin = self._validate_hyperparams()
        X_val = validate_data(
            self, X, dtype=np.float64, ensure_min_samples=1, ensure_min_features=1
        )
        X_a = l2_normalize(X_val) if self.normalize else X_val

        if y is None:
            Y_b = X_a.copy()
        else:
            y_raw = np.asarray(y, dtype=float)
            if y_raw.ndim == 1:
                if y_raw.shape[0] != X_a.shape[0]:
                    msg = (
                        "Domain A and Domain B must have the same number of "
                        "paired/aligned associative slots."
                    )
                    raise ValueError(msg)
                # Standard scikit-learn supervised 1D `y` (e.g. Pipeline/check_estimator):
                # treat as unsupervised auto-associative memory over Domain A.
                Y_b = X_a.copy()
            else:
                Y_val = validate_array(y_raw, name="y")
                if X_a.shape[0] != Y_val.shape[0]:
                    msg = (
                        "Domain A and Domain B must have the same number of "
                        "paired/aligned associative slots."
                    )
                    raise ValueError(msg)
                Y_b = l2_normalize(Y_val) if self.normalize else Y_val

        self._beta_ = beta
        self._max_iter_ = max_iter
        self._tol_ = tol
        self.n_iter_ = 1
        self.X_a_ = X_a
        self.Y_b_ = Y_b

        # Associative memory banks frequently store unique/orthogonal topics (1 slot per concept),
        # so leave-one-out masking would misclassify stored topics as gaps. Use an explicit
        # `gap_margin` buffer above the in-bank energy quantile instead.
        sim_train = self.X_a_ @ self.X_a_.T
        energies = -logsumexp_scaled(sim_train, beta=beta)
        self.gap_threshold_ = float(np.quantile(energies, gap_quantile)) + gap_margin
        return self

    def _resonate_prepared(self, q_a: np.ndarray, top_k: int = 3) -> dict[str, Any]:
        n_queries = q_a.shape[0]
        converged = np.zeros(n_queries, dtype=bool)
        iterations = np.zeros(n_queries, dtype=int)

        curr_a = q_a.copy()
        curr_b = np.zeros((n_queries, self.Y_b_.shape[1]), dtype=float)
        probs = np.zeros((n_queries, self.X_a_.shape[0]), dtype=float)

        for step in range(1, self._max_iter_ + 1):
            z_a = self._beta_ * (curr_a @ self.X_a_.T)
            z_a -= np.max(z_a, axis=1, keepdims=True)
            p_a = np.exp(z_a)
            p_a /= np.sum(p_a, axis=1, keepdims=True)
            next_b = p_a @ self.Y_b_
            if self.normalize:
                next_b = l2_normalize(next_b)

            z_b = self._beta_ * (next_b @ self.Y_b_.T)
            z_b -= np.max(z_b, axis=1, keepdims=True)
            p_b = np.exp(z_b)
            p_b /= np.sum(p_b, axis=1, keepdims=True)
            next_a = p_b @ self.X_a_
            if self.normalize:
                next_a = l2_normalize(next_a)

            diff = np.linalg.norm(next_a - curr_a, axis=1)
            newly_conv = (diff < self._tol_) & (~converged)
            converged[newly_conv] = True
            iterations[newly_conv] = step

            active = ~converged | newly_conv
            curr_a[active] = next_a[active]
            curr_b[active] = next_b[active]
            probs[active] = p_b[active]

            if np.all(converged):
                break

        iterations[~converged] = self._max_iter_

        k = min(top_k, self.Y_b_.shape[0])
        top_b_idx = np.argsort(-probs, axis=1)[:, :k]
        top_b_weights = np.take_along_axis(probs, top_b_idx, axis=1)

        return {
            "resonant_a": curr_a,
            "resonant_b": curr_b,
            "top_b_indices": top_b_idx,
            "top_b_weights": top_b_weights,
            "converged": converged,
            "iterations": iterations,
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects Domain A queries `X` to their resonant Domain B representations."""
        check_is_fitted(self, ["X_a_", "Y_b_", "gap_threshold_"])
        Q = validate_data(self, X, dtype=np.float64, reset=False, ensure_min_samples=1)
        q_a = l2_normalize(Q) if self.normalize else Q
        result: np.ndarray = self._resonate_prepared(q_a, top_k=1)["resonant_b"]
        return result

    def resonate(self, query_a: np.ndarray, top_k: int = 3) -> dict[str, Any]:
        """
        Reverberates `query_a` between Domain A and Domain B (`A -> B -> A`)
        until convergence and returns the resonant Domain B concepts.
        """
        check_is_fitted(self, ["X_a_", "Y_b_", "gap_threshold_"])
        q_a = self._prep_query(query_a)
        return self._resonate_prepared(q_a, top_k=top_k)

    def detect_knowledge_gap(self, query_a: np.ndarray) -> dict[str, Any]:
        """Determines whether `query_a` falls outside the associative basins of the vault."""
        check_is_fitted(self, ["X_a_", "Y_b_", "gap_threshold_"])
        q_a = self._prep_query(query_a)
        sim = q_a @ self.X_a_.T
        energy = -logsumexp_scaled(sim, beta=self._beta_)
        is_gap = energy > self.gap_threshold_
        return {
            "energy": energy,
            "threshold": self.gap_threshold_,
            "is_gap": is_gap,
        }
