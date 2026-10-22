"""
BOLT-003: Continuous Hetero-Associative Bidirectional Associative Memory (`HopfieldBAM`).
Bridges two distinct knowledge spaces (e.g., Obsidian Vault Notes <-> Technical Book Chunks)
via iterative cross-attention reverberation and detects unindexed Knowledge Gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .energy import l2_normalize, logsumexp_scaled, validate_array, validate_beta


class HopfieldBAM:
    """
    Modern Continuous Bidirectional Associative Memory (BAM).
    """

    def __init__(
        self,
        beta: float = 12.0,
        max_iter: int = 15,
        tol: float = 1e-6,
        gap_quantile: float = 0.98,
        normalize: bool = True,
    ) -> None:
        self.beta = validate_beta(beta)
        if not (0.0 < float(gap_quantile) <= 1.0):
            raise ValueError(f"gap_quantile must be in (0.0, 1.0], got {gap_quantile!r}.")
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.gap_quantile = float(gap_quantile)
        self.normalize = bool(normalize)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (scikit-learn BaseEstimator protocol)."""
        return {
            "beta": self.beta,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "gap_quantile": self.gap_quantile,
            "normalize": self.normalize,
        }

    def set_params(self, **params: Any) -> HopfieldBAM:
        """Set parameters for this estimator."""
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k!r} for HopfieldBAM.")
            setattr(self, k, v)
        self.beta = validate_beta(self.beta)
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "X_a_"):
            raise ValueError("This HopfieldBAM instance is not fitted yet. Call 'fit' first.")

    def _prep(self, M: np.ndarray) -> np.ndarray:
        arr = validate_array(M)
        return l2_normalize(arr) if self.normalize else arr

    def fit(self, X_domain_a: np.ndarray, Y_domain_b: np.ndarray) -> HopfieldBAM:
        X_a = self._prep(X_domain_a)
        Y_b = self._prep(Y_domain_b)
        if X_a.shape[0] != Y_b.shape[0]:
            msg = (
                "Domain A and Domain B must have the same number of "
                "paired/aligned associative slots."
            )
            raise ValueError(msg)
        self.X_a_ = X_a
        self.Y_b_ = Y_b

        sim_train = self.X_a_ @ self.X_a_.T
        energies = -logsumexp_scaled(sim_train, beta=self.beta)
        self.gap_threshold_ = float(np.quantile(energies, self.gap_quantile)) + 0.05
        return self

    def resonate(self, query_a: np.ndarray, top_k: int = 3) -> dict[str, Any]:
        """
        Reverberates `query_a` between Domain A and Domain B (`A -> B -> A`)
        until convergence and returns the resonant Domain B concepts.
        """
        self._check_is_fitted()
        q_a = self._prep(query_a)
        n_queries = q_a.shape[0]
        converged = np.zeros(n_queries, dtype=bool)
        iterations = np.zeros(n_queries, dtype=int)

        curr_a = q_a.copy()
        curr_b = np.zeros((n_queries, self.Y_b_.shape[1]), dtype=float)
        probs = np.zeros((n_queries, self.X_a_.shape[0]), dtype=float)

        for step in range(1, self.max_iter + 1):
            z_a = self.beta * (curr_a @ self.X_a_.T)
            z_a -= np.max(z_a, axis=1, keepdims=True)
            p_a = np.exp(z_a)
            p_a /= np.sum(p_a, axis=1, keepdims=True)
            next_b = p_a @ self.Y_b_
            if self.normalize:
                next_b = l2_normalize(next_b)

            z_b = self.beta * (next_b @ self.Y_b_.T)
            z_b -= np.max(z_b, axis=1, keepdims=True)
            p_b = np.exp(z_b)
            p_b /= np.sum(p_b, axis=1, keepdims=True)
            next_a = p_b @ self.X_a_
            if self.normalize:
                next_a = l2_normalize(next_a)

            diff = np.linalg.norm(next_a - curr_a, axis=1)
            newly_conv = (diff < self.tol) & (~converged)
            converged[newly_conv] = True
            iterations[newly_conv] = step

            active = ~converged | newly_conv
            curr_a[active] = next_a[active]
            curr_b[active] = next_b[active]
            probs[active] = p_b[active]

            if np.all(converged):
                break

        iterations[~converged] = self.max_iter

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

    def detect_knowledge_gap(self, query_a: np.ndarray) -> dict[str, Any]:
        """Determines whether `query_a` falls outside the associative basins of the vault."""
        self._check_is_fitted()
        q_a = self._prep(query_a)
        sim = q_a @ self.X_a_.T
        energy = -logsumexp_scaled(sim, beta=self.beta)
        is_gap = energy > self.gap_threshold_
        return {
            "energy": energy,
            "threshold": self.gap_threshold_,
            "is_gap": is_gap,
        }
