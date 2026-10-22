"""
Core mathematical primitives for Modern Continuous Hopfield Networks.
Implements shifted log-sum-exp energy, Rectified Lagrangian (RecLag, AAAI 2025),
softmax associative retrieval weights, and state update steps.
"""

from __future__ import annotations

import numpy as np


def validate_beta(beta: float) -> float:
    """Validates that inverse temperature beta is strictly positive and finite."""
    b = float(beta)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError(f"beta must be strictly positive and finite, got {beta!r}.")
    return b


def validate_array(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Converts input to 2D float64 ndarray and rejects NaN/Inf values."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization safe against zero-norm vectors."""
    arr = validate_array(X)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def logsumexp_scaled(sim: np.ndarray, beta: float) -> np.ndarray:
    """
    Numerically stable computation of beta^{-1} * log(mean_i exp(beta * sim_{q,i})).
    Converges to max_i sim_{q,i} as beta -> infinity, and to mean_i sim_{q,i} as beta -> 0.
    """
    b = validate_beta(beta)
    z = b * sim
    m = np.max(z, axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.mean(np.exp(z - m), axis=1))) / b


def attention_weights(memories: np.ndarray, queries: np.ndarray, beta: float) -> np.ndarray:
    """Computes softmax(beta * queries @ memories.T) row-wise."""
    b = validate_beta(beta)
    mem = validate_array(memories, "memories")
    qry = validate_array(queries, "queries")
    sim = qry @ mem.T
    z = b * sim
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def hopfield_energy(
    memories: np.ndarray,
    queries: np.ndarray,
    beta: float = 8.0,
) -> np.ndarray:
    """
    Modern Hopfield Lyapunov Energy (Ramsauer et al., ICLR 2021):
        E(xi) = -beta^{-1} log(sum_i exp(beta * x_i^T xi)) + 0.5 * ||xi||^2
    Guaranteed to decrease monotonically under xi_{t+1} = X^T softmax(beta * X xi_t).
    """
    b = validate_beta(beta)
    mem = validate_array(memories, "memories")
    qry = validate_array(queries, "queries")
    sim = qry @ mem.T
    z = b * sim
    m = np.max(z, axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.sum(np.exp(z - m), axis=1))) / b
    quad = 0.5 * np.sum(qry * qry, axis=1)
    return -lse + quad


def hopfield_update_step(
    memories: np.ndarray,
    queries: np.ndarray,
    beta: float = 8.0,
) -> np.ndarray:
    """Single concave-convex procedure (CCCP) update step: xi_{t+1} = P * X."""
    weights = attention_weights(memories, queries, beta=beta)
    return weights @ validate_array(memories, "memories")


def reclag_score(
    memories: np.ndarray,
    queries: np.ndarray,
    beta: float = 8.0,
    tau_rect: float = 0.5,
) -> np.ndarray:
    """
    Rectified Lagrangian OOD anomaly score (Moriai et al., AAAI 2025).
    Applies ReLU(sim - tau_rect) before energy aggregation so low-density OOD
    queries collapse to the trivial origin attractor (score = 0.0).
    """
    b = validate_beta(beta)
    mem = validate_array(memories, "memories")
    qry = validate_array(queries, "queries")
    sim = qry @ mem.T
    rect = np.maximum(0.0, sim - float(tau_rect))
    max_rect = np.max(rect, axis=1)
    active = max_rect > 0.0
    scores = np.zeros(qry.shape[0], dtype=float)
    if np.any(active):
        scores[active] = -logsumexp_scaled(rect[active], beta=b)
    return scores
