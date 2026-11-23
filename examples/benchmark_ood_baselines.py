#!/usr/bin/env python3
"""
Reproducible OOD / Novelty Benchmark: `HopfieldOOD` vs `IsolationForest`,
`LocalOutlierFactor`, and `kNN` across standard scikit-learn tabular datasets
and high-dimensional normalized embeddings.

Usage:
    python examples/benchmark_ood_baselines.py
    python examples/benchmark_ood_baselines.py --json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scikit_hopfield import HopfieldOOD


@dataclass
class BenchmarkRow:
    dataset: str
    detector: str
    auroc: float
    auprc: float
    unseen_id_fpr: float
    elapsed_ms: float


def _build_datasets(
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of `(name, X_id_train, y_id_train, X_id_test, X_ood_test)` tuples.
    All tabular datasets are standardized using fit-only statistics on `X_id_train`.
    """
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    # 1. Digits (64D): classes 0-7 = ID, classes 8-9 = OOD
    digits = load_digits()
    id_mask = digits.target < 8
    X_id_tr, X_id_te, y_id_tr, _ = train_test_split(
        digits.data[id_mask],
        digits.target[id_mask],
        test_size=0.30,
        random_state=seed,
        stratify=digits.target[id_mask],
    )
    scaler = StandardScaler().fit(X_id_tr)
    datasets.append(
        (
            "digits (64d, 8-9 OOD)",
            scaler.transform(X_id_tr),
            y_id_tr,
            scaler.transform(X_id_te),
            scaler.transform(digits.data[~id_mask]),
        )
    )

    # 2. Breast Cancer (30D): benign (1) = ID, malignant (0) = OOD
    bc = load_breast_cancer()
    id_mask = bc.target == 1
    X_id_tr, X_id_te = train_test_split(
        bc.data[id_mask], test_size=0.30, random_state=seed
    )
    scaler = StandardScaler().fit(X_id_tr)
    datasets.append(
        (
            "breast_cancer (30d)",
            scaler.transform(X_id_tr),
            np.zeros(X_id_tr.shape[0], dtype=int),
            scaler.transform(X_id_te),
            scaler.transform(bc.data[~id_mask]),
        )
    )

    # 3. Wine (13D): classes 0-1 = ID, class 2 = OOD
    wine = load_wine()
    id_mask = wine.target < 2
    X_id_tr, X_id_te, y_id_tr, _ = train_test_split(
        wine.data[id_mask],
        wine.target[id_mask],
        test_size=0.30,
        random_state=seed,
        stratify=wine.target[id_mask],
    )
    scaler = StandardScaler().fit(X_id_tr)
    datasets.append(
        (
            "wine (13d, cls-2 OOD)",
            scaler.transform(X_id_tr),
            y_id_tr,
            scaler.transform(X_id_te),
            scaler.transform(wine.data[~id_mask]),
        )
    )

    # 4. High-D Normalized Embeddings (128D): 5 ID semantic clusters + shifted OOD
    rng = np.random.default_rng(seed)
    dim = 128
    centers = rng.normal(size=(5, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X_id_tr = np.vstack([centers[c] + rng.normal(scale=0.14, size=(120, dim)) for c in range(5)])
    y_id_tr = np.repeat(np.arange(5), 120)
    X_id_te = np.vstack([centers[c] + rng.normal(scale=0.14, size=(60, dim)) for c in range(5)])
    ood_centers = rng.normal(size=(2, dim))
    ood_centers /= np.linalg.norm(ood_centers, axis=1, keepdims=True)
    X_ood_te = np.vstack(
        [ood_centers[c] + rng.normal(scale=0.14, size=(100, dim)) for c in range(2)]
    )
    datasets.append(("embeddings (128d, 5-cls)", X_id_tr, y_id_tr, X_id_te, X_ood_te))

    return datasets


def _evaluate_knn(
    X_tr: np.ndarray,
    X_id_te: np.ndarray,
    X_ood_te: np.ndarray,
    contamination: float,
    k: int = 5,
) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_tr)
    # Leave-one-out kNN distance on training set (skip index 0 = self)
    tr_dists, _ = nn.kneighbors(X_tr, n_neighbors=k + 1)
    tr_scores = np.mean(tr_dists[:, 1:], axis=1)
    thresh = float(np.quantile(tr_scores, 1.0 - contamination))

    X_eval = np.vstack([X_id_te, X_ood_te])
    eval_dists, _ = nn.kneighbors(X_eval, n_neighbors=k)
    eval_scores = np.mean(eval_dists, axis=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_true = np.concatenate([np.zeros(X_id_te.shape[0]), np.ones(X_ood_te.shape[0])])
    auroc = float(roc_auc_score(y_true, eval_scores))
    auprc = float(average_precision_score(y_true, eval_scores))
    fpr = float(np.mean(eval_scores[: X_id_te.shape[0]] > thresh))
    return auroc, auprc, fpr, elapsed_ms


def run_benchmark(contamination: float = 0.05, seed: int = 42) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    for name, X_tr, y_tr, X_id_te, X_ood_te in _build_datasets(seed=seed):
        X_eval = np.vstack([X_id_te, X_ood_te])
        y_true = np.concatenate([np.zeros(X_id_te.shape[0]), np.ones(X_ood_te.shape[0])])

        estimators: list[tuple[str, Any, bool]] = [
            (
                "HopfieldOOD(full, b=8)",
                HopfieldOOD(beta=8.0, mode="full", contamination=contamination),
                False,
            ),
            (
                "HopfieldOOD(reclag, b=8)",
                HopfieldOOD(beta=8.0, mode="reclag", tau_rect=0.4, contamination=contamination),
                False,
            ),
            (
                "HopfieldOOD(she, b=8)",
                HopfieldOOD(beta=8.0, mode="she", contamination=contamination),
                True,
            ),
            (
                "HopfieldOOD(kmeans, k=32)",
                HopfieldOOD(
                    beta=8.0,
                    mode="kmeans",
                    n_prototypes=32,
                    contamination=contamination,
                    random_state=seed,
                ),
                False,
            ),
            (
                "IsolationForest",
                IsolationForest(contamination=contamination, random_state=seed),
                False,
            ),
            (
                "LocalOutlierFactor(k=20)",
                LocalOutlierFactor(
                    n_neighbors=min(20, X_tr.shape[0] - 1),
                    novelty=True,
                    contamination=contamination,
                ),
                False,
            ),
        ]

        for det_name, est, use_y in estimators:
            t0 = time.perf_counter()
            if use_y:
                est.fit(X_tr, y_tr)
            else:
                est.fit(X_tr)
            # Standard scikit-learn OutlierMixin: lower decision_function = more anomalous
            anomaly_scores = -est.decision_function(X_eval)
            id_preds = est.predict(X_id_te)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            rows.append(
                BenchmarkRow(
                    dataset=name,
                    detector=det_name,
                    auroc=float(roc_auc_score(y_true, anomaly_scores)),
                    auprc=float(average_precision_score(y_true, anomaly_scores)),
                    unseen_id_fpr=float(np.mean(id_preds == -1)),
                    elapsed_ms=elapsed_ms,
                )
            )

        auroc, auprc, fpr, elapsed_ms = _evaluate_knn(
            X_tr, X_id_te, X_ood_te, contamination=contamination, k=5
        )
        rows.append(
            BenchmarkRow(
                dataset=name,
                detector="kNN(k=5, LOO)",
                auroc=auroc,
                auprc=auprc,
                unseen_id_fpr=fpr,
                elapsed_ms=elapsed_ms,
            )
        )

    return rows


def _print_markdown_table(rows: list[BenchmarkRow], contamination: float) -> None:
    print(f"Benchmark OOD Baselines (target unseen-ID FPR = {contamination:.2%})\n")
    print("| Dataset | Detector | AUROC | AUPRC | Unseen-ID FPR | Time (ms) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| {r.dataset} | `{r.detector}` | {r.auroc:.4f} | {r.auprc:.4f} | "
            f"{r.unseen_id_fpr:.1%} | {r.elapsed_ms:.1f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HopfieldOOD against sklearn baselines.")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    rows = run_benchmark(contamination=args.contamination, seed=args.seed)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        _print_markdown_table(rows, contamination=args.contamination)


if __name__ == "__main__":
    main()
