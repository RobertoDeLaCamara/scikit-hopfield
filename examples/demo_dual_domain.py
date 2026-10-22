#!/usr/bin/env python3
"""
Dual-Domain Verification Demo for `scikit-hopfield`:
1. Domain 1 (Cybersecurity / Tabular IDS): `HopfieldOOD` with beta-dial, SHE, and RecLag
   on simulated FT-Transformer CLS embeddings (`d=256`).
2. Domain 2 (Knowledge Graph / Obsidian <-> Books): `HopfieldMetastableCluster` beta-bifurcation
   and `HopfieldBAM` cross-domain resonance + Knowledge Gap detection.
"""

from __future__ import annotations

import numpy as np

from scikit_hopfield import HopfieldBAM, HopfieldMetastableCluster, HopfieldOOD


def run_ids_novelty_demo() -> None:
    print("=== 1. IDS Novelty Channel Demo (FT-Transformer CLS d=256) ===")
    rng = np.random.default_rng(2026)
    # Simulate 3 known network traffic classes in d=256 embedding space
    centers = rng.normal(size=(3, 256))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X_id = np.vstack([centers[c] + rng.normal(scale=0.04, size=(80, 256)) for c in range(3)])
    y_id = np.repeat([0, 1, 2], 80)

    # Zero-day attack flows in orthogonal subspace
    X_zeroday = rng.normal(size=(20, 256))

    for mode in ("she", "full", "reclag"):
        det = HopfieldOOD(beta=12.0, mode=mode, tau_rect=0.5, contamination=0.05).fit(X_id, y_id)
        id_mean = float(np.mean(det.decision_function(X_id)))
        ood_mean = float(np.mean(det.decision_function(X_zeroday)))
        tpr = float(np.mean(det.predict(X_zeroday) == -1)) * 100.0
        print(
            f"  Mode={mode:6s} | ID mean energy={id_mean:+.4f} | "
            f"Zero-Day mean energy={ood_mean:+.4f} | Detection Rate={tpr:.1f}%"
        )


def run_vault_bam_demo() -> None:
    print("\n=== 2. Obsidian Vault <-> Books BAM & Metastable Bifurcation Demo ===")
    rng = np.random.default_rng(1998)
    # 4 thematic clusters across 40 notes
    note_anchors = np.eye(4, 32)
    X_notes = np.vstack([note_anchors[i] + rng.normal(scale=0.03, size=(10, 32)) for i in range(4)])

    clusterer = HopfieldMetastableCluster(merge_tol=1e-2)
    spectrum = clusterer.bifurcation_spectrum(X_notes, betas=[0.01, 2.0, 10.0, 80.0])
    print(f"  Metastable Bifurcation Spectrum (beta -> n_clusters): {spectrum}")

    # Paired Book Chunks in Domain B
    Y_books = np.roll(note_anchors, shift=4, axis=1)
    bam = HopfieldBAM(beta=16.0).fit(note_anchors, Y_books)
    query_known = note_anchors[1:2] + rng.normal(scale=0.02, size=(1, 32))
    query_gap = np.zeros((1, 32))
    query_gap[0, 25] = 1.0

    res = bam.resonate(query_known, top_k=2)
    gap_check = bam.detect_knowledge_gap(np.vstack([query_known, query_gap]))
    print(
        f"  BAM Resonance Top-1 Book Slot: {int(res['top_b_indices'][0, 0])} "
        f"(weight={float(res['top_b_weights'][0, 0]):.4f}, converged={bool(res['converged'][0])})"
    )
    print(
        f"  Knowledge Gap Check -> Known Topic is_gap={bool(gap_check['is_gap'][0])}, "
        f"Unindexed Topic is_gap={bool(gap_check['is_gap'][1])}"
    )


if __name__ == "__main__":
    run_ids_novelty_demo()
    run_vault_bam_demo()
