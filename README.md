# `scikit-hopfield` — Modern Continuous Hopfield Energy Estimators

`scikit-learn` and `PyOD` compatible estimators built on **Modern Continuous Hopfield Networks** (Ramsauer et al., ICLR 2021; SHE, ICLR 2023; Hopfield Boosting, 2024; Rectified Lagrangian, AAAI 2025; Freeman & Skapura BAM).

Developed and governed using the **AI Software Factory (`~/claude-software-factory`)** and **BADSFA (Behavior-Agent-Driven Software Factory Architecture)** dual-harness for **Claude Code** and **Google Antigravity**.

---

## Estimators & Mathematical Regimes

| Estimator | `scikit-learn` Protocol | Energy / Dynamical Regime | Target Applications |
| :--- | :--- | :--- | :--- |
| **`HopfieldOOD`** | `BaseEstimator`, `OutlierMixin` | Pre-iteration energy $-\beta^{-1}\operatorname{lse}(\beta, \mathbf{X}\boldsymbol{\xi})$ with continuous $\beta$-dial, `she`, `kmeans`, `boosting`, and `reclag` ($\tau_{\text{rect}}$ origin attractor). | Tabular & embedding OOD / Zero-Day detection (`FT-Transformer` IDS novelty channel, FinTech fraud, clinical abstention, LLM prompt guardrails). |
| **`HopfieldMetastableCluster`** | `BaseEstimator`, `ClusterMixin` | Iterative CCCP fixed-point updates $\boldsymbol{\xi}_{t+1} = \mathbf{X}^T\operatorname{softmax}(\beta\mathbf{X}\boldsymbol{\xi}_t)$ across $\beta$ bifurcation regimes. | Emergent hierarchical clustering without fixing $k$ a priori; multi-scale topic synthesis in knowledge graphs. |
| **`HopfieldBAM`** | `BaseEstimator`, `TransformerMixin` | Bidirectional cross-attention reverberation ($\mathbf{A} \to \mathbf{B} \to \mathbf{A}$) + OOD Knowledge Gap thresholding. | Cross-domain associative retrieval (e.g., Obsidian Vault Notes $\leftrightarrow$ Technical Book Chunks) and unindexed topic detection. |

---

## Quickstart

```python
import numpy as np
from scikit_hopfield import HopfieldOOD, HopfieldMetastableCluster, HopfieldBAM

# 1. Out-of-Distribution / Novelty Detection with Rectified Lagrangian (AAAI 2025)
det = HopfieldOOD(beta=12.0, mode="reclag", tau_rect=0.5, contamination=0.05).fit(X_train, y_train)
scores = det.decision_function(X_test)
preds = det.predict(X_test)              # 1 = inlier, -1 = outlier
attribution = det.explain(X_test, top_k=3)

# 2. Metastable Bifurcation Clustering
clusterer = HopfieldMetastableCluster(beta=8.0).fit(X_embeddings)
spectrum = clusterer.bifurcation_spectrum(X_embeddings, betas=[0.01, 2.0, 8.0, 64.0])

# 3. Cross-Domain Bidirectional Associative Memory (Notes <-> Books)
bam = HopfieldBAM(beta=15.0).fit(X_vault_notes, Y_book_chunks)
resonance = bam.resonate(query_note, top_k=3)
gap = bam.detect_knowledge_gap(query_note)
```

---

## BADSFA & Software Factory Governance

- **SSOT Rules:** [`AGENTS.md`](./AGENTS.md) & [`CLAUDE.md`](./CLAUDE.md)
- **Executable Specifications:** [`specs/BOLT-001-hopfield-ood.feature`](./specs/BOLT-001-hopfield-ood.feature), [`specs/BOLT-002-metastable-clustering.feature`](./specs/BOLT-002-metastable-clustering.feature), [`specs/BOLT-003-heteroassociative-bam.feature`](./specs/BOLT-003-heteroassociative-bam.feature)
- **Lifecycle Gates:** [`.agents/hooks.json`](./.agents/hooks.json) (`PreToolUse` Anti-Reward Hacking via `guard_specs.py` & `Stop` Checkable Done via `checkable_done.py`)
- **Software Factory Artifacts:** `.factory/artifacts/` (`prd.md`, `user_stories.json`, `architecture_diagram.mmd`, `design_decisions.json`, `adrs.json`, `test_gaps.json`, `threat_model.json`, `security_findings.json`, `code_findings.json`)
