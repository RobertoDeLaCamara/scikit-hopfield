# Product Requirements Document (PRD): `scikit-hopfield`

- **Version:** 1.0.0
- **Owner:** `@TPM` & `@Requirements`
- **Architecture Reference:** `ADR-0001` (`scikit_hopfield`)

---

## 1. Executive Summary & Problem Statement

Modern Hopfield Networks (Ramsauer et al., 2021) provide an exact mathematical bridge between associative memory energy landscapes, attention mechanisms, and density/prototype estimation controlled by an inverse temperature parameter $\beta$. However, practical Python data science and security ecosystems lack a unified, numerically stable, `scikit-learn` / `PyOD`-compliant library:
- **`pytorch-ood`** restricts Hopfield energy to the degenerate $\beta \to 0$ Simplified Hopfield Energy (`SHE`) limit coupled to PyTorch vision `DataLoader` pipelines.
- **`PyOD`** lacks any Modern Hopfield `OutlierMixin` estimator supporting the continuous $\beta$-dial, auxiliary outlier exposure (**Hopfield Boosting**, Auer et al., 2024), or **Rectified Lagrangian (`RecLag`)** trivial-attractor clamping (AAAI 2025).
- **Knowledge Management & Clustering** ecosystems lack a `ClusterMixin` for **Metastable Bifurcation Clustering** across $\beta$ temperature regimes and a continuous **Bidirectional Associative Memory (`HopfieldBAM`)** for cross-domain concept resonance and unindexed knowledge gap detection.

`scikit-hopfield` delivers a zero-bloat, NumPy-first (`scikit-learn` compliant) package powered by a shared, numerically stable `HopfieldEnergyCore`.

---

## 2. Target Use Cases

### Use Case 1: IDS FT-Transformer Novelty Channel (`cnds` / `ML-IDS`)
- **Context:** Real-time Network Intrusion Detection Systems (`cnds` and `ML-IDS`) extract tabular flow representations (e.g., CICFlowMeter features embedded via FT-Transformer or deep encoders). Supervised classifiers excel on known attacks but fail silently on zero-day flows or novel C2 TLS/JA3 profiles.
- **Role of `scikit-hopfield`:** `HopfieldOOD` acts as a plug-and-play novelty channel over L2-normalized latent flow embeddings:
  - **Continuous $\beta$-Dial & `SHE`:** Tunes sensitivity smoothly between compact class prototypes (`mode="she"`, $\mathcal{O}(K \cdot d)$ latency for high-throughput line-rate scoring) and fine-grained 1-NN boundary manifolds ($\beta \to \infty$, `mode="full"`).
  - **Rectified Lagrangian (`mode="reclag"`):** Clamps flow embeddings whose maximum similarity to benign/known patterns falls below $\tau_{\text{rect}}$ directly to the trivial zero-activation origin attractor, producing bounded, high-confidence novelty alerts without false attraction to distant memories.
  - **Hopfield Boosting (`X_outliers`):** Incorporates historical honeypot/anomalous flows as repulsive memories to sharpen the decision boundary between benign jitter and adversarial probes.
  - **Explainable Attribution (`explain()`):** Returns top-$k$ stored network flow prototypes and softmax attention weights $\text{softmax}(\beta X \xi)$ for immediate SOC/Guardian root-cause triage.

### Use Case 2: Obsidian Vault $\leftrightarrow$ Technical Books Associative Resonance (`NeuroStack` / `pdf-books-mcp`)
- **Context:** Personal Knowledge Management (PKM) systems index thousands of personal Markdown notes (Domain A: Obsidian Vault) alongside chunked technical literature (Domain B: `pdf-books-mcp`). Flat top-$k$ cosine retrieval fails to capture multi-step conceptual synthesis, hierarchical topic coalescence, or unindexed blind spots.
- **Role of `scikit-hopfield`:**
  - **Metastable Concept Clustering (`HopfieldMetastableCluster`):** Evolves note embeddings along the continuous Hopfield fixed-point update $\xi_{t+1} = X^\top \text{softmax}(\beta X \xi_t)$. Scanning $\beta$ (`bifurcation_spectrum`) reveals hierarchical topic structure—from a single global theme ($\beta \to 0$) to coarse domains (intermediate $\beta$) to individual atomic notes ($\beta \to \infty$).
  - **Cross-Domain Resonance (`HopfieldBAM`):** Bridges Domain A (Vault Notes) and Domain B (Book Chunks) via continuous bidirectional reverberation ($A \to B \to A$). Noisy or partial note queries settle into resonant cross-domain attractors, surfacing foundational book chapters that complete the note's theoretical context.
  - **Knowledge Gap Detection (`detect_knowledge_gap()`):** Evaluates free-energy thresholds across paired domains to deterministically flag research queries that lack coverage in the existing vault or book library (`is_gap = True`).

---

## 3. MoSCoW Prioritization Table

| Priority | ID | Feature / Requirement | Component | Target Bolt |
| :--- | :--- | :--- | :--- | :--- |
| **Must Have** | `REQ-CORE-01` | Numerically stable `HopfieldEnergyCore` using max-shifted `logsumexp` ($\beta \in [10^{-4}, 500]$ without `NaN`/`Inf`), Rectified Lagrangian energy, softmax attention weights, and fixed-point step. | `HopfieldEnergyCore` | `BOLT-001..003` |
| **Must Have** | `REQ-OOD-01` | `HopfieldOOD` (`BaseEstimator`, `OutlierMixin`) supporting `mode in {"full", "she", "reclag"}`, continuous `beta`, `tau_rect`, contamination threshold calibration (`threshold_`, `decision_scores_`), `decision_function(X)`, and `predict(X)`. | `HopfieldOOD` | `BOLT-001` |
| **Must Have** | `REQ-OOD-02` | Hopfield Boosting via `fit(X, y=None, X_outliers=None)` penalizing proximity to auxiliary outlier memories $X_{\text{ood}}$ while rewarding proximity to in-distribution memories $X_{\text{id}}$. | `HopfieldOOD` | `BOLT-001` |
| **Must Have** | `REQ-OOD-03` | Explainable memory attribution via `explain(X, top_k=3)` returning top-$k$ memory indices and normalized attention weights summing to $1.0$. | `HopfieldOOD` | `BOLT-001` |
| **Must Have** | `REQ-CLU-01` | `HopfieldMetastableCluster` (`BaseEstimator`, `ClusterMixin`) with guaranteed Lyapunov monotonic energy descent ($\Delta E \le 10^{-6}$ per step) and attractor merging within `merge_tol`. | `HopfieldMetastableCluster` | `BOLT-002` |
| **Must Have** | `REQ-CLU-02` | Inverse temperature bifurcation sweep `bifurcation_spectrum(X, betas)` demonstrating transition from single global mean state ($\beta \to 0$) to distinct semantic cluster attractors (intermediate $\beta$). | `HopfieldMetastableCluster` | `BOLT-002` |
| **Must Have** | `REQ-BAM-01` | `HopfieldBAM` (`BaseEstimator`) supporting paired/cross-linked domain fitting `fit(X_domain_a, Y_domain_b)` and bidirectional reverberation `resonate(query_a, top_k)` converging within `max_iter`. | `HopfieldBAM` | `BOLT-003` |
| **Must Have** | `REQ-BAM-02` | Unindexed topic detection `detect_knowledge_gap(query_a)` flagging `is_gap = True` when query energy exceeds the calibrated in-distribution associative energy threshold. | `HopfieldBAM` | `BOLT-003` |
| **Should Have** | `REQ-API-01` | Full `scikit-learn` estimator ergonomics (`get_params`, `set_params`, input validation, L2-normalization handling) and zero heavy runtime dependencies beyond `numpy`, `scipy`, and `scikit-learn`. | All | `BOLT-001..003` |
| **Could Have** | `REQ-ACC-01` | Optional PyTorch tensor dispatch backend for GPU-accelerated batch scoring on large memory banks ($N > 10^5$). | `HopfieldEnergyCore` | Future |
| **Won't Have (v1)** | `REQ-NON-01` | End-to-end training of backbone feature extractors (e.g., training FT-Transformer or text embedding models inside `scikit-hopfield`). | Out of Scope | N/A |

---

## 4. Non-Functional & Mathematical Requirements

1. **Numerical Stability:** For any L2-normalized state $\xi \in \mathbb{R}^d$ and memory matrix $X \in \mathbb{R}^{N \times d}$, energy $E_\beta(\xi) = -\beta^{-1} \log \sum_{i=1}^N \exp(\beta x_i^\top \xi) + \frac{1}{2}\|\xi\|_2^2$ must use `scipy.special.logsumexp` or max-shift trick so $\beta = 200.0$ converges smoothly to $-\max_i (x_i^\top \xi) + \frac{1}{2}\|\xi\|_2^2$ within $0.02$ tolerance.
2. **Lyapunov Monotonicity:** Every iteration of `hopfield_step` must satisfy $E_\beta(\xi_{t+1}) - E_\beta(\xi_t) \le 10^{-6}$.
3. **Deterministic Verification:** All three bolts (`BOLT-001`, `BOLT-002`, `BOLT-003`) must pass 100% of Gherkin BDD scenarios in `specs/*.feature`.
