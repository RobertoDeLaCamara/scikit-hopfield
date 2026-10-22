# Architectural Memory — `scikit-hopfield`

- **2026-10-22 (ADR-0001):** Unified Option 1 (`HopfieldOOD` for tabular/IDS novelty detection with $\beta$-dial, SHE, Hopfield Boosting, and RecLag) and Option 2 (`HopfieldMetastableCluster` + `HopfieldBAM` for Obsidian/Books associative clustering and cross-domain resonance) under a single `scikit-learn`-compatible package governed by BADSFA.
- **Numerical Stability:** Always compute $\beta^{-1}\log\sum_i \exp(\beta z_i)$ using the shifted log-sum-exp identity $m + \beta^{-1}\log\sum_i \exp(\beta z_i - m)$ where $m = \max_i(\beta z_i)$ to prevent overflow at large $\beta$ ($\beta \ge 64$).
