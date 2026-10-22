# BADSFA Canonical Repository Guidelines — `scikit-hopfield`

This repository is governed by **BADSFA (Behavior-Agent-Driven Software Factory Architecture)**. Both **Google Antigravity** (`agy` / IDE) and **Claude Code** (`claude`) consume this file as the Single Source of Truth (SSOT).

## 1. Domain & Mathematical Invariants

`scikit-hopfield` provides `scikit-learn` and `PyOD` compatible estimators built on **Modern Continuous Hopfield Networks** (Ramsauer et al., ICLR 2021; SHE, ICLR 2023; Hopfield Boosting, 2024; Rectified Lagrangian, AAAI 2025; Freeman & Skapura BAM):

1. **Lyapunov Monotonicity Invariant:** Every iterative state update $\boldsymbol{\xi}_{t+1} = \mathbf{X}^T \operatorname{softmax}(\beta \mathbf{X} \boldsymbol{\xi}_t)$ MUST monotonically decrease (or keep constant at fixed points) the Hopfield energy $E_\beta(\boldsymbol{\xi})$ within floating-point tolerance ($\Delta E \le 10^{-6}$).
2. **Asymptotic $\beta$-Dial Limits:**
   - As $\beta \to \infty$, negated log-sum-exp energy $-\beta^{-1}\operatorname{lse}(\beta, \mathbf{X}\boldsymbol{\xi})$ on $L_2$-normalized patterns MUST converge to $-\max_i \cos(\mathbf{x}_i, \boldsymbol{\xi})$ (1-Nearest Neighbor).
   - Mode `she` (Simplified Hopfield Energy) MUST store class/cluster centroids and compute energy against those prototypes.
   - Mode `reclag` (Rectified Lagrangian) MUST apply a rectified threshold $\tau_{\text{rec}}$ on similarity before aggregation so low-density OOD queries collapse to the trivial origin attractor.
3. **Anti-Reward Hacking (Strict Role Separation):**
   - `@Developer` (`BADSFA_ROLE=DEVELOPER`, default) is **strictly forbidden** from modifying `specs/**` or `tests/**` (enforced via `.agents/hooks.json` `PreToolUse`).
   - Only `@QA` (`BADSFA_ROLE=QA`) or `@Architect` (`BADSFA_ROLE=ARCHITECT`) may modify `specs/**` or `tests/**`.
4. **Checkable Done (`Stop` Hook):**
   - No task may terminate while unit/contract tests fail.

## 2. Canonical Commands

- **Run Contract & Unit Suite:** `python -m pytest -v` (or `python -m unittest discover -s tests -t .`)
- **Run Multi-Domain Demo (IDS OOD + Vault/Books BAM):** `python examples/demo_dual_domain.py`
- **Verify BADSFA Gates:** Activate skill `verify-contracts`

## 3. Persistent Architectural Memory

@[Architectural Memory](.agents/memory/MEMORY.md)
