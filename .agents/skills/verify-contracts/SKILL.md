---
name: verify-contracts
description: >-
  Executes BADSFA behavioral contract verification for scikit-hopfield (asymptotic beta limits,
  Lyapunov monotonicity, RecLag OOD separation, and BAM resonance).
---

# BADSFA Contract Verification Skill (`verify-contracts`)

1. **Check Worktree Integrity:** Run `git status --short` and verify no unauthorized edits in `specs/` or `tests/`.
2. **Execute Mathematical Contract Suite:** Run `python -m unittest discover -s tests -t . -v` (or `pytest -v`).
3. **Execute Dual-Domain Verification Demo:** Run `python examples/demo_dual_domain.py` to validate both IDS OOD Novelty Detection and Obsidian/Books BAM Resonance.
