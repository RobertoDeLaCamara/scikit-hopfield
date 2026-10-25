# HopfieldOOD validation on the cnds / ML-IDS FT-Transformer

Post-hoc novelty channel over the **frozen** unified FT-Transformer
(`cnds/models/unified/unified_ft_transformer.pt`, d_token=256, 76 features, 10 classes).
Run 2026-10-25. Nothing was written to cnds, ML-IDS or MLflow.

## Reproduce

```bash
PY=~/repos/ML-IDS/.venv/bin/python        # torch CPU, sklearn 1.6.1
export PYTHONPATH=~/repos/scikit-hopfield/src
OUT=/path/to/workdir
$PY extract.py $OUT      # CLS embeddings + logits (70/15/15 stratified split, seed 42)
$PY verify_lib.py        # fast torch path vs HopfieldOOD.decision_function (diff ~1e-7)
$PY part_a.py $OUT       # benign-only memory, attack vs benign
$PY part_b.py $OUT       # leave-one-attack-class-out (LOACO), frozen encoder
$PY she_lib.py $OUT      # HopfieldOOD(mode="she") LOACO via the library
$PY part_c.py $OUT       # Spearman vs IsolationForest + rank fusion
```

`extract.py` has absolute paths to the cnds / ML-IDS checkouts. It reproduces the published
test macro-F1 (0.6194 vs 0.6197). Raw outputs are in `results/`.

## Results (test split)

**Part A: memory = benign train, attack vs benign (20% attacks).** Saturated and tied.

| Method | AUROC |
|---|---|
| `1 - P(Benign)` (supervised) | 0.9959 |
| `HopfieldOOD` full, mem 20k, beta=8 | 0.9947 |
| `HopfieldOOD` kmeans 256, beta=4 | 0.9947 |
| IsolationForest on embeddings | 0.9934 |
| Benign centroid | 0.9925 |
| MSP / Energy | 0.981 |

**Part B: LOACO (memory <= 3000/class from the other classes; beta/tau picked on val).**

| Method | Mean AUROC |
|---|---|
| `HopfieldOOD(mode="she")`, beta >= 32 | **0.960** |
| `1 - P(Benign)` | 0.914 |
| Energy / MSP | 0.899 / 0.896 |
| kNN (k=1 / 50) | 0.859 / 0.858 |
| `HopfieldOOD` full / reclag (beta=16) | 0.801 |
| Global centroid | 0.607 |

**Part C: diversity vs IsolationForest.**

| Setting | Spearman rho | Fusion gain |
|---|---|---|
| A, `HopfieldOOD` full | 0.958 | +0.0000 (redundant) |
| B, `she` (mean of 9 classes) | 0.681 | none: val picks w=1; 50/50 gives 0.881 vs 0.960 |

## Caveats

- Part B is **not** a zero-day test. The encoder saw the held-out class in training, so
  embedding-memory methods (especially `she`) are optimistic. An honest LOACO needs the
  FT-Transformer retrained without each class (9 trainings, GPU).
- Single split and seed. Worms (n=37), Backdoor (n=68) and Analysis (n=58) have wide CIs.
- UNSW-NB15 test data, not LAN traffic: calibrate thresholds on real traffic before acting on it.
- `mode="full"` over the whole benign memory (~250k) took ~39 min per split; use `she` or `kmeans`.

## Recommendation

Keep `1 - P(Benign)` for detection. Add `HopfieldOOD(mode="she")` as a secondary novelty signal
in shadow mode. Do not blend IsolationForest into it.
