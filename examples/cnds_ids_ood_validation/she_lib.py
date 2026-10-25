import json
import sys

import numpy as np
from common import load
from sklearn.metrics import roc_auc_score

from scikit_hopfield import HopfieldOOD

D = load(sys.argv[1]); Etr, _, ytr = D["train"]; BETAS = [2, 4, 8, 16, 32, 64, 128, 512]
LAB = ["Benign","Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Recon","Shellcode","Worms"]
R = {b: {"val": [], "test": []} for b in BETAS}; per = {b: {} for b in BETAS}
for c in range(1, 10):
    r = np.random.default_rng(c)
    idx = np.concatenate([r.choice(np.where(ytr == k)[0], min(3000, (ytr == k).sum()), replace=False) for k in range(10) if k != c])
    M, ym = Etr[idx], ytr[idx]
    for b in BETAS:
        det = HopfieldOOD(beta=b, mode="she").fit(M, ym)          # the library's own SHE mode (one prototype per remaining class)
        for sp in ("val", "test"):
            E, _, y = D[sp]; a = roc_auc_score((y == c).astype(int), det.decision_function(E)); R[b][sp].append(a)
            if sp == "test": per[b][LAB[c]] = round(a, 3)
print("lib HopfieldOOD(mode='she') LOACO mean AUROC")
for b in BETAS: print(f"beta={b:<4d} val {np.mean(R[b]['val']):.4f}  test {np.mean(R[b]['test']):.4f}")
bb = max(BETAS, key=lambda b: np.mean(R[b]["val"])); print("selected on val: beta", bb, per[bb])
json.dump({"mean": {b: [float(np.mean(R[b]["val"])), float(np.mean(R[b]["test"]))] for b in BETAS}, "sel": bb, "per_class": per[bb]}, open("results_B_she.json", "w"), indent=1)
