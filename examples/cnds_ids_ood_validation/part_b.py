"""Part B: Leave-One-Attack-Class-Out with the FROZEN FT-Transformer.
CAVEAT: the encoder/classifier saw the held-out class during training; only the Hopfield/kNN memory excludes it.
=> optimistic for embedding-memory methods, pessimistic for MSP/Energy. Not a true zero-day test."""
import json
import sys
import time

import numpy as np
import torch
from common import load, logit_scores, metrics, nrm
from sklearn.metrics import roc_auc_score

S = sys.argv[1]; torch.set_num_threads(20)
BETAS = [2, 4, 8, 16, 32]; TAUS = [0.3, 0.5]; CAP = 3000
LABELS = ["Benign","Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Recon","Shellcode","Worms"]

def scores_torch(Q, M, chunk=2048):
    Mt = torch.from_numpy(M); logN = float(np.log(len(M)))
    keys = [f"full_b{b}" for b in BETAS] + [f"reclag_b{b}_t{t}" for b in BETAS for t in TAUS] + ["knn1", "knn50"]
    out = {k: [] for k in keys}
    for i in range(0, len(Q), chunk):
        Sm = torch.from_numpy(Q[i:i + chunk]) @ Mt.T
        top = torch.topk(Sm, min(50, Sm.shape[1]), dim=1).values
        out["knn1"].append((-top[:, 0]).numpy()); out["knn50"].append((-top[:, -1]).numpy())
        for b in BETAS:
            out[f"full_b{b}"].append((-(torch.logsumexp(b * Sm, 1) - logN) / b).numpy())
            for t in TAUS:
                rect = torch.clamp(Sm - t, min=0.0); act = rect.max(1).values > 0
                sc = torch.zeros(len(Sm)); 
                if act.any(): sc[act] = -(torch.logsumexp(b * rect[act], 1) - logN) / b
                out[f"reclag_b{b}_t{t}"].append(sc.numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}

# --- sanity: torch path == library HopfieldOOD ---
from scikit_hopfield import HopfieldOOD

rng = np.random.default_rng(1); M0 = rng.normal(size=(300, 32)).astype(np.float32); Q0 = rng.normal(size=(100, 32)).astype(np.float32)
fast = scores_torch(nrm(Q0), nrm(M0))
d1 = np.abs(HopfieldOOD(beta=8, mode="full").fit(M0).energy_score(Q0) - fast["full_b8"]).max()
d2 = np.abs(HopfieldOOD(beta=8, mode="reclag", tau_rect=0.3).fit(M0).energy_score(Q0) - fast["reclag_b8_t0.3"]).max()
print(f"lib check: full {d1:.1e}  reclag {d2:.1e}", flush=True); assert d1 < 1e-4 and d2 < 1e-4

D = load(S); Etr, _, ytr = D["train"]
res = {"val": {}, "test": {}}; raw = {}
for c in range(1, 10):
    t0 = time.time(); r = np.random.default_rng(c)
    idx = np.concatenate([r.choice(np.where(ytr == k)[0], min(CAP, (ytr == k).sum()), replace=False) for k in range(10) if k != c])
    M = nrm(Etr[idx]); ym = ytr[idx]
    for split in ("val", "test"):
        E, L, y = D[split]; Q = nrm(E); ood = (y == c).astype(int)
        sc = scores_torch(Q, M)
        sc["centroid_all"] = -(Q @ nrm(M.mean(0, keepdims=True)).T)[:, 0]
        cent = nrm(np.vstack([M[ym == k].mean(0) for k in np.unique(ym)]))       # SHE: one prototype per remaining class
        she = Q @ cent.T; sc["she_maxcos"] = -she.max(1)
        for k, v in logit_scores(L).items(): sc["logits:" + k] = v
        res[split][c] = {k: metrics(v, ood) for k, v in sc.items()}
        res[split][c]["_n_ood"] = int(ood.sum())
        if split == "test": raw[c] = (y, sc)
    print(f"held-out {c} {LABELS[c]:<9s} memory={len(M)} n_ood(test)={res['test'][c]['_n_ood']} {time.time()-t0:.0f}s", flush=True)

keys = [k for k in res["val"][1] if k != "_n_ood"]
mean_val = {k: np.mean([res["val"][c][k]["auroc"] for c in range(1, 10)]) for k in keys}
mean_test = {k: np.mean([res["test"][c][k]["auroc"] for c in range(1, 10)]) for k in keys}
fams = {"hopfield_full": [k for k in keys if k.startswith("full_")], "hopfield_reclag": [k for k in keys if k.startswith("reclag_")],
        "knn1": ["knn1"], "knn50": ["knn50"], "centroid": ["centroid_all"], "she_maxcos": ["she_maxcos"],
        "msp": ["logits:msp"], "energy": ["logits:energy"], "p_nonbenign": ["logits:p_nonbenign"]}
sel = {f: max(ks, key=lambda k: mean_val[k]) for f, ks in fams.items()}
out = {"selected": sel, "mean_val_auroc": {f: mean_val[k] for f, k in sel.items()}, "mean_test_auroc": {f: mean_test[k] for f, k in sel.items()},
       "per_class_test": {f: {LABELS[c]: res["test"][c][k] for c in range(1, 10)} for f, k in sel.items()},
       "n_ood_test": {LABELS[c]: res["test"][c]["_n_ood"] for c in range(1, 10)}, "all": res}
# bootstrap CI (test) for selected hopfield_full
ci = {}
k = sel["hopfield_full"]; rng = np.random.default_rng(0)
for c in range(1, 10):
    y, sc = raw[c]; ood = (y == c).astype(int); s = sc[k]; pos = np.where(ood == 1)[0]; neg = np.where(ood == 0)[0]; a = []
    for _ in range(100):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, 20000)])
        a.append(roc_auc_score(ood[i], s[i]))
    ci[LABELS[c]] = [float(x) for x in np.percentile(a, [2.5, 97.5])]
out["ci95_hopfield_full_test"] = ci
json.dump(out, open(f"{S}/results_B.json", "w"), indent=1)
print("\nSELECTED ON VAL:", sel)
print(f"\n{'method':18s}{'mean val':>9s}{'mean test':>10s}   " + " ".join(f"{LABELS[c][:7]:>7s}" for c in range(1, 10)))
for f in fams:
    print(f"{f:18s}{out['mean_val_auroc'][f]:9.4f}{out['mean_test_auroc'][f]:10.4f}   " + " ".join(f"{out['per_class_test'][f][LABELS[c]]['auroc']:7.3f}" for c in range(1, 10)))
print("n_ood(test):", out["n_ood_test"]); print("hopfield_full 95% CI:", {k: [round(x, 3) for x in v] for k, v in ci.items()})
