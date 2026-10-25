"""Spearman(HopfieldOOD she / full, IsolationForest) + rank fusion. Same protocol as part_b (LOACO) and part_a (benign memory)."""
import json
import sys

import numpy as np
import torch
from common import load, nrm
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

torch.set_num_threads(20)
S = sys.argv[1]; D = load(S); Etr, _, ytr = D["train"]
LAB = ["Benign","Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Recon","Shellcode","Worms"]
def rk(x): return rankdata(x) / len(x)
def fuse(a, b, w): return w * rk(a) + (1 - w) * rk(b)
W = [0, .25, .5, .75, 1]
def hop_full(Q, M, beta, chunk=2048):
    Mt = torch.from_numpy(M); lg = float(np.log(len(M))); o = []
    for i in range(0, len(Q), chunk):
        s = torch.from_numpy(Q[i:i+chunk]) @ Mt.T
        o.append((-(torch.logsumexp(beta * s, 1) - lg) / beta).numpy())
    return np.concatenate(o).astype(np.float64)
out = {"B": {}, "A": {}}
# ---------- Part B: LOACO ----------
rows = []
for c in range(1, 10):
    r = np.random.default_rng(c)
    idx = np.concatenate([r.choice(np.where(ytr == k)[0], min(3000, (ytr == k).sum()), replace=False) for k in range(10) if k != c])
    M, ym = Etr[idx], ytr[idx]
    iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(M)
    cent = nrm(np.vstack([nrm(M)[ym == k].mean(0) for k in np.unique(ym)]))
    sc = {}
    for sp in ("val", "test"):
        E, _, y = D[sp]; ood = (y == c).astype(int)
        she = -(nrm(E) @ cent.T).max(1); isf = -iso.score_samples(E)
        sc[sp] = (she, isf, ood)
    she, isf, ood = sc["test"]
    rho = spearmanr(she, isf)[0]; rho_id = spearmanr(she[ood == 0], isf[ood == 0])[0]; rho_ood = spearmanr(she[ood == 1], isf[ood == 1])[0]
    vs, vi, vo = sc["val"]
    wv = max(W, key=lambda w: roc_auc_score(vo, fuse(vs, vi, w)))
    row = dict(cls=LAB[c], rho=rho, rho_id=rho_id, rho_ood=rho_ood, auc_she=roc_auc_score(ood, she), auc_if=roc_auc_score(ood, isf),
               auc_fuse50=roc_auc_score(ood, fuse(she, isf, .5)), w_val=wv, auc_fuse_valw=roc_auc_score(ood, fuse(she, isf, wv)))
    rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
out["B"] = rows
m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "cls"}; out["B_mean"] = m
print("B mean:", {k: round(v, 3) for k, v in m.items()})
# ---------- Part A: benign memory, attack vs benign ----------
ben = np.where(ytr == 0)[0]; r = np.random.default_rng(0); idxA = r.choice(ben, 20000, replace=False)
M = Etr[idxA]; iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(M); Mn = nrm(M); c0 = nrm(Mn.mean(0, keepdims=True))
A = {}
for sp in ("val", "test"):
    E, L, y = D[sp]; ood = (y != 0).astype(int); Q = nrm(E)
    A[sp] = dict(hop=hop_full(Q, Mn, 8), cent=-(Q @ c0.T)[:, 0], isf=-iso.score_samples(E), ood=ood)
t = A["test"]; ra = {}
for nm in ("hop", "cent"):
    ra[nm] = dict(rho=spearmanr(t[nm], t["isf"])[0], rho_id=spearmanr(t[nm][t["ood"] == 0], t["isf"][t["ood"] == 0])[0], rho_ood=spearmanr(t[nm][t["ood"] == 1], t["isf"][t["ood"] == 1])[0],
                  auc=roc_auc_score(t["ood"], t[nm]), auc_if=roc_auc_score(t["ood"], t["isf"]))
    v = A["val"]; wv = max(W, key=lambda w: roc_auc_score(v["ood"], fuse(v[nm], v["isf"], w)))
    ra[nm].update(auc_fuse50=roc_auc_score(t["ood"], fuse(t[nm], t["isf"], .5)), w_val=wv, auc_fuse_valw=roc_auc_score(t["ood"], fuse(t[nm], t["isf"], wv)))
out["A"] = ra; print("A:", json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in ra.items()}, indent=1))
json.dump(out, open("results_C.json", "w"), indent=1, default=float)
