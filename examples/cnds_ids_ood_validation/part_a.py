"""Part A: one-class novelty detection. Memory = BENIGN train embeddings only.
Positive (OOD) = any attack. Hyper-parameters chosen on VAL, reported on TEST."""
import json
import sys
import time

import numpy as np
import torch
from common import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest

S = sys.argv[1]; torch.set_num_threads(20)
D = load(S); LABELS = ["Benign","Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Recon","Shellcode","Worms"]
Etr, Ltr, ytr = D["train"]
ben = nrm(Etr[ytr == 0]); print("benign train", ben.shape, flush=True)
rng = np.random.default_rng(0)
mem = {"mem20k": ben[rng.choice(len(ben), 20000, replace=False)],
       "km256": nrm(MiniBatchKMeans(256, n_init=1, random_state=0, batch_size=8192).fit(ben).cluster_centers_),
       "centroid1": nrm(ben.mean(0, keepdims=True)),
       "benign_all": ben}
iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=20).fit(Etr[ytr == 0][rng.choice(len(ben), 50000, replace=False)])
res = {}
for split in ("val", "test"):
    E, L, y = D[split]; Q = nrm(E); is_ood = (y != 0).astype(int)
    sc = {"logits:" + k: v for k, v in logit_scores(L).items()}
    sc["iforest(emb)"] = -iso.score_samples(E)
    for mname, M in mem.items():
        if mname == "benign_all":   # full memory: only the cheap kNN scores (Hopfield full = 250k x beta sweep is wasteful)
            t0 = time.time(); h = hopfield_scores(Q, M, chunk=1024)
            sc.update({f"{mname}:{k}": v for k, v in h.items() if k.startswith("knn") or k in ("full_b8","full_b32","full_b64")})
            print(split, mname, f"{time.time()-t0:.0f}s", flush=True)
        else:
            h = hopfield_scores(Q, M); sc.update({f"{mname}:{k}": v for k, v in h.items()})
    res[split] = {k: metrics(v, is_ood) for k, v in sc.items()}
    res[split + "_scores_keys"] = list(sc)
    if split == "test":
        res["per_class_test"] = {}
        for k in ("logits:msp", "logits:energy", "logits:p_nonbenign", "iforest(emb)"):
            res["per_class_test"][k] = {LABELS[c]: metrics(sc[k][(y == 0) | (y == c)], (y[(y == 0) | (y == c)] != 0).astype(int))["auroc"] for c in range(1, 10)}
        np.savez(f"{S}/scores_test_A.npz", y=y, **{k.replace(":", "__"): v.astype(np.float32) for k, v in sc.items()})
        np.save(f"{S}/_sc_keys.npy", np.array(list(sc)))
        TESTSC = sc
    else:
        VALSC = sc
# model selection on VAL (AUROC) among Hopfield / kNN families
fam = {}
for k, m in res["val"].items():
    f = k.split(":")[0] if ":" in k else k
    fam.setdefault(f, []).append((m["auroc"], k))
best = {}
for f, lst in fam.items():
    a, k = max(lst); best[k] = dict(val=res["val"][k], test=res["test"][k])
    if k.split(":")[0] in mem:
        E, L, y = D["test"]; is_ood = (y != 0).astype(int)
        res.setdefault("per_class_test", {})[k] = {LABELS[c]: metrics(TESTSC[k][(y == 0) | (y == c)], (y[(y == 0) | (y == c)] != 0).astype(int))["auroc"] for c in range(1, 10)}
res["best_by_val"] = best
res["n"] = {"val_ood_frac": float((D["val"][2] != 0).mean()), "test_ood_frac": float((D["test"][2] != 0).mean())}
json.dump(res, open(f"{S}/results_A.json", "w"), indent=1)
print(f"{'selected config (val-chosen)':44s} {'val AUROC':>9} | {'TEST AUROC':>10} {'AP':>6} {'FPR95':>6}")
for k, v in sorted(best.items(), key=lambda kv: -kv[1]["test"]["auroc"]):
    print(f"{k:44s} {v['val']['auroc']:9.4f} | {v['test']['auroc']:10.4f} {v['test']['ap']:6.4f} {v['test']['fpr95']:6.3f}")
