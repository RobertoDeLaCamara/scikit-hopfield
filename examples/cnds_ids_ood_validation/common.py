import numpy as np
import torch
from scipy.special import logsumexp as sp_lse
from sklearn.metrics import average_precision_score, roc_auc_score

from scikit_hopfield.energy import logsumexp_scaled

BETAS = [1, 2, 4, 8, 16, 32, 64]
TAUS = [0.3, 0.5, 0.7]

def load(S):
    d = {}
    for n in ("train", "val", "test"):
        z = np.load(f"{S}/{n}.npz"); d[n] = (z["emb"], z["logits"], z["y"])
    return d

def nrm(X):
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)).astype(np.float32)

def metrics(score, is_ood):
    """is_ood=1 is the positive class; higher score = more anomalous."""
    auroc = roc_auc_score(is_ood, score)
    ap = average_precision_score(is_ood, score)
    thr = np.quantile(score[is_ood == 1], 0.05)           # threshold keeping 95% TPR
    fpr95 = float(np.mean(score[is_ood == 0] >= thr))
    return dict(auroc=float(auroc), ap=float(ap), fpr95=fpr95)

def logit_scores(logits):
    p = np.exp(logits - sp_lse(logits, axis=1, keepdims=True))
    return dict(msp=1 - p.max(1), energy=-sp_lse(logits, axis=1), p_nonbenign=1 - p[:, 0])

def hopfield_scores(Qn, Mn, chunk=4096):
    """All beta / tau variants from one similarity pass. Qn, Mn L2-normalised float32.
    Uses the library's logsumexp_scaled; reclag follows energy.reclag_score (verified separately)."""
    out = {f"full_b{b}": [] for b in BETAS}
    out.update({f"reclag_b{b}_t{t}": [] for b in BETAS for t in TAUS})
    kmax = 50
    knn1, knn50 = [], []
    Mt = torch.from_numpy(Mn)
    for i in range(0, len(Qn), chunk):
        S = torch.from_numpy(Qn[i:i + chunk]) @ Mt.T
        Sn = S.numpy()
        top = torch.topk(S, min(kmax, S.shape[1]), dim=1).values.numpy()
        knn1.append(-top[:, 0]); knn50.append(-top[:, -1])
        for b in BETAS:
            out[f"full_b{b}"].append(-logsumexp_scaled(Sn, b))
            for t in TAUS:
                rect = np.maximum(0.0, Sn - t)
                act = rect.max(1) > 0
                sc = np.zeros(len(Sn), dtype=np.float64)
                if act.any():
                    sc[act] = -logsumexp_scaled(rect[act], b)
                out[f"reclag_b{b}_t{t}"].append(sc)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["knn1"] = np.concatenate(knn1); res["knn50"] = np.concatenate(knn50)
    return res
