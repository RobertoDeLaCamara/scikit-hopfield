"""Extract CLS embeddings + logits from the frozen unified FT-Transformer for the
reproduced 70/15/15 stratified split (seed 42). Read-only on cnds/ML-IDS."""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/roberto/repos/cnds")
from src.models.ft_transformer import build_from_checkpoint, load_checkpoint

OUT = sys.argv[1]
torch.set_num_threads(20)
bundle = load_checkpoint("/home/roberto/repos/cnds/models/unified/unified_ft_transformer.pt")
model = build_from_checkpoint(bundle).eval()
scaler = joblib.load("/home/roberto/repos/cnds/models/unified/unified_scaler.pkl")
meta = json.load(open("/home/roberto/repos/ML-IDS/models/unified/unified_metadata.json"))
d = "/home/roberto/repos/ML-IDS/data/CIC-IDS2017/"
X = pd.read_csv(d + "Data.csv").values.astype(np.float32)
y = pd.read_csv(d + "Label.csv").values.ravel().astype(np.int64)
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.15/0.85, stratify=ytv, random_state=42)
cnt = {str(c): int((ytr == c).sum()) for c in range(10)}
print("split sizes", len(ytr), len(yva), len(yte), "| train counts match metadata:", cnt == meta["class_counts_train"])
@torch.inference_mode()
def run(Xr):
    Xr = np.nan_to_num(Xr, nan=0.0, posinf=1e9, neginf=-1e9)
    Xs = scaler.transform(Xr).astype(np.float32)
    embs, logs = [], []
    for i in range(0, len(Xs), 4096):
        t = torch.from_numpy(Xs[i:i+4096])
        tok = t.unsqueeze(-1) * model.tokenizer_w + model.tokenizer_b
        z = torch.cat([model.cls.expand(t.size(0), -1, -1), tok], dim=1)
        e = model.norm(model.encoder(z)[:, 0])
        embs.append(e.numpy()); logs.append(model.head(e).numpy())
    return np.vstack(embs), np.vstack(logs)
for name, Xr, yr in [("test", Xte, yte), ("val", Xva, yva), ("train", Xtr, ytr)]:
    t0 = time.time(); e, l = run(Xr)
    pred = l.argmax(1)
    from sklearn.metrics import f1_score
    print(name, e.shape, f"{time.time()-t0:.0f}s", "f1_macro=%.4f" % f1_score(yr, pred, average="macro"), flush=True)
    np.savez(f"{OUT}/{name}.npz", emb=e.astype(np.float32), logits=l.astype(np.float32), y=yr)
