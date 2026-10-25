# Check the fast path equals the library's own HopfieldOOD.decision_function (full + reclag).
import numpy as np
from common import *

from scikit_hopfield import HopfieldOOD

rng = np.random.default_rng(0)
M = rng.normal(size=(300, 32)).astype(np.float32); Q = rng.normal(size=(200, 32)).astype(np.float32)
Mn, Qn = nrm(M), nrm(Q)
fast = hopfield_scores(Qn, Mn)
for b in (2, 16):
    lib = HopfieldOOD(beta=b, mode="full").fit(M).decision_function(Q)
    print("full b", b, "max|diff|", np.abs(lib - fast[f"full_b{b}"]).max())
for t in (0.3, 0.5):
    lib = HopfieldOOD(beta=8, mode="reclag", tau_rect=t).fit(M).decision_function(Q)
    print("reclag t", t, "max|diff|", np.abs(lib - fast[f"reclag_b8_t{t}"]).max())
