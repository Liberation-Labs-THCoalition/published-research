#!/usr/bin/env python3
"""Gate findings 6 + 7 on the presence rank audit.

F6: n=1 seed, no dispersion.  -> 20 seeds, mean +- SD (population SD, ddof=0, stated).
F7: my construction gave A and B the SAME decaying coefficient scale, so their singular
    directions came out in the same ORDER, so the SV1 discard removed a CORRESPONDING
    direction from each. Real conditions differing in effective rank are exactly where
    that need not hold -- "the synthetic setup assumes away the interaction it is testing."
    -> PERMUTED arm: same shared directions, A's coefficient scaling REVERSED vs B's.

Dimensions reduced from d=1024/n=2048 to d=256/n=512: the claim is about the RATIO
(r-1)/k and chance k/d, both preserved in form. chance here = 8/256 = 0.031.
"""
import numpy as np, statistics as st

D, K, N_ROWS = 256, 8, 512
SEEDS = [20260907, 20260908, 777, 31415, 42, 999, 12345, 8675309, 2718, 1618,
         101, 202, 303, 404, 505, 606, 707, 808, 909, 111]

def basis(X, k=K):
    _, _, Vt = np.linalg.svd(X, full_matrices=False); return Vt[1:k+1, :].T
def eff_rank(X, energy=0.99, kmax=K+1):
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)[:kmax]**2
    return int(np.searchsorted(np.cumsum(s)/np.sum(s), energy) + 1)
def p_old(A, B):
    s = np.linalg.svd(basis(A).T @ basis(B), compute_uv=False); return float(np.sum(s**2)/K)
def p_rn(A, B):
    s = np.linalg.svd(basis(A).T @ basis(B), compute_uv=False)
    return float(np.sum(s**2) / max(min(eff_rank(A), eff_rank(B)) - 1, 1))
def make(rank, rng, shared, scale):
    coef = rng.standard_normal((N_ROWS, rank)) * scale[:rank]
    return coef @ shared[:, :rank].T + rng.standard_normal((N_ROWS, D)) * 1e-3

print(f"  d={D}, k={K}, n_rows={N_ROWS}, chance=k/d={K/D:.3f}, {len(SEEDS)} seeds, SD is ddof=0\n")
print(f"    {'rank':>5} | {'OLD aligned':>16} {'RN aligned':>16} | {'OLD perm':>16} {'RN perm':>16}")
print("    " + "-"*78)
bad = []
for r in range(2, K + 1):
    c = {k: [] for k in ("oa", "ra", "op", "rp")}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        full = np.linalg.qr(rng.standard_normal((D, K + 1)))[0]
        desc = np.linspace(3.0, 1.0, K + 1)
        B  = make(K + 1, rng, full, desc)
        Aa = make(r, rng, full, desc)
        Ap = make(r, rng, full, desc[::-1].copy())
        c["oa"].append(p_old(Aa, B)); c["ra"].append(p_rn(Aa, B))
        c["op"].append(p_old(Ap, B)); c["rp"].append(p_rn(Ap, B))
    f = lambda k: f"{st.mean(c[k]):.3f}+-{st.pstdev(c[k]):.3f}"
    print(f"    {r:>5} | {f('oa'):>16} {f('ra'):>16} | {f('op'):>16} {f('rp'):>16}")
    if st.mean(c["rp"]) < 0.85:
        bad.append((r, round(st.mean(c["rp"]), 3)))

print("\n  VERDICT — the fix under PERMUTED spectral ordering:")
if bad:
    print(f"    FIX DEGRADES. Rows below 0.85: {bad}")
    print("    The fix relied on A and B sharing spectral order. Gate finding 7 STANDS.")
else:
    print("    FIX HOLDS under different spectral orderings. Finding 7 raised a real gap")
    print("    in the evidence; the arm closes it.")
