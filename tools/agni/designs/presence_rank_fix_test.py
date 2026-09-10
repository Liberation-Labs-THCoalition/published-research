#!/usr/bin/env python3
"""Does a rank-normalised presence recover containment? (follow-on to #99)

#99 established: presence = (r-1)/k. The k in the denominator is NOMINAL rank, but the
overlap can only occupy min(r_A, r_B) dimensions. So the candidate fix is to normalise by
the SMALLER SUBSPACE'S OWN EFFECTIVE RANK instead of by k.

    presence_rn(A,B) = sum_j sigma_j^2 / (min(r_A, r_B) - 1)

r is estimated from the singular-value spectrum at analysis time -- the SVD is already
being computed, so this costs nothing extra.

TEST: same containment setup. A strictly inside B, vary rank(A). A rank-free metric
returns ~1.0 at every row.

CONTROLS, and they must be able to fail:
  POS   identical data       -> ~1.0
  NEG   independent data     -> near chance, NOT inflated by the new denominator.
        The denominator shrinks, so a broken fix inflates the null -- that is the
        specific way this fix could fake success, and it is what NEG watches for.
"""
import numpy as np

import sys
D, K, SEED, N_ROWS = 1024, 8, 20260908, 2048
if len(sys.argv) > 1: SEED = int(sys.argv[1])   # gate finding 6: seed must be varyable
CHANCE = K / D


def spectrum(X):
    return np.linalg.svd(X, full_matrices=False, compute_uv=False)


def basis(X, k=K):
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[1:k + 1, :].T
    assert W.shape == (X.shape[1], k)
    return W


def eff_rank(X, energy=0.99, kmax=K + 1):
    """Effective rank: dims to reach `energy` of squared spectral mass, capped at kmax."""
    s = spectrum(X)[:kmax] ** 2
    c = np.cumsum(s) / np.sum(s)
    return int(np.searchsorted(c, energy) + 1)


def presence_old(WA, WB, k=K):
    s = np.linalg.svd(WA.T @ WB, compute_uv=False)
    return float(np.sum(s ** 2) / k)


def presence_rn(XA, XB, k=K):
    """Rank-normalised: divide by the smaller subspace's own effective rank, minus the
    skipped SV1. Floor the denominator at 1 so a rank-1 condition cannot divide by zero."""
    WA, WB = basis(XA, k), basis(XB, k)
    s = np.linalg.svd(WA.T @ WB, compute_uv=False)
    denom = max(min(eff_rank(XA), eff_rank(XB)) - 1, 1)
    return float(np.sum(s ** 2) / denom)


def make(rank, rng, shared=None, noise=1e-3):
    dirs = shared[:, :rank] if shared is not None else np.linalg.qr(rng.standard_normal((D, rank)))[0]
    coef = rng.standard_normal((N_ROWS, rank)) * np.linspace(3.0, 1.0, rank)
    return coef @ dirs.T + rng.standard_normal((N_ROWS, D)) * noise


def main():
    rng = np.random.default_rng(SEED)
    Xa, Xb = make(K + 1, rng), make(K + 1, rng)
    pos, neg = presence_rn(Xa, Xa), presence_rn(Xa, Xb)
    pos_ok, neg_ok = pos > 0.95, neg < 10 * CHANCE
    print(f"  CONTROLS (rank-normalised metric)")
    print(f"    [{'PASS' if pos_ok else 'FAIL'}] positive  identical   -> {pos:.4f}   (need >0.95)")
    print(f"    [{'PASS' if neg_ok else 'FAIL'}] negative  independent -> {neg:.4f}   (need <{10*CHANCE:.3f})")
    if not (pos_ok and neg_ok):
        print("\n  ** CONTROL FAILED — the fix is not evaluable. No verdict. **")
        return 1

    print(f"\n  A CONTAINED IN B. A rank-free metric returns ~1.0 on every row.\n")
    print(f"    {'rank(A)':>8}{'presence_OLD':>15}{'presence_RN':>14}{'est rank(A)':>13}")
    print("    " + "-" * 50)
    full = np.linalg.qr(rng.standard_normal((D, K + 1)))[0]
    rn = []
    for r in range(2, K + 1):
        B, A = make(K + 1, rng, shared=full), make(r, rng, shared=full)
        old = presence_old(basis(A), basis(B))
        new = presence_rn(A, B)
        rn.append(new)
        print(f"    {r:>8}{old:>15.4f}{new:>14.4f}{eff_rank(A):>13}")

    rn = np.array(rn)
    spread = rn.max() - rn.min()
    print(f"\n  RN spread across ranks: {spread:.4f}   mean {rn.mean():.4f}")
    print("\n  VERDICT")
    if rn.min() > 0.85 and spread < 0.15:
        print("    FIX WORKS. Rank-normalised presence recovers containment at every rank.")
        print("    presence is repairable at ANALYSIS time -- no rerun, no new capture.")
    else:
        print("    FIX DOES NOT WORK. Rank normalisation does not recover containment;")
        print("    presence needs replacing, not rescaling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
