#!/usr/bin/env python3
"""#99 / kill #59 — IS `presence` A RANK-BOUNDED ESTIMATOR?

Answered synthetically. No model, no GPU: the question is about the estimator's
mathematics, not about any particular data.

THE ESTIMATOR, implemented exactly as identity-geometry defines it (main.tex:105-130):

    For each condition: concatenate V-projection outputs across probes, take the
    rank-k SVD SKIPPING the first singular vector (it encodes input length, r=0.79),
    giving W in R^{d x k}. Then

        presence(A, B) = (1/k) * sum_{j=1..k} sigma_j^2 ( W_A^T W_B )

    Paper's parameters: k = 8, d = 1024, chance = k/d = 0.0078.

THE WORRY. Both bases are forced to rank k=8. If a condition's signal actually
occupies r < 8 dimensions, the remaining 8-r basis vectors are noise-determined and
arbitrary. Two such bases overlap at chance in those directions. So presence should
fall toward r/k EVEN WHEN one subspace is entirely contained in the other -- i.e. the
metric would report "these identities differ" when the truth is "one condition has
lower effective rank."

Identity conditions differ in exactly that way: a bare-assistant prompt and a rich
persona prompt do not produce the same amount of structure.

THE TEST. Construct A strictly INSIDE B, vary A's effective rank, hold the shared
structure at 100%. If presence tracks r/k rather than staying near 1.0, the estimator
is rank-bounded and every cross-condition comparison in the paper inherits it.

CONTROLS, both directions:
  POSITIVE  identical data -> presence must be ~1.0. If not, the implementation is wrong
            and no other row means anything.
  NEGATIVE  independent random data -> presence must be ~k/d = 0.0078. If it is not near
            chance, the null is wrong and every "separation" is measured against a bad floor.
A run where either control fails is UNINTERPRETABLE and says so rather than reporting rows.
"""
import numpy as np

import sys
D, K, SEED = 1024, 8, 20260907
if len(sys.argv) > 1: SEED = int(sys.argv[1])   # gate finding 6: seed must be varyable
N_ROWS = 2048   # > D, so the row space can span R^D
CHANCE = K / D


def basis(X, k=K):
    """Rank-k basis in FEATURE space, skipping SV1 — the paper's construction.

    W must be in R^{d x k}. numpy's U spans the ROW space (n_rows-dim), NOT the
    feature space. Taking U silently produced chance overlap on every row --
    including where two subspaces were IDENTICAL -- and the POSITIVE CONTROL DID
    NOT CATCH IT, because identical data yields identical U either way. A control
    that passes under both the right and the wrong construction is not a control.
    Hence the shape assertion, which can actually fail."""
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[1:k + 1, :].T
    assert W.shape == (X.shape[1], k), f"basis must be (d,k)=({X.shape[1]},{k}), got {W.shape}"
    return W


def presence(WA, WB, k=K):
    """(1/k) * sum of squared singular values of W_A^T W_B."""
    s = np.linalg.svd(WA.T @ WB, compute_uv=False)
    return float(np.sum(s ** 2) / k)


def make(rank, n_rows, rng, shared_dirs=None, noise=1e-3):
    # n_rows MUST exceed d, else the row space spans only n_rows dims and
    # chance alignment is k/n_rows rather than k/d. The negative control
    # caught exactly this at n_rows=64 (0.102 ~= 8/64), which is why it exists.
    """A matrix whose row space has the given effective rank.
    If shared_dirs is given, the signal is drawn ONLY from those directions."""
    dirs = shared_dirs[:, :rank] if shared_dirs is not None else \
        np.linalg.qr(rng.standard_normal((D, rank)))[0]
    coef = rng.standard_normal((n_rows, rank)) * np.linspace(3.0, 1.0, rank)
    return coef @ dirs.T + rng.standard_normal((n_rows, D)) * noise


def main():
    rng = np.random.default_rng(SEED)
    print(f"  presence: k={K}, d={D}, chance=k/d={CHANCE:.4f}\n")

    # ---------------------------------------------------------------- controls
    Xa = make(K + 1, N_ROWS, rng)
    pos = presence(basis(Xa), basis(Xa))
    Xb = make(K + 1, N_ROWS, rng)
    neg = presence(basis(Xa), basis(Xb))
    pos_ok = pos > 0.99
    neg_ok = neg < 10 * CHANCE

    print("  CONTROLS")
    print(f"    [{'PASS' if pos_ok else 'FAIL'}] positive: identical data -> {pos:.4f}  (need >0.99)")
    print(f"    [{'PASS' if neg_ok else 'FAIL'}] negative: independent data -> {neg:.4f}  (need <{10*CHANCE:.3f})")
    if not (pos_ok and neg_ok):
        print("\n  ** A CONTROL FAILED. Every row below is uninterpretable. Stopping. **")
        return 1

    # ------------------------------------------- the test: A strictly inside B
    print(f"\n  A IS ENTIRELY CONTAINED IN B. Shared structure is 100% at every row.")
    print(f"  If presence is rank-free it stays ~1.0. If rank-bounded it tracks r/k.\n")
    print(f"    {'rank(A)':>8}{'presence(A,B)':>16}{'r/k':>10}{'|diff|':>10}")
    print("    " + "-" * 44)
    full = np.linalg.qr(rng.standard_normal((D, K + 1)))[0]
    rows = []
    for r in range(1, K + 1):
        # B spans all K+1 shared directions; A spans only the first r of them
        B = make(K + 1, N_ROWS, rng, shared_dirs=full)
        A = make(r,     N_ROWS, rng, shared_dirs=full)
        p = presence(basis(A), basis(B))
        rows.append((r, p, r / K))
        print(f"    {r:>8}{p:>16.4f}{r/K:>10.3f}{abs(p - r/K):>10.3f}")

    # ------------------------------------------------------------------ verdict
    ranks = np.array([r for r, _, _ in rows], float)
    pres = np.array([p for _, p, _ in rows], float)
    corr = float(np.corrcoef(ranks, pres)[0, 1])
    lowest, highest = pres[0], pres[-1]

    print(f"\n  corr(rank(A), presence) = {corr:+.3f}")
    print(f"  presence at rank 1 = {lowest:.4f}   at rank {K} = {highest:.4f}")

    # A run where even the FULL-RANK row sits at chance has not tested anything.
    # Without this branch the script reported "NOT rank-bounded" off a broken setup.
    if highest < 0.5:
        print(f"\n  ** SETUP FAILED: presence at rank k is {highest:.4f}, near chance.")
        print( "     A is contained in B at every row, so this row MUST be high.")
        print( "     This run measured the construction, not the estimator. No verdict. **")
        return 1

    bounded = corr > 0.9 and lowest < 0.5
    print("\n  VERDICT")
    if bounded:
        print("    RANK-BOUNDED — CONFIRMED.")
        print("    A is 100% inside B at every row, yet presence falls with rank(A).")
        print("    The metric conflates 'lower effective rank' with 'different identity'.")
        print("    Any cross-condition presence comparison where the conditions differ in")
        print("    effective rank is confounded, and prompt length/richness drives rank.")
    else:
        print("    NOT rank-bounded by this test. Presence stayed high while rank varied,")
        print("    so containment is recovered regardless of rank. Kill #59 does not")
        print("    reproduce synthetically; the concern needs a different formulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
