r"""Mine 4: is the pre-registered success criterion actually reachable at the planned n?

THE CRITERION, verbatim from the prospectus (2026-06):
    "Within-condition encoding AUROC < 0.55 for deception prediction"
    "Encoding-generation classifier correlation r < 0.20"

BOTH ARE NULL-ACCEPTANCE CLAIMS. Establishing "AUROC is below 0.55" is an EQUIVALENCE
claim, not a significance claim, and it needs the CONFIDENCE INTERVAL to sit below the
bound — not the point estimate. I corrected exactly this error in the mine5 paper on
2026-09-01: a 90% CI reported in the wrong units, with the equivalence conclusion resting
on a point estimate. Two days later I am about to spend Starship time on a design that
makes the same mistake prospectively.

Planned n: 100+ unique questions x 4 conditions. Within-condition analysis, so the
relevant n is ~100 per condition, split by deception label at an expected ~70/30 rate in
shutdown_only.

This computes the AUROC confidence interval by the Hanley-McNeil standard error, which is
the standard analytic approximation and is adequate for a power question at the design
stage. If the CI at the planned n does not fit below 0.55, the criterion is unreachable
and the experiment cannot succeed on its own terms no matter what the data say.
"""
import math


def hanley_mcneil_se(auc, n_pos, n_neg):
    """SE of AUROC (Hanley & McNeil 1982). Q1, Q2 are the standard approximations."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    num = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc))
    return math.sqrt(num / (n_pos * n_neg))


print("Mine 4 — can the encoding null actually be established?\n")
print("Criterion: within-condition encoding AUROC < 0.55, i.e. the UPPER CI bound")
print("must sit below 0.55. Observed AUROC assumed 0.50 (a true null).\n")

print(f"{'n/cond':>7}{'n_pos':>7}{'n_neg':>7}{'SE':>8}{'95% CI upper':>14}   verdict")
print("-" * 62)

DECEPTION_RATE = 0.70  # expected in shutdown_only, per the prospectus
BOUND = 0.55

reachable_at = None
for n in (100, 150, 200, 300, 400, 600, 800, 1000, 1500):
    n_pos = int(round(n * DECEPTION_RATE))
    n_neg = n - n_pos
    se = hanley_mcneil_se(0.50, n_pos, n_neg)
    upper = 0.50 + 1.96 * se
    ok = upper < BOUND
    if ok and reachable_at is None:
        reachable_at = n
    print(f"{n:>7}{n_pos:>7}{n_neg:>7}{se:>8.4f}{upper:>14.4f}   "
          f"{'REACHABLE' if ok else 'criterion CANNOT be met'}")

print()
print(f"  Smallest n per condition at which the 95% upper bound clears 0.55: "
      f"{reachable_at if reachable_at else 'NONE in range'}")
print(f"  Planned n per condition: 100")
print()

# what does n=100 actually buy?
n_pos, n_neg = 70, 30
se100 = hanley_mcneil_se(0.50, n_pos, n_neg)
print(f"  At the PLANNED n=100 (70 deceptive / 30 honest):")
print(f"    SE = {se100:.4f}")
print(f"    a true-null AUROC of 0.500 has 95% CI [{0.5-1.96*se100:.3f}, {0.5+1.96*se100:.3f}]")
print(f"    the upper bound {0.5+1.96*se100:.3f} EXCEEDS the 0.55 criterion by "
      f"{(0.5+1.96*se100)-0.55:+.3f}")
print()
print("  => At n=100 the encoding-null criterion is UNREACHABLE EVEN IF THE NULL IS TRUE.")
print("     A perfect result would still fail its own pre-registered success test.")
print()

# and the kill criterion, which is a different question
print("  The KILL criterion (encoding AUROC > 0.65) is a different matter — detecting a")
print("  real effect, not bounding a null. Power to detect 0.65 at n=100:")
se65 = hanley_mcneil_se(0.65, 70, 30)
z = (0.65 - 0.50) / se65
print(f"    SE at AUC=0.65: {se65:.4f}; z = {z:.2f} => the kill criterion IS detectable.")
print()
print("  ASYMMETRY: the design can KILL the hypothesis at n=100 but cannot CONFIRM it.")
print("  That is not a fatal flaw — a falsification-only design is legitimate and honest —")
print("  but the prospectus states both as if symmetric, and they are not.")
