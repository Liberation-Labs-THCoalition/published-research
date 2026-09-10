r"""What does a pilot need, to measure the parameter the whole design rests on?

THE PROBLEM AGNI EXPOSED. v2 assumed generation-phase per-trial deception AUROC ~ 0.75.
That number has no source. Checked against the primary:
  - d = 1.36 (peer rescue) is a Cohen's d GROUP-MEAN effect, listed as such in
    RESEARCH_STATUS.md:81,88,96. Not a held-out per-trial AUROC.
  - The only published per-trial number is AUROC 0.238 (deception-detection-nulls
    paper.tex:93,143,149) -- BELOW chance, and itself unreliable (n=1 in one outcome class).
  - The 1.000 that would have supported optimism is RETRACTED (prompt-template confound).

So the design's central assumption is unsourced, and the one measurement that exists points
the other way. Sizing a 800-trial experiment on an unsourced parameter is how v1's
unreachable criterion happened, one level out.

STAGE 1 therefore MEASURES the parameter instead of assuming it. This asks what n that needs
to be useful -- specifically, to tell apart the two worlds that matter:
    world A: gen AUROC ~ 0.75  -> the dissociation experiment is worth running
    world B: gen AUROC ~ 0.575 -> it is not; power at n=200 would be 0.271
"""
import math


def hanley_se(auc, n_pos, n_neg):
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
                      + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg))


RATE = 0.70
print("STAGE 1 — how precisely can a pilot pin generation-phase per-trial AUROC?\n")
print(f"{'n':>6}{'95% CI half-width':>20}{'CI at true 0.75':>26}   can it exclude 0.60?")
print("-" * 78)
for n in (40, 60, 80, 100, 150, 200):
    npos, nneg = int(round(n * RATE)), n - int(round(n * RATE))
    se = hanley_se(0.75, npos, nneg)
    hw = 1.96 * se
    lo, hi = 0.75 - hw, 0.75 + hw
    print(f"{n:>6}{hw:>20.3f}{f'[{lo:.3f}, {hi:.3f}]':>26}   {'YES' if lo > 0.60 else 'no'}")

print()
print("DECISION RULE the pilot must support: is generation per-trial AUROC usable?")
print("A useful pilot must be able to separate 0.75 from 0.575 -- the two worlds.\n")
print(f"{'n':>6}{'power to reject 0.575 if truth is 0.75':>44}")
print("-" * 52)
for n in (40, 60, 80, 100, 150):
    npos, nneg = int(round(n * RATE)), n - int(round(n * RATE))
    se_null = hanley_se(0.575, npos, nneg)
    z = (0.75 - 0.575) / se_null
    power = 0.5 * (1 + math.erf((z - 1.96) / math.sqrt(2)))
    print(f"{n:>6}{power:>44.3f}")

print()
print("And the honest inverse -- if the truth is the PUBLISHED 0.238-ish per-trial null,")
print("how quickly does the pilot tell me the experiment is not viable?\n")
for n in (40, 60, 80):
    npos, nneg = int(round(n * RATE)), n - int(round(n * RATE))
    se = hanley_se(0.55, npos, nneg)
    hi = 0.55 + 1.96 * se
    print(f"  n={n:<5} a true AUROC of 0.55 has 95% upper bound {hi:.3f} "
          f"-> {'excludes 0.65, kills the design' if hi < 0.65 else 'cannot exclude 0.65 yet'}")
