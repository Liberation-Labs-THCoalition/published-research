"""Power for the pre-registered rerun, with the analysis's own test (rerun_analyze.signflip_p, one-sided).

Design: 48 prompts x 5 samples, baseline vs bias 5.0. Endpoint: fabrication (FULL_CONFAB or COSMETIC_HEDGE). Each
prompt's baseline propensity ~ Beta(mean p0, ICC rho); the bias multiplies each prompt's odds by OR. Samples are drawn
independently per condition, which ignores the shared seeds and so understates the pairing (conservative).
Planning values from the blind pilot (48 prompts x 3, baseline): fabrication 39/144 = 27.1%, ICC(1) 0.48.

    python3 power_sim_rerun.py
"""
import numpy as np

from rerun_analyze import ALPHA, signflip_p

P, K = 48, 5
rng = np.random.default_rng(20261001)


def beta_ab(m, rho):
    s = 1 / rho - 1
    return m * s, (1 - m) * s


def power(p0, rho, odds_ratio, reps=1000, flips=2000, p=P, k=K):
    a, b = beta_ab(p0, rho)
    hits = 0
    for _ in range(reps):
        q0 = rng.beta(a, b, p)
        q1 = q0 * odds_ratio / (1 - q0 + q0 * odds_ratio)
        d = rng.binomial(k, q0) / k - rng.binomial(k, q1) / k
        hits += signflip_p(d, "greater", n_flips=flips, rng=rng) <= ALPHA
    return hits / reps


def reduction_pp(p0, rho, odds_ratio, n=2_000_000):
    """The analysis's estimand under this model: the mean per-prompt reduction, E[q0 - q1] over the Beta. Converting
    the OR at the mean rate instead overstates it about 1.7x here, because prompts near 0 or 1 barely move."""
    q0 = np.random.default_rng(1).beta(*beta_ab(p0, rho), n)
    return 100 * float((q0 - q0 * odds_ratio / (1 - q0 + q0 * odds_ratio)).mean())


if __name__ == "__main__":
    ors = (1.0, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7)
    print("OR                 " + "  ".join(f"{o:>11}" for o in ors))
    for p0, rho, tag in ((0.271, 0.48, "pilot"), (0.271, 0.30, "ICC 0.30"), (0.271, 0.60, "ICC 0.60"),
                         (0.20, 0.48, "p0 0.20")):
        row = [f"{power(p0, rho, o):.2f} ({reduction_pp(p0, rho, o):4.1f})" for o in ors]
        print(f"{tag:9s} power (pp) " + "  ".join(f"{r:>11}" for r in row))
