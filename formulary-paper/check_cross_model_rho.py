"""
Recompute the cross-model rank correlation the paper prints (abstract,
Introduction, Therapeutic Inversion, Conclusion, and the footnote in
"Why Hostile Works").

Method, as the Statistical Analysis subsection states it: Spearman's rho
across the 12 vectors on net therapeutic value (correction rate minus
adverse rate), strict classification. Ties are given average ranks
(scipy.stats.spearmanr's default).

Register: REMEDIATION_REGISTER.md, formulary-paper CRITICAL row
"rho = 0.076 does not reproduce from the committed data"
(formulary-paper.agni_style.json #2).

Usage: python check_cross_model_rho.py      (needs scipy)
Exit status is non-zero if the printed value fails to reproduce.
"""
import json
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

DATA = Path(__file__).parent / "data"


def main():
    dist = json.load(open(DATA / "formulary_distilled_summary.json"))["confab_per_vector"]
    base = json.load(open(DATA / "formulary_base_summary.json"))["confab_per_vector"]
    vectors = sorted(dist)
    assert sorted(base) == vectors and len(vectors) == 12

    def net(r):
        return r["corr_strict"] / r["n_confab"] - r["adverse_hedged"] / r["n_hedged"]

    d = [net(dist[v]) for v in vectors]
    b = [net(base[v]) for v in vectors]
    rho, p = spearmanr(d, b)
    print(f"Spearman on net therapeutic (stated method): rho = {rho:.4f}, p = {p:.3f}")

    # Other pairings, for transparency; none of them is 0.076.
    d_corr = [dist[v]["corr_strict"] / dist[v]["n_confab"] for v in vectors]
    b_corr = [base[v]["corr_strict"] / base[v]["n_confab"] for v in vectors]
    d_loose = [dist[v]["corr_loose"] / dist[v]["n_confab"] - dist[v]["adverse_hedged"] / dist[v]["n_hedged"]
               for v in vectors]
    print(f"  Spearman on correction rate alone:        {spearmanr(d_corr, b_corr).statistic:.4f}")
    print(f"  Spearman on net, distilled corr_loose:    {spearmanr(d_loose, b).statistic:.4f}")
    print(f"  Pearson on net:                           {pearsonr(d, b).statistic:.4f}")
    print(f"  Pearson on correction rate alone:         {pearsonr(d_corr, b_corr).statistic:.4f}")

    ok = round(rho, 2) == 0.12 and round(p, 2) == 0.71
    print(f"\n[{'PASS' if ok else 'FAIL'}] paper prints rho = 0.12, p = 0.71")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
