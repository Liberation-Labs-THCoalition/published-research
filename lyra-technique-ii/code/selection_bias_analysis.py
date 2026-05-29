"""
LT2-B1: Quantify max-over-grid selection bias in denoising deltas.

Dwayne's audit: best_denoised = max(raw, denoised[k]) is scored on the
same CV that reports it, with no nested loop. Max-over-grid manufactures
positive AUROC deltas by chance, which may fully account for the observed
"denoising helps" deltas.

This reanalysis quantifies the selection bias from the summary statistics
present in the scale_sweep files (raw, denoised[rank], null_std). For each
task we simulate the max-over-grid selection under the null sampling noise
and compare the expected inflation to the OBSERVED delta.

LIMITATION: A definitive test requires nested CV on the raw 150x5 feature
matrices, which are not in the aggregated files (they need GPU re-extraction).
This analysis uses null_std as the AUROC estimation-noise proxy (Dwayne's
SE~0.06 assumption, confirmed by the data: null_std ~ 0.062). It establishes
whether the observed deltas exceed the selection-noise envelope.

Verdict logic:
  - If observed delta <= 95th percentile of null max-over-grid inflation:
    delta is INDISTINGUISHABLE from selection noise.
  - If observed delta > 95th percentile: denoising delta SURVIVES.
"""
import json
import glob
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
np.random.seed(42)
N_SIM = 100000


def selection_envelope(raw, n_grid, se, n_sim=N_SIM):
    """
    Under H0 (denoising does nothing), each grid estimate is the true raw
    AUROC plus N(0, se) estimation noise. best_denoised = max(raw, grid...).
    Returns the distribution of (best - raw) inflation.
    """
    # raw is one estimate; the grid adds n_grid more noisy estimates of the
    # same underlying value. max-over-(1+n_grid) noisy draws, minus raw.
    # Conservative model: all draws ~ N(raw, se); inflation = max(draws) - raw_obs
    draws = np.random.normal(raw, se, size=(n_sim, n_grid + 1))
    best = draws.max(axis=1)
    return best - raw


def main():
    files = sorted(glob.glob(str(DATA / "canonical_scale_sweep_*_results.json")))
    print(f"Analyzing {len(files)} scale-sweep models\n")
    print(f"{'Model':<38} {'Task':<20} {'raw':>6} {'delta':>7} {'env95':>7} {'verdict':>14}")
    print("-" * 95)

    rows = []
    for f in files:
        model = Path(f).stem.replace("canonical_scale_sweep_", "").replace("_results", "")
        d = json.load(open(f))
        for task in d.get("tasks", []):
            raw = task["raw"]
            delta = task["delta"]
            denoised = task.get("denoised", {})
            n_grid = len(denoised) if isinstance(denoised, dict) else 1
            se = task.get("null_std", 0.062)
            if delta <= 0:
                verdict = "no-help"
                env95 = 0.0
            else:
                env = selection_envelope(raw, n_grid, se)
                env95 = np.percentile(env, 95)
                verdict = "SURVIVES" if delta > env95 else "selection-noise"
            rows.append({
                "model": model, "task": task["task"], "raw": raw,
                "delta": delta, "n_grid": n_grid, "se": se,
                "env95": env95, "verdict": verdict,
            })
            task_short = task["task"][:20]
            print(f"{model:<38} {task_short:<20} {raw:>6.3f} {delta:>7.3f} {env95:>7.3f} {verdict:>14}")

    # Summary
    confab_rows = [r for r in rows if "confab" in r["task"].lower()]
    helps = [r for r in rows if r["delta"] > 0]
    survives = [r for r in rows if r["verdict"] == "SURVIVES"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total task-model cells: {len(rows)}")
    print(f"Cells with positive delta (denoising 'helps'): {len(helps)}")
    print(f"Cells where delta SURVIVES selection envelope: {len(survives)}")
    print(f"Cells where delta is selection-noise: {len(helps) - len(survives)}")
    print()
    print("Confabulation tasks specifically:")
    for r in confab_rows:
        print(f"  {r['model']:<38} delta={r['delta']:.3f}, env95={r['env95']:.3f} -> {r['verdict']}")

    n_confab_survive = sum(1 for r in confab_rows if r["verdict"] == "SURVIVES")
    print(f"\nConfab denoising deltas surviving selection bias: "
          f"{n_confab_survive}/{len(confab_rows)}")

    out = DATA / "selection_bias_results.json"
    json.dump({"n_sim": N_SIM, "rows": rows,
               "n_positive_delta": len(helps),
               "n_survives": len(survives),
               "confab_survive": n_confab_survive,
               "confab_total": len(confab_rows)},
              open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
