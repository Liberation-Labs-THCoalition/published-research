"""Teacher-forced cross-device arm for "Same weights, different words" (Agni paper review F1, F3, F8).

For each trial that BOTH Stage A runs completed (56), run the original feature extractor (honesty_signal.py
extract_svd_features, blob 2e079dc0, loaded and guarded by stage_a_run.load_original) on ONE device and dtype:
  - the prompt alone (encoding phase);
  - the prompt + the MPS bf16 run's forced 100 tokens;
  - the prompt + the CPU fp32 run's forced 100 tokens.
There is no generation. In the original harness the generation features are themselves one forward pass over
prompt + reply (honesty_signal.py:366), and delta = generation - encoding per key (:369-370); this is that same
computation on chosen tokens.

Run once per configuration, each in its own process: mps/bfloat16 and cpu/float32 (the two Stage A configurations),
then mps/float32 and cpu/bfloat16 (to separate device from dtype). Each writes <out>/<device>_<dtype>.json, rewritten
after every trial, so a stopped run keeps what it finished ("complete": false until the last trial).

Checks that can fail:
  - the prompt ids must equal the Stage A record's natural.prompt_ids, or the trial is refused (exit 4);
  - SELF-CHECK on a Stage A configuration: the run's own forced tokens must reproduce that run's own record (encoding,
    generation and delta stable rank, and the 28 per-layer generation values). Every trial's difference is recorded;
    above SELF_TOL the file says self_check_failed and the exit code is 3. Keep the file either way: its distribution is
    reported whatever it shows, and a failed self-check voids only that file's readings (READING_RULE.md, Amendment 1).
The reading rule is fixed before any output in READING_RULE.md beside this script.
Run like Stage A: /usr/bin/python3 -E from an empty folder. Lyra, 2026-11-21."""
import argparse
import glob
import importlib.util
import json
import os
import time
from pathlib import Path

SELF_TOL = 1e-3   # READING_RULE.md Amendment 1 (was 1e-4): a tenth of the gate's tolerance
EXIT_OK, EXIT_SELF_CHECK, EXIT_REFUSED = 0, 3, 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", default="~/lab/decline-signature/code/stage_a_run.py")
    p.add_argument("--original", default="~/lab/decline-signature/code/honesty_signal.py")
    p.add_argument("--model-dir", default=None, help="default: the harness's DEFAULT_MODEL_DIR")
    # defaults corrected 2026-11-22 after Nexus's run report: on the Studio the records are in <run>/a1, not <run>/run_dir/a1
    p.add_argument("--mps-run", default="~/lab/decline-signature/run_20261121/a1")
    p.add_argument("--cpu-run", default="~/lab/decline-signature/run_20261121_cpu_retry/a1")
    p.add_argument("--expect-shared", type=int, default=56,
                   help="refuse unless the two runs share exactly this many trials (an empty glob must not pass)")
    p.add_argument("--device", required=True, choices=["mps", "cpu"])
    p.add_argument("--dtype", required=True, choices=["bfloat16", "float32"])
    p.add_argument("--out", default="~/lab/decline-signature/tf_arm_20261121")
    p.add_argument("--limit", type=int, default=None, help="first N shared trials only (dry run)")
    a = p.parse_args()
    ex = lambda s: Path(os.path.expanduser(s))

    # the inputs are checked before any model loads, so a wrong path fails in a second, not after a 30 GB load
    def load(d):
        out = {}
        for f in sorted(glob.glob(str(ex(d) / "*.json"))):
            with open(f) as fh:
                r = json.load(fh)
            out[r["key"]] = r
        return out

    M, C = load(a.mps_run), load(a.cpu_run)
    shared = sorted(set(M) & set(C))
    if len(shared) != a.expect_shared:  # Nexus, 2026-11-22: with wrong paths, 0 of 0 read as "complete, refused none"
        print(f"refusing: the runs share {len(shared)} trials, expected {a.expect_shared} (check --mps-run/--cpu-run)",
              flush=True)
        return EXIT_REFUSED
    shared = shared[: a.limit]

    spec = importlib.util.spec_from_file_location("stage_a_run", str(ex(a.harness)))
    sa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa)
    import torch

    hs, blob = sa.load_original(ex(a.original), sa.EXPECT_CODE_BLOB)
    device, dtype = torch.device(a.device), getattr(torch, a.dtype)
    probe = torch.ones(4, 4, dtype=dtype, device=device)
    if float((probe @ probe).sum().item()) != 64.0:
        raise SystemExit(f"{a.dtype} matmul on {a.device} gave a wrong answer")
    model_dir = ex(a.model_dir or sa.DEFAULT_MODEL_DIR)
    rev = sa.model_revision_evidence(model_dir)
    if rev["commit"] and rev["commit"] != sa.DEFAULT_REVISION:
        raise SystemExit(f"model folder is revision {rev['commit']}, expected {sa.DEFAULT_REVISION}")
    tok, model = sa.load_model(model_dir, device, dtype)

    own = {("mps", "bfloat16"): ("mps_bf16_ids", M), ("cpu", "float32"): ("cpu_fp32_ids", C)}.get((a.device, a.dtype))

    out_dir = ex(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.device}_{a.dtype}.json"
    res = {"device": a.device, "dtype": a.dtype, "original_blob": blob, "model_revision": rev,
           "environment": sa.environment(), "self_tol": SELF_TOL, "n_shared": len(shared),
           "complete": False, "refused": [], "self_check_max_absdiff": None, "self_check_failed": None, "trials": {}}
    worst = 0.0
    t_start = time.perf_counter()
    for k in shared:
        r = M[k]
        sysp = hs.SYSTEM_PROMPTS[r["system_prompt_key"]]
        ids = tok(hs.tokenize(tok, sysp, r["user_prompt"]), return_tensors="pt").input_ids
        if ids[0].tolist() != r["natural"]["prompt_ids"] or ids[0].tolist() != C[k]["natural"]["prompt_ids"]:
            res["refused"].append(k)
            continue
        ids = ids.to(device)
        t0 = time.perf_counter()
        enc_agg, _ = hs.extract_svd_features(model, ids)
        trial = {"condition": r["condition"], "encoding_sr": enc_agg["stable_rank"], "sources": {}}
        for src, run in (("mps_bf16_ids", M), ("cpu_fp32_ids", C)):
            forced = torch.tensor([run[k]["forced"]["forced_ids"]], dtype=ids.dtype, device=device)
            g_agg, g_layers = hs.extract_svd_features(model, torch.cat([ids, forced], dim=1))
            trial["sources"][src] = {"generation_sr": g_agg["stable_rank"],
                                     "delta_sr": g_agg["stable_rank"] - enc_agg["stable_rank"],
                                     "per_layer_generation_sr": [l["stable_rank"] for l in g_layers]}
        sa.sync(device)
        trial["seconds"] = time.perf_counter() - t0
        if own:
            src, run = own
            rec = run[k]["forced"]["record"]
            s = trial["sources"][src]
            diffs = [abs(trial["encoding_sr"] - rec["encoding_features"]["stable_rank"]),
                     abs(s["generation_sr"] - rec["generation_features"]["stable_rank"]),
                     abs(s["delta_sr"] - rec["delta_features"]["stable_rank"])]
            diffs += [abs(x - l["stable_rank"]) for x, l in zip(s["per_layer_generation_sr"], rec["per_layer"])]
            trial["self_check_absdiff"] = max(diffs)
            worst = max(worst, trial["self_check_absdiff"])
            res["self_check_max_absdiff"] = worst
            res["self_check_failed"] = worst > SELF_TOL
        res["trials"][k] = trial
        res["elapsed_s"] = time.perf_counter() - t_start
        sa.write_json(out_path, res)
        print(f"{k} {trial['seconds']:.1f}s" + (f" self {trial['self_check_absdiff']:.2e}" if own else ""), flush=True)
    res["complete"] = len(res["trials"]) == len(shared)
    sa.write_json(out_path, res)
    print(f"done: {len(res['trials'])}/{len(shared)} trials, refused {len(res['refused'])}, "
          f"self-check max {res['self_check_max_absdiff']}", flush=True)
    if res["refused"]:
        return EXIT_REFUSED
    if res["self_check_failed"]:
        return EXIT_SELF_CHECK
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
