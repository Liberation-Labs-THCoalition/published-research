#!/usr/bin/env python3
"""Run DESIGN.md's pre-registered dev selection and write the frozen config. Dev only.

Each stage sweeps its grid on top of the previous winner, at a 16k budget. The winner has the highest
dev all-evidence recall. Ties go to the simpler config: fewer channels, then fewer parameters that
differ from v10.DEFAULT, then grid order. Budget: the smallest of BUDGETS with recall >= 0.95, else
32k. The frozen config is written read-only, and every stage's table goes to runs/stages_<stamp>.json.
Nothing here reads the held-out split.
"""
import itertools, json, os, sys, time
sys.path.insert(0, "/mnt/data1/lme_v2/v10")
import v10

ROOT = "/mnt/data1/lme_v2/v10"
BUDGETS = [8500, 12000, 16000, 24000, 32000]
STAGES = [
    ("s1_channels", {"channels": [["bm25"], ["dense"], ["bm25", "dense"]]}),
    ("s2_scoring", {"second_weight": [0.0, 0.5, 1.0], "w_dense": [0.5, 1.0, 2.0]}),
    ("s3_packing", {"whole_top": [99, 5, 3], "window": [2, 3]}),
    ("s4_time", {"time": [False, True], "w_time": [0.5, 1.0]}),
]

def complexity(cfg):
    diffs = sum(cfg[k] != v for k, v in v10.DEFAULT.items() if k not in ("channels", "budget"))
    return (len(cfg["channels"]), diffs)

def main():
    data, idxs = v10.load_split("dev")
    v9 = {r["qid"]: r for r in json.load(open(v10.V9_AUTOPSY))}
    dense = v10.Dense()
    best = {**v10.DEFAULT, "budget": 16000}
    log = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": []}
    for name, grid in STAGES:
        rows = []
        for vals in itertools.product(*grid.values()):
            over = dict(zip(grid, vals))
            cfg = dict(best)
            if name == "s4_time":
                if not over["time"] and over["w_time"] != grid["w_time"][0]:
                    continue                                   # "off" is one config, not two
                chans = [c for c in best["channels"] if c != "time"] + (["time"] if over["time"] else [])
                cfg.update(channels=chans, w_time=over["w_time"] if over["time"] else v10.DEFAULT["w_time"])
            else:
                cfg.update(over)
            s = v10.summarise(v10.evaluate(data, idxs, cfg, dense), v9)
            rows.append({"over": over, "cfg": cfg, "hash": v10.cfg_hash(cfg), "all_sessions": float(s["all_sessions"]),
                         "median_tokens": s["median_tokens"], "by_type": s["by_type"]})
            print(f"{name} {json.dumps(over):60} all_sess={s['all_sessions']:.3f} tok={s['median_tokens']:.0f}", flush=True)
        top = max(r["all_sessions"] for r in rows)
        winner = min((r for r in rows if r["all_sessions"] == top), key=lambda r: complexity(r["cfg"]))
        best = dict(winner["cfg"])
        print(f"  -> {name} winner {winner['hash']} {json.dumps(winner['over'])} all_sess={top:.3f}", flush=True)
        log["stages"].append({"stage": name, "grid": grid, "rows": rows, "winner": winner["hash"]})

    curve = []
    for b in BUDGETS:
        cfg = {**best, "budget": b}
        s = v10.summarise(v10.evaluate(data, idxs, cfg, dense), v9)
        curve.append({"budget": b, "all_sessions": float(s["all_sessions"]), "median_tokens": s["median_tokens"],
                      "by_type": s["by_type"], "v9_strict": float(s["v9_all_sessions_strict"]),
                      "v9_lenient": float(s["v9_all_sessions_lenient"])})
        print(f"budget {b:6} all_sess={s['all_sessions']:.3f} tok={s['median_tokens']:.0f}", flush=True)
    chosen = next((c["budget"] for c in curve if c["all_sessions"] >= 0.95), BUDGETS[-1])
    best["budget"] = chosen
    log.update(curve=curve, chosen_budget=chosen, frozen=best, frozen_hash=v10.cfg_hash(best),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    json.dump(log, open(f"{ROOT}/runs/stages_{stamp}.json", "w"), indent=1, default=float)
    frozen = f"{ROOT}/frozen_v10.json"
    assert not os.path.exists(frozen), "frozen_v10.json already exists; refusing to overwrite a frozen config"
    json.dump(best, open(frozen, "w"), indent=1)
    os.chmod(frozen, 0o444)
    print(f"FROZEN {v10.cfg_hash(best)} budget {chosen} -> {frozen}")

if __name__ == "__main__":
    main()
