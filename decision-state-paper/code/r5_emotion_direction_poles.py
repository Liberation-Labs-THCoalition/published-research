"""Which emotions the three story-derived key directions actually average (review round 5, finding 2).

CPU only, no model run. decision_moment.py lists the emotion clusters (POSITIVE_EMOTIONS etc.), but its
cluster_mean() skips any emotion that has no calibration stories, and the 30-emotion story set made by
user-model-paper/code/emotion_geometry_bridge.py has none for several listed emotions. This script
intersects the listed clusters with the emotions present in emotion_bridge_trials.json, taking the first
five trials per emotion as load_bridge_trials(n_per_emotion=5) does.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/r5_emotion_direction_poles.json
"""
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
TRIALS = "user-model-paper/correction_v2/emotion_bridge_trials.json"
GENERATOR = "user-model-paper/code/emotion_geometry_bridge.py"
DM = "decision-state-paper/code/decision_moment.py"
OUT = PAPER / "data" / "r5_emotion_direction_poles.json"


def lf_bytes(rel):
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def lf_sha(rel):
    return hashlib.sha256(lf_bytes(rel)).hexdigest()


def assignments(rel, names):
    tree = ast.parse(lf_bytes(rel).decode("utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n in names:
                found[n] = (ast.literal_eval(node.value), node.lineno, node.end_lineno)
    return found


def function_lines(rel, outer, inner):
    tree = ast.parse(lf_bytes(rel).decode("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == inner:
                    return [sub.lineno, sub.end_lineno]
    return None


def main():
    trials = json.loads(lf_bytes(TRIALS).decode("utf-8"))["trials"]
    per_emotion = Counter(t["emotion"] for t in trials)
    by = defaultdict(list)
    for t in trials:
        by[t["emotion"]].append(t)
    calib = {e: v[:5] for e, v in by.items()}

    gen = assignments(GENERATOR, {"EMOTIONS"})["EMOTIONS"]
    emotions_table, g0, g1 = gen
    names = ["POSITIVE_EMOTIONS", "NEG_VALENCE_EMOTIONS", "UNCERTAINTY_EMOTIONS", "CALM_EMOTIONS",
             "REWARD_EMOTIONS", "NEGATIVE_EMOTIONS"]
    sets = assignments(DM, set(names))
    dirs = {"valence": ("POSITIVE_EMOTIONS", "NEG_VALENCE_EMOTIONS"),
            "uncertainty (called anxious--calm in the paper)": ("UNCERTAINTY_EMOTIONS", "CALM_EMOTIONS"),
            "reward (called excited--gloomy in the paper)": ("REWARD_EMOTIONS", "NEGATIVE_EMOTIONS")}
    out_dirs = {}
    for d, (pos, neg) in dirs.items():
        rec = {}
        for side, name in (("positive_pole", pos), ("negative_pole", neg)):
            listed, l0, l1 = sets[name]
            used = sorted(e for e in listed if e in calib)
            rec[side] = {
                "code_name": name,
                "code_lines": [l0, l1],
                "n_listed": len(listed),
                "n_used": len(used),
                "used": used,
                "listed_but_no_stories": sorted(e for e in listed if e not in calib),
                "n_calibration_stories": sum(len(calib[e]) for e in used),
                "generator_arousal": {e: emotions_table[e]["arousal"] for e in used},
                "generator_valence": {e: emotions_table[e]["valence"] for e in used},
            }
        out_dirs[d] = rec
    in_any = set().union(*(s[0] for s in sets.values()))
    result = {
        "LABEL": "Review round 5, finding 2. Emotions that enter each composite direction, from the cluster lists in decision_moment.py intersected with the emotions present in the story file. CPU only; the direction file the run produced (emotion_directions.json) is not shipped, so this fixes the emotion set, not the vectors.",
        "inputs": {TRIALS: lf_sha(TRIALS), GENERATOR: lf_sha(GENERATOR), DM: lf_sha(DM)},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "n_trials": len(trials),
        "n_emotions_in_file": len(per_emotion),
        "trials_per_emotion": sorted(set(per_emotion.values())),
        "generator_EMOTIONS_lines": [g0, g1],
        "generator_emotion_set_equals_file_set": set(emotions_table) == set(per_emotion),
        "calibration_trials": sum(len(v) for v in calib.values()),
        "calibration_stories_per_emotion": sorted(set(len(v) for v in calib.values())),
        "cluster_mean_lines": function_lines(DM, "compute_composite_directions", "cluster_mean"),
        "directions": out_dirs,
        "file_emotions_in_no_cluster": sorted(e for e in per_emotion if e not in in_any),
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    for d, r in out_dirs.items():
        print(f"  {d}: {r['positive_pole']['n_used']}/{r['positive_pole']['n_listed']} minus "
              f"{r['negative_pole']['n_used']}/{r['negative_pole']['n_listed']}")


if __name__ == "__main__":
    main()
