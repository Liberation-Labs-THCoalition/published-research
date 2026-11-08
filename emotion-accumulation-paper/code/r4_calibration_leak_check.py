"""Agni r4 review, finding 11: does the compass's calibration set include stories that carry the
generator's leaked reasoning?

The compass directions (emotion_directions.json, not shipped) are built by
decision-state-paper/code/decision_moment.py, load_bridge_trials(): the first five stories per
emotion, in file order, from emotion_bridge_trials.json. This script applies the PRE-SET leak
pattern of user-model-paper/correction_v2/leak_split.py (the one R1R2_RESULTS.md reports) to that
selection of the released stories file, and records which records match. It writes no story text
beyond each matched story's first five words.

It assumes the compass was calibrated from this released file in this order; the direction file
that would confirm it is not shipped.

    python r4_calibration_leak_check.py [--repo <published-research root>] [--out <json>]
"""
import argparse, ast, hashlib, json, re
from collections import defaultdict
from pathlib import Path

RELEASED_TRIALS_SHA256 = "fc5938feccc57c68db0c1625236839d7405708219911dc2c532de836eb836e7e"  # leak_split.py SHA["trials"]
# Verbatim from user-model-paper/correction_v2/leak_split.py (LEAK, the pre-set pattern).
LEAK = re.compile(r"<analysis>|</?think|The user wants|the user (is asking|wants)|key constraint", re.I)
N_PER_EMOTION = 5  # decision_moment.py: calibrate() calls load_bridge_trials(bridge_dir, n_per_emotion=5)


def cluster_sets(dm_path):
    tree = ast.parse(dm_path.read_text(encoding="utf-8"))
    sets = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "").endswith("_EMOTIONS"):
            sets[node.targets[0].id] = ast.literal_eval(node.value)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Path(a.repo)
    src = repo / "user-model-paper" / "correction_v2" / "emotion_bridge_trials.json"
    dm = repo / "decision-state-paper" / "code" / "decision_moment.py"
    out_path = Path(a.out) if a.out else repo / "emotion-accumulation-paper" / "data" / "r4_calibration_leak_check_results.json"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    if got != RELEASED_TRIALS_SHA256:
        raise SystemExit("stories file is not the released one; refusing")
    trials = json.loads(src.read_text(encoding="utf-8"))["trials"]

    # Same selection as decision_moment.load_bridge_trials: group by emotion in file order, first five each.
    by_emotion = defaultdict(list)
    for i, t in enumerate(trials):
        by_emotion[t["emotion"]].append(i)
    selected = [i for idxs in by_emotion.values() for i in idxs[:N_PER_EMOTION]]

    sets = cluster_sets(dm)
    directions = {  # decision_moment.compute_composite_directions
        "valence": ("POSITIVE_EMOTIONS", "NEG_VALENCE_EMOTIONS"),
        "uncertainty (anxious-calm)": ("UNCERTAINTY_EMOTIONS", "CALM_EMOTIONS"),
        "reward": ("REWARD_EMOTIONS", "NEGATIVE_EMOTIONS"),
    }

    def feeds(emotion):
        return [name for name, (pos, neg) in directions.items()
                if emotion in sets.get(pos, set()) or emotion in sets.get(neg, set())]

    matches = []
    for i in selected:
        t = trials[i]
        if LEAK.search(t["story_text"]):
            m = LEAK.search(t["story_text"])
            matches.append({"record_index_0based": i, "emotion": t["emotion"], "topic_idx": t["topic_idx"],
                            "story_idx": t["story_idx"], "matched_alternative": m.group(0),
                            "first_five_words": " ".join(t["story_text"].split()[:5]),
                            "feeds_directions": feeds(t["emotion"])})
    all_leaked = sum(bool(LEAK.search(t["story_text"])) for t in trials)
    used = {e for e in by_emotion if feeds(e)}
    res = {
        "script": "emotion-accumulation-paper/code/r4_calibration_leak_check.py",
        "input": "user-model-paper/correction_v2/emotion_bridge_trials.json",
        "input_sha256": got,
        "pattern": LEAK.pattern,
        "n_stories_in_file": len(trials),
        "n_matching_in_whole_file": all_leaked,
        "n_emotions": len(by_emotion),
        "n_selected_first_five_per_emotion": len(selected),
        "n_selected_from_emotions_that_feed_a_direction": sum(trials[i]["emotion"] in used for i in selected),
        "n_selected_matching": len(matches),
        "matches": matches,
        "caveat": "assumes the compass was calibrated from this released file in this order; emotion_directions.json is not shipped",
    }
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "matches"}, ensure_ascii=False))
    for m in matches:
        print(json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
