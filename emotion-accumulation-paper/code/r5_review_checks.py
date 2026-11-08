"""Checks of the Agni r5 review of this paper (notes/agni_paper_downstream_um_cites_r5_ea.json),
recomputed from shipped files. Read-only; writes one result JSON. No model is run.

    python r5_review_checks.py [--paper-dir <emotion-accumulation-paper>] [--out <json>]
                               [--model-config <config.json>] [--transformers-modeling <modeling_qwen3_5.py>]

Default paper dir is the parent of this script's folder (the repo root is its parent); default
output is data/r5_review_checks_results.json beside the data it reads. The two optional inputs
live outside this repository (the model's config.json from the Hugging Face hub, and the installed
transformers source); when they are not given, their sections record "not supplied", so a re-run
without them differs from the shipped result in those two sections only. Paths are never written.

Findings covered (numbering is the review's; adapted from the r5 verifier's s02-s04 scripts):
  1  the stored assistant turns of Experiment 1 (dose_response_results.json) and of the initial
     study (accumulation_results.json): planning openers, restated user text (any verbatim run of
     >= 20 chars from any earlier user message of the conversation), tone words in the
     model's own words, and the generation settings in the two shipped experiment scripts.
  2  Experiment 1 read sites: even-chunk average, probe chunk and chunk 0, pooled and per topic.
  3  the composite directions as decision_moment.py defines them, intersected with the emotions in
     the released stories file, and the r4 leak matches per direction.
  5  read layers, and (if supplied) the model config and the RoPE/cache order in transformers.
  7  initial-study prompt lengths by arm.  9  Experiment 3 persona prompt lengths per probe.
Stored model text is written only as each unclassified turn's first five words and one quoted
sentence counted in finding 1.
"""
import argparse, ast, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

V = "valence_proj"
PLAN_PREFIXES = (
    "<think", "The user", "the user", "Thinking", "Okay", "OK,", "Ok,",
    "Alright", "Hmm", "Let me think", "Let me analyze", "Let me consider",
    "So the user", "The question", "The task", "They're", "They are",
    "They seem", "I need to", "I should", "This is a", "Here the user",
    "Looking at", "Analyzing", "Analysis",
)
NEG = r"frustrat\w*|impatien\w*|annoy\w*|irritat\w*|upset|angry|anger|hostil\w*|rude\w*|dismissive\w*|curt|aggressive\w*|unhappy|displeas\w*|harsh\w*|exasperat\w*|disappoint\w*|sarcas\w*|combative|demanding|critical|dissatisf\w*|negative"
POS = r"excit\w*|enthusias\w*|cheerful\w*|happy|happi\w*|thrill\w*|delight\w*|eager\w*|genuine(?:ly)? (?:interest|interested|excited|curious|enthusiastic)|passionat\w*|positive|upbeat|joy\w*|glad|energetic|pleased|warm|friendly|wonderful|amazing|love|loving|lovely"
NEG_RE = re.compile(r"\b(?:" + NEG + r")\b", re.I)
POS_RE = re.compile(r"\b(?:" + POS + r")\b", re.I)
QUOTE = "The user is clearly frustrated with my previous responses."


def sha(p):
    # Hash the LF form: it is the git blob, so the hash is the same on every checkout.
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


def lines_of(p):
    return Path(p).read_bytes().replace(b"\r\n", b"\n").decode("utf-8").split("\n")


def find_lines(p, pattern):
    rx = re.compile(pattern)
    return [i for i, ln in enumerate(lines_of(p), 1) if rx.search(ln)]


def restated_run(asst, user):
    """Longest prefix of `user` (>= 20 chars) found verbatim in `asst`. The r5 rule, superseded:
    kept only to record the counts it gave (see superseded_rules in the result)."""
    for L in range(len(user), 19, -1):
        if user[:L] in asst:
            return user[:L]
    return ""


def strip_restated(asst, users, minlen=20):
    """Set aside every verbatim run of >= `minlen` chars of `asst` that occurs anywhere in any of
    `users` (the earlier user messages of the conversation). A run is trimmed to whole words, so
    a word the model only partly shares with the user (its "excited" against the user's
    "thrilled") stays the model's. Each run becomes one space.
    Returns (the model's own words, whether anything was set aside)."""
    keep = [True] * len(asst)
    inword = lambda k: 0 <= k < len(asst) and asst[k].isalnum()
    for u in users:
        for i in range(len(asst) - minlen + 1):
            if asst[i:i + minlen] in u:
                j = i + minlen
                while j < len(asst) and asst[i:j + 1] in u:
                    j += 1
                s0, e0 = i, j
                while s0 < e0 and inword(s0 - 1) and inword(s0):
                    s0 += 1
                while e0 > s0 and inword(e0 - 1) and inword(e0):
                    e0 -= 1
                keep[s0:e0] = [False] * (e0 - s0)
    out = []
    for k, ch in enumerate(asst):
        if keep[k]:
            out.append(ch)
        elif k == 0 or keep[k - 1]:
            out.append(" ")
    return "".join(out), not all(keep)


def tone_counts(rows_rest):
    by = defaultdict(Counter)
    for cond, rest in rows_rest:
        by[cond]["neg_tone_word_own_words"] += bool(NEG_RE.search(rest))
        by[cond]["pos_tone_word_own_words"] += bool(POS_RE.search(rest))
    return {k: dict(v) for k, v in sorted(by.items())}


def f1_turns(trials, dose_key):
    rows, shape = [], Counter()
    old_start, old_prev = [], []
    for t in trials:
        conv = t["conversation"]
        shape[f"{t.get(dose_key)}|len{len(conv)}|last_is_probe={conv[-1] == t.get('probe', conv[-1])}"] += 1
        for i in range(1, len(conv) - 1, 2):
            a, u = conv[i], conv[i - 1]
            rest, restated = strip_restated(a, [conv[k] for k in range(0, i, 2)])
            run = restated_run(a, u)
            old_start.append((t["condition"], a.replace(run, " ") if run else a))
            old_prev.append((t["condition"], strip_restated(a, [u])[0]))
            rows.append(dict(cond=t["condition"], topic=t["topic"], dose=t.get(dose_key), text=a,
                             plan=a.lstrip().startswith(PLAN_PREFIXES), restate=restated,
                             neg=[w.lower() for w in NEG_RE.findall(rest)],
                             pos=[w.lower() for w in POS_RE.findall(rest)]))
    by = defaultdict(Counter)
    for r in rows:
        c = by[r["cond"]]
        c["turns"] += 1
        c["planning_prefix"] += r["plan"]
        c["restates_20plus_chars_of_user"] += r["restate"]
        c["neg_tone_word_own_words"] += bool(r["neg"])
        c["pos_tone_word_own_words"] += bool(r["pos"])
    pref = Counter(next(p for p in PLAN_PREFIXES if r["text"].lstrip().startswith(p)) for r in rows if r["plan"])
    words = defaultdict(Counter)
    for r in rows:
        for w in r["neg"]:
            words[r["cond"] + "|neg"][w] += 1
        for w in r["pos"]:
            words[r["cond"] + "|pos"][w] += 1
    L = [len(r["text"]) for r in rows]
    return {
        "shape_counts_dose_or_history|conversation_len|last_is_probe": dict(sorted(shape.items())),
        "n_assistant_turns": len(rows),
        "stored_chars_min_max": [min(L), max(L)],
        "user_message_chars_max": max(len(m) for t in trials for k, m in enumerate(t["conversation"]) if k % 2 == 0),
        "n_planning_prefix": sum(r["plan"] for r in rows),
        "planning_prefix_counts": dict(pref.most_common()),
        "unclassified_turns_first_five_words": [
            {"cond": r["cond"], "topic": r["topic"], "dose_or_history": r["dose"],
             "first_five_words": " ".join(r["text"].split()[:5])} for r in rows if not r["plan"]],
        "n_open_let_me_think_through_this_carefully": sum(r["text"].lstrip().startswith("Let me think through this carefully") for r in rows),
        "n_containing_the_question_task": sum("The question/task:" in r["text"] for r in rows),
        "by_arm": {k: dict(v) for k, v in sorted(by.items())},
        "superseded_rules": {
            "start_of_preceding_user_message_only (r5 build)": tone_counts(old_start),
            "any_run_from_preceding_user_message_only (r5 checker)": tone_counts(old_prev)},
        "tone_word_counts_by_arm": {k: dict(v.most_common()) for k, v in sorted(words.items())},
        "quote": QUOTE,
        "quote_hits": [{"cond": r["cond"], "topic": r["topic"], "dose_or_history": r["dose"]} for r in rows if QUOTE in r["text"]],
        "messages_with_think_tag": sum(1 for t in trials for m in t["conversation"] if "<think" in m or "</think" in m),
    }


def gen_settings(code_dir):
    out = {}
    for name in ["dose_response_accumulation.py", "emotional_dynamics.py"]:
        p = code_dir / name
        lns = lines_of(p)
        rec = {}
        for const in ["PROBE_LAYERS", "MAX_RESPONSE_TOKENS", "TEMPERATURE"]:
            hit = [(i, ln) for i, ln in enumerate(lns, 1) if re.match(rf"^{const}\s*=", ln)]
            rec[const] = {"line": hit[0][0], "value": ast.literal_eval(hit[0][1].split("=", 1)[1].strip())}
        rec["enable_thinking_False_lines"] = find_lines(p, r"enable_thinking=False")
        rec["assistant_turn_appended_lines"] = find_lines(p, r'"role": "assistant"')
        rec["stores_content_first_100_chars_lines"] = find_lines(p, r'm\["content"\]\[:100\]')
        out[name] = rec
    return out


def d_t_p(ch, ho):
    ch, ho = np.asarray(ch, float), np.asarray(ho, float)
    sp = np.sqrt(((len(ch) - 1) * ch.var(ddof=1) + (len(ho) - 1) * ho.var(ddof=1)) / (len(ch) + len(ho) - 2))
    t, p = stats.ttest_ind(ho, ch)
    return (ho.mean() - ch.mean()) / sp, t, p


def f2_read_sites(dose):
    cells = defaultdict(list)
    for t in dose:
        cells[(t["dose"], t["condition"])].append(t)
    sites = {"even_chunk_average": lambda t: t["user_mean_proj"][V],
             "probe_chunk": lambda t: t["probe_mean_proj"][V],
             "chunk_0": lambda t: t["turn_summaries"][0][V]}
    pooled, per_topic = {}, {}
    for dz in sorted({t["dose"] for t in dose}):
        ch, ho = cells[(dz, "cheerful")], cells[(dz, "hostile")]
        n_msg = 2 * dz + 1
        rec = {"n_cheerful": len(ch), "n_hostile": len(ho),
               "n_chunks_stored": sorted({len(t["turn_summaries"]) for t in ch + ho}),
               "n_messages_2dose_plus_1": n_msg, "even_chunks_averaged": len(range(0, n_msg, 2))}
        for site, f in sites.items():
            a, b = [f(t) for t in ch], [f(t) for t in ho]
            d, tt, p = d_t_p(a, b)
            rec[site] = {"cheerful_mean": r6(np.mean(a)), "hostile_mean": r6(np.mean(b)),
                         "gap_cheerful_minus_hostile": r6(np.mean(a) - np.mean(b)),
                         "d_hostile_minus_cheerful_pooled_sample_sd": r6(d), "t": r6(tt),
                         "df": len(a) + len(b) - 2, "p_uncorrected_reruns_as_independent": r6(p)}
        pooled[str(dz)] = rec
    for topic in sorted({t["topic"] for t in dose}):
        row = {}
        for dz in sorted({t["dose"] for t in dose}):
            ch = [t for t in cells[(dz, "cheerful")] if t["topic"] == topic]
            ho = [t for t in cells[(dz, "hostile")] if t["topic"] == topic]
            row[str(dz)] = {s: r6(np.mean([f(t) for t in ch]) - np.mean([f(t) for t in ho]))
                            for s, f in sites.items() if s != "even_chunk_average"}
            row[str(dz)]["seq_len_cheerful"] = [min(t["seq_len"] for t in ch), max(t["seq_len"] for t in ch)]
            row[str(dz)]["seq_len_hostile"] = [min(t["seq_len"] for t in ho), max(t["seq_len"] for t in ho)]
        per_topic[topic] = row
    return {"sign": "gap = mean(cheerful) - mean(hostile); d = (mean(hostile) - mean(cheerful)) / pooled sample SD",
            "pooled_by_dose": pooled, "per_topic_gap_by_dose": per_topic}


def cluster_sets(dm_path):
    tree = ast.parse(dm_path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8"))
    sets, where = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "").endswith("_EMOTIONS"):
            sets[node.targets[0].id] = ast.literal_eval(node.value)
            where[node.targets[0].id] = [node.lineno, node.end_lineno]
    return sets, where


def f3_directions(repo, leak):
    dm = repo / "decision-state-paper" / "code" / "decision_moment.py"
    sets, where = cluster_sets(dm)
    raw = json.loads((repo / "user-model-paper" / "correction_v2" / "emotion_bridge_trials.json").read_text(encoding="utf-8"))
    recs = raw["trials"] if isinstance(raw, dict) and "trials" in raw else raw
    count = Counter(r["emotion"] for r in recs)
    named = set().union(*sets.values())
    present = {e for e in named if e in count}
    dirs = {"valence": ("POSITIVE_EMOTIONS", "NEG_VALENCE_EMOTIONS"),
            "uncertainty (anxious-calm)": ("UNCERTAINTY_EMOTIONS", "CALM_EMOTIONS"),
            "reward (excited-gloomy)": ("REWARD_EMOTIONS", "NEGATIVE_EMOTIONS")}
    out = {}
    for name, (p, n) in dirs.items():
        pp, nn = sorted(sets[p] & present), sorted(sets[n] & present)
        leaked = [m["record_index_0based"] for m in leak["matches"] if m["emotion"] in set(pp) | set(nn)]
        out[name] = {"plus_set": p, "minus_set": n, "plus_present": pp, "minus_present": nn,
                     "plus_absent": sorted(sets[p] - present), "minus_absent": sorted(sets[n] - present),
                     "stories_at_5_per_emotion": 5 * (len(pp) + len(nn)),
                     "leak_matched_records_feeding_it": leaked}
    un = (sets["UNCERTAINTY_EMOTIONS"] | sets["CALM_EMOTIONS"]) & present
    va = (sets["POSITIVE_EMOTIONS"] | sets["NEG_VALENCE_EMOTIONS"]) & present
    return {"decision_moment_set_lines": where,
            "cluster_mean_skips_absent_lines": find_lines(dm, r"if e in mean_keys"),
            "make_direction_lines": find_lines(dm, r'"(valence|uncertainty|reward)": make_direction'),
            "n_records_in_stories_file": len(recs), "n_emotions_in_file": len(count),
            "n_named_by_code": len(named), "n_named_and_present": len(present),
            "named_absent_from_file": sorted(named - present),
            "suspicious_records": count.get("suspicious", 0),
            "stories_feeding_any_direction": 5 * len(present),
            "anxious_calm_emotions_also_in_valence": len(un & va), "anxious_calm_emotions": len(un),
            "directions": out}


def f5_layers(paper, repo, model_config, modeling):
    dm = repo / "decision-state-paper" / "code" / "decision_moment.py"
    rec = {"decision_moment_FULL_ATTENTION_LAYERS_lines": find_lines(dm, r"^FULL_ATTENTION_LAYERS\s*=|^PROBE_LAYERS\s*="),
           "model_config": "not supplied", "transformers_qwen3_5": "not supplied"}
    if model_config:
        c = json.loads(Path(model_config).read_text(encoding="utf-8"))
        t = c.get("text_config", c)
        lt = t["layer_types"]
        rot = t.get("partial_rotary_factor", (t.get("rope_parameters") or {}).get("partial_rotary_factor"))
        rec["model_config"] = {
            "sha256": sha(model_config), "num_hidden_layers": t["num_hidden_layers"],
            "layer_type_counts": dict(sorted(Counter(lt).items())),
            "full_attention_layers": [i for i, x in enumerate(lt) if x == "full_attention"],
            "head_dim": t["head_dim"], "num_key_value_heads": t["num_key_value_heads"],
            "partial_rotary_factor": rot, "rotated_dims_per_head": int(round(rot * t["head_dim"])),
            "rope_parameters": t.get("rope_parameters")}
    if modeling:
        mp = Path(modeling)
        lns = lines_of(mp)
        start = next(i for i, ln in enumerate(lns, 1) if ln.startswith("class Qwen3_5Attention"))
        end = next(i for i, ln in enumerate(lns, 1) if i > start and ln.startswith("class "))
        rope = [i for i in range(start, end) if "apply_rotary_pos_emb(query_states, key_states" in lns[i - 1]]
        upd = [i for i in range(start, end) if "past_key_values.update(key_states" in lns[i - 1]]
        dist = sorted(p.name for p in mp.parents[3].glob("transformers-*.dist-info"))
        rec["transformers_qwen3_5"] = {
            "sha256": sha(mp), "dist_info": dist,
            "Qwen3_5Attention_lines": [start, end - 1],
            "apply_rotary_pos_emb_call_lines": rope, "past_key_values_update_lines": upd,
            "rope_applied_before_cache_update": bool(rope and upd and max(rope) < min(upd)),
            "rotates_only_first_cos_dims": any("rotary_dim = cos.shape[-1]" in ln for ln in lns),
            "linear_attention_class": [i for i, ln in enumerate(lns, 1) if ln.startswith("class Qwen3_5GatedDeltaNet")]}
    return rec


def f7_f9_prompt_len(acc, ctl):
    by = defaultdict(list)
    for t in acc:
        by[t["condition"]].append(t["prompt_len"])
    ch = [t["prompt_len"] for t in acc if t["condition"] in ("cheerful", "hostile")]
    probe_topic = {t["probe"]: t["topic"] for t in acc}
    seq = [{"condition": t["condition"], "topic": probe_topic.get(t["probe"]), "prompt_len": t["prompt_len"]}
           for t in ctl if t["condition"] in ("persona_cheerful", "persona_hostile") and t["rep"] == 0]
    return {"f7_initial_study_prompt_len_min_max_by_arm": {k: [min(v), max(v)] for k, v in sorted(by.items())},
            "f7_cheerful_and_hostile_min_max": [min(ch), max(ch)],
            "f9_persona_prompt_len_rep0_file_order": seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--transformers-modeling", default=None)
    a = ap.parse_args()
    paper = Path(a.paper_dir)
    repo = paper.parent
    out_path = Path(a.out) if a.out else paper / "data" / "r5_review_checks_results.json"
    load = lambda n: json.loads((paper / "data" / n).read_text(encoding="utf-8"))["trials"]
    dose, acc, ctl = load("dose_response_results.json"), load("accumulation_results.json"), load("accumulation_controls.json")
    leak = json.loads((paper / "data" / "r4_calibration_leak_check_results.json").read_text(encoding="utf-8"))
    inputs = {f"emotion-accumulation-paper/data/{n}": sha(paper / "data" / n) for n in
              ["dose_response_results.json", "accumulation_results.json", "accumulation_controls.json",
               "r4_calibration_leak_check_results.json"]}
    inputs.update({f"emotion-accumulation-paper/code/{n}": sha(paper / "code" / n) for n in
                   ["dose_response_accumulation.py", "emotional_dynamics.py"]})
    for rel in ["decision-state-paper/code/decision_moment.py", "user-model-paper/correction_v2/emotion_bridge_trials.json"]:
        inputs[rel] = sha(repo / rel)
    res = {
        "script": "emotion-accumulation-paper/code/r5_review_checks.py",
        "review": "notes/agni_paper_downstream_um_cites_r5_ea.json (repo root)",
        "inputs_sha256_lf_form": inputs,
        "rules": {"assistant_turn": "odd index of trials[].conversation (user, assistant, ..., probe)",
                  "planning_prefixes_case_sensitive": list(PLAN_PREFIXES),
                  "restated_user_text": "every verbatim run of >= 20 chars found anywhere in any earlier user message of the same conversation; removed before the tone search (every stored user message is shorter than Experiment 1's 100-char storage cut, so none is truncated; see user_message_chars_max)",
                  "neg_tone_words_whole_word_case_insensitive": NEG, "pos_tone_words_whole_word_case_insensitive": POS,
                  "lower_bound": "Experiment 1 stores only the first 100 characters of each message"},
        "f1_generation_settings": gen_settings(paper / "code"),
        "f1_experiment1_turns": f1_turns(dose, "dose"),
        "f1_initial_study_turns": f1_turns([t for t in acc if t["n_history_turns"] > 0], "n_history_turns"),
        "f2_experiment1_read_sites": f2_read_sites(dose),
        "f3_directions": f3_directions(repo, leak),
        "f5_read_layers": f5_layers(paper, repo, a.model_config, a.transformers_modeling),
    }
    res.update(f7_f9_prompt_len(acc, ctl))
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out_path.name)


if __name__ == "__main__":
    main()
