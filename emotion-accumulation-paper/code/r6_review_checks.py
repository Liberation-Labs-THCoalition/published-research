"""Checks of the Agni r6 review of this paper (notes/agni_paper_downstream_um_cites_r6_ea.json),
recomputed from shipped files. Read-only; writes one result JSON. No model is run.

    python r6_review_checks.py --tokenizer <dir> [--paper-dir <emotion-accumulation-paper>] [--out <json>]

--tokenizer is a local snapshot folder of the BASE model Qwen/Qwen3.5-27B from the Hugging Face hub
(it must hold tokenizer.json and tokenizer_config.json). The distilled model's own
tokenizer_config.json is not in this repository, so every number that depends on the chat template
is a proxy computed with the base template; the result records the LF-form sha256 of the two
tokenizer files, never their path. The proxy is checked by exact token arithmetic: see
f1_initial_render_vs_prompt_len, x_initial_prompt_len_minus_render_gen_think_open,
f1_exp1_deficit_vs_all_60_hist (55 of 72 at exactly 0) and f7_f8_exp2.

Default paper dir is the parent of this script's folder (the repo root is its parent); default
output is data/r6_review_checks_results.json beside the data it reads. Inputs are hashed in their
LF form, so a checkout with CRLF line endings gives the same record. Paths are never written.

Findings covered (numbering is the review's):
  1  the chat template keeps, of each earlier assistant turn, only what follows a closing think
     tag. Initial study (accumulation_results.json): which stored turns carry the tag, their
     prompt_len against the rendered length with and without stripping, and the tone counts over
     the encoded text. Experiment 1 (dose_response_results.json): each conversation's seq_len
     against the empty-assistant render plus 60 tokens per generated turn (its cap); a
     conversation is "short" when it falls below that length (every non-zero deficit is >= 34
     tokens, so the SHORT_MIN rule of < 10 used for the exclusion sets selects the same 17);
     pooled and per-topic gaps with and without the short ones.
  2  pooled SD and d by dose (f2_exp1_all).
  3  what chunk 0 holds at each dose (f3_chunk0, x_chunk0_dose1).
  4  the calibration selection of r4_calibration_leak_check.py, matched against both the pre-set
     LEAK pattern and the post-data BROAD pattern of user-model-paper/correction_v2/leak_split.py.
  7  Experiment 2 (emotional_dynamics_results.json): rendered turn-1 length and the assistant
     tokens implied by each step of seq_len.
  8  Experiment 2's read site: turn-1 render with the generation prompt, thinking on and off.
Stored model text is written only as the first five words of an encoded reply (finding 1).
"""
import argparse, ast, hashlib, importlib.util, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy import stats as _st

V = "valence_proj"
SHORT_MIN = 10
CAP = 60  # Experiment 1's max_new_tokens per generated turn (dose_response_accumulation.py)


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def r4(x):
    return None if x is None else round(float(x), 4)


class Tok:
    def __init__(self, d):
        d = Path(d)
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.template = cfg["chat_template"]
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(d / "tokenizer.json"))
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise RuntimeError(msg)
        env.globals["raise_exception"] = raise_exception
        self.tpl = env.from_string(self.template)
        self.hashes = {f: sha_lf(d / f) for f in ("tokenizer.json", "tokenizer_config.json")}

    def render(self, msgs, gen, think_off=True):
        kw = {"messages": msgs, "add_generation_prompt": gen}
        if think_off:
            kw["enable_thinking"] = False
        return self.tpl.render(**kw)

    def ids(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def n(self, text):
        return len(self.ids(text))


def msgs_of(conv, nostrip=False):
    out = []
    for i, c in enumerate(conv):
        m = {"role": "user" if i % 2 == 0 else "assistant", "content": c}
        if nostrip and i % 2 == 1:
            m["reasoning_content"] = ""  # the template then keeps content whole (no split on </think>)
        out.append(m)
    return out


def encoded_text(a):
    """What the base template keeps of an earlier assistant turn."""
    a = a.strip()
    if "</think>" in a:
        a = a.split("</think>")[-1].lstrip("\n")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = Path(a.paper_dir)
    repo = P.parent
    out_path = Path(a.out) if a.out else P / "data" / "r6_review_checks_results.json"
    inputs = {
        "emotion-accumulation-paper/data/accumulation_results.json": P / "data/accumulation_results.json",
        "emotion-accumulation-paper/data/dose_response_results.json": P / "data/dose_response_results.json",
        "emotion-accumulation-paper/data/emotional_dynamics_results.json": P / "data/emotional_dynamics_results.json",
        "emotion-accumulation-paper/code/r5_review_checks.py": P / "code/r5_review_checks.py",
        "emotion-accumulation-paper/code/r4_review_checks.py": P / "code/r4_review_checks.py",
        "emotion-accumulation-paper/code/r4_calibration_leak_check.py": P / "code/r4_calibration_leak_check.py",
        "emotion-accumulation-paper/code/emotional_dynamics.py": P / "code/emotional_dynamics.py",
        "user-model-paper/correction_v2/emotion_bridge_trials.json": repo / "user-model-paper/correction_v2/emotion_bridge_trials.json",
        "user-model-paper/correction_v2/leak_split.py": repo / "user-model-paper/correction_v2/leak_split.py",
        "decision-state-paper/code/decision_moment.py": repo / "decision-state-paper/code/decision_moment.py",
    }
    res = {"script": "emotion-accumulation-paper/code/r6_review_checks.py",
           "review": "notes/agni_paper_downstream_um_cites_r6_ea.json (repo root)",
           "inputs_sha256_lf_form": {k: sha_lf(v) for k, v in inputs.items()}}
    tok = Tok(a.tokenizer)
    res["tokenizer"] = {"source": "Qwen/Qwen3.5-27B base snapshot (proxy; the distill's template is not available)",
                        "files_sha256_lf_form": tok.hashes}
    r5m = load_mod("r5chk", P / "code/r5_review_checks.py")
    r4m = load_mod("r4chk", P / "code/r4_review_checks.py")
    lkm = load_mod("r4leak", P / "code/r4_calibration_leak_check.py")

    acc = json.loads((P / "data/accumulation_results.json").read_text(encoding="utf-8"))
    dose = json.loads((P / "data/dose_response_results.json").read_text(encoding="utf-8"))
    dyn = json.loads((P / "data/emotional_dynamics_results.json").read_text(encoding="utf-8"))

    # ---------- 1: initial study, rendered length vs prompt_len ----------
    match = Counter()
    rows = []
    for t in acc["trials"]:
        conv = t["conversation"]
        n_s = {g: tok.n(tok.render(msgs_of(conv), g)) for g in (False, True)}
        n_ns = {g: tok.n(tok.render(msgs_of(conv, True), g)) for g in (False, True)}
        has = any("</think>" in m for m in conv[1::2][: len(conv) // 2])
        for g in (False, True):
            match[f"gen={g}|strip_render_equals_prompt_len"] += n_s[g] == t["prompt_len"]
            match[f"gen={g}|nostrip_render_equals_prompt_len"] += n_ns[g] == t["prompt_len"]
        rows.append(dict(t=t, n_s=n_s, n_ns=n_ns, has=has))
    res["f1_initial_render_vs_prompt_len"] = {"n_trials": len(rows), "exact_match_counts": dict(match)}
    cellpl = defaultdict(list)
    for r in rows:
        cellpl[(r["t"]["topic"], r["t"]["condition"])].append(r["t"]["prompt_len"])
    think = []
    for r in rows:
        if r["has"]:
            t = r["t"]
            ai = next(i for i in range(1, len(t["conversation"]), 2) if "</think>" in t["conversation"][i])
            txt = t["conversation"][ai]
            pre, post = txt.split("</think>", 1)
            think.append({"topic": t["topic"], "condition": t["condition"], "rep": t["rep"], "idx": t["idx"],
                          "assistant_turn_index_in_conversation": ai,
                          "prompt_len": t["prompt_len"], "cell_prompt_lens": cellpl[(t["topic"], t["condition"])],
                          "render_strip_gen_false": r["n_s"][False], "render_strip_gen_true": r["n_s"][True],
                          "render_nostrip_gen_false": r["n_ns"][False], "render_nostrip_gen_true": r["n_ns"][True],
                          "tokens_before_think_close_incl_tag": tok.n(pre + "</think>"),
                          "chars_before_think_close": len(pre), "chars_after_think_close": len(post),
                          "post_first_five_words": " ".join(post.split()[:5])})
    res["f1_initial_think_trials"] = think
    short_init = [{"topic": tp, "condition": c, "prompt_lens": v} for (tp, c), v in sorted(cellpl.items()) if len(set(v)) > 1]
    res["f1_initial_cells_with_unequal_prompt_len"] = short_init

    # ---------- 1: initial study tone recount on encoded text ----------
    def count_turns(trials, use_encoded):
        by = defaultdict(Counter)
        for t in trials:
            conv = t["conversation"]
            for i in range(1, len(conv) - 1, 2):
                txt = encoded_text(conv[i]) if use_encoded else conv[i]
                rest, _ = r5m.strip_restated(txt, [conv[k] for k in range(0, i, 2)])
                c = by[t["condition"]]
                c["turns"] += 1
                c["planning_prefix"] += txt.lstrip().startswith(r5m.PLAN_PREFIXES)
                c["neg_tone_word_own_words"] += bool(r5m.NEG_RE.search(rest))
                c["pos_tone_word_own_words"] += bool(r5m.POS_RE.search(rest))
        return {k: dict(v) for k, v in sorted(by.items())}
    hist = [t for t in acc["trials"] if len(t["conversation"]) > 1]
    res["f1_initial_tone_counts"] = {"generated_text": count_turns(hist, False),
                                     "encoded_text_base_template": count_turns(hist, True)}

    # ---------- 1: Experiment 1 rendered lengths ----------
    tr = dose["trials"]
    cell = defaultdict(list)
    for t in tr:
        cell[(t["topic"], t["dose"], t["condition"])].append(t)
    e1 = []
    for t in tr:
        conv = t["conversation"]
        n_stored = tok.n(tok.render(msgs_of(conv), False))
        empty = [c if i % 2 == 0 else "" for i, c in enumerate(conv)]
        n_frame = tok.n(tok.render(msgs_of(empty), False))
        k = len(conv) // 2
        full = n_frame + CAP * k
        cmax = max(x["seq_len"] for x in cell[(t["topic"], t["dose"], t["condition"])])
        e1.append({"idx": t["idx"], "topic": t["topic"], "dose": t["dose"], "condition": t["condition"], "rep": t["rep"],
                   "seq_len": t["seq_len"], "render_stored_text": n_stored, "render_empty_assistant": n_frame,
                   "assistant_tokens_encoded": t["seq_len"] - n_frame, "assistant_tokens_if_all_turns_60": CAP * k,
                   "deficit_vs_all_60": full - t["seq_len"], "deficit_vs_cell_max": cmax - t["seq_len"],
                   "stored_text_longer_than_encoded": n_stored > t["seq_len"], V: t["user_mean_proj"][V],
                   "chunk0_n_tokens": t["turn_summaries"][0]["n_tokens"]})
    res["f1_exp1_deficit_vs_all_60_hist"] = dict(sorted(Counter(r["deficit_vs_all_60"] for r in e1).items()))
    res["f1_exp1_deficit_vs_cell_max_hist"] = dict(sorted(Counter(r["deficit_vs_cell_max"] for r in e1).items()))
    res["f1_exp1_stored_text_longer_than_encoded"] = [
        {k: r[k] for k in ("idx", "topic", "dose", "condition", "rep", "seq_len", "render_stored_text", "render_empty_assistant")}
        for r in e1 if r["stored_text_longer_than_encoded"]]
    res["f1_exp1_trials"] = [{k: (r4(v) if k == V else v) for k, v in r.items()} for r in e1]

    # ---------- 1 / 2: gaps and SDs, all and excluding short ----------
    def stats_for(sel):
        out = {}
        for d in dose["dose_levels"]:
            c = [r[V] for r in sel if r["dose"] == d and r["condition"] == "cheerful"]
            h = [r[V] for r in sel if r["dose"] == d and r["condition"] == "hostile"]
            sc, sh = np.std(c, ddof=1), np.std(h, ddof=1)
            sp = np.sqrt(((len(c) - 1) * sc ** 2 + (len(h) - 1) * sh ** 2) / (len(c) + len(h) - 2))
            g = np.mean(c) - np.mean(h)
            pt = {}
            for tp in ("garden", "cooking", "career"):
                cc = [r[V] for r in sel if r["dose"] == d and r["condition"] == "cheerful" and r["topic"] == tp]
                hh = [r[V] for r in sel if r["dose"] == d and r["condition"] == "hostile" and r["topic"] == tp]
                pt[tp] = {"gap": r4(np.mean(cc) - np.mean(hh)) if cc and hh else None, "n_c": len(cc), "n_h": len(hh),
                          "sd_c": r4(np.std(cc, ddof=1)) if len(cc) > 1 else None,
                          "sd_h": r4(np.std(hh, ddof=1)) if len(hh) > 1 else None}
            tt = _st.ttest_ind(h, c, equal_var=True)
            out[str(d)] = {"n_c": len(c), "n_h": len(h), "mean_c": r4(np.mean(c)), "mean_h": r4(np.mean(h)),
                           "d_table_sign_hostile_minus_cheerful": r4(-g / sp), "p_student_t_reruns_as_independent": r4(tt.pvalue),
                           "gap_pooled_trials": r4(g),
                           "mean_of_topic_gaps": r4(np.mean([v["gap"] for v in pt.values() if v["gap"] is not None])),
                           "pooled_sd": r4(sp), "d_gap_over_pooled_sd": r4(g / sp), "per_topic": pt}
        return out
    res["f2_exp1_all"] = stats_for(e1)
    sh_abs = [r for r in e1 if r["deficit_vs_all_60"] < SHORT_MIN]
    sh_cell = [r for r in e1 if r["deficit_vs_cell_max"] < SHORT_MIN]
    res["f1_exp1_short_rule"] = {"short_min_tokens": SHORT_MIN,
                                 "n_short_vs_all_60": len(e1) - len(sh_abs), "n_short_vs_cell_max": len(e1) - len(sh_cell)}
    res["f1_exp1_excl_short_vs_all_60"] = stats_for(sh_abs)
    res["f1_exp1_excl_short_vs_cell_max"] = stats_for(sh_cell)
    res["f1_exp1_excl_stored_longer_only"] = stats_for([r for r in e1 if not r["stored_text_longer_than_encoded"]])

    # ---------- 3: chunk-0 composition ----------
    hdr = tok.n("<|im_start|>assistant\n")
    c0 = defaultdict(list)
    for t in tr:
        conv = t["conversation"]
        u1 = tok.n(tok.render([{"role": "user", "content": conv[0]}], False))
        cs = t["seq_len"] // len(conv)
        c0[t["dose"]].append({"chunk": cs, "user1_rendered": u1, "assistant_content_tokens_in_chunk0": cs - u1 - hdr,
                              "matches_turn_summary": cs == t["turn_summaries"][0]["n_tokens"]})
    res["f3_chunk0"] = {"assistant_header_tokens": hdr, "by_dose": {
        str(d): {"chunk_min_max": [min(x["chunk"] for x in v), max(x["chunk"] for x in v)],
                 "user1_rendered_min_max": [min(x["user1_rendered"] for x in v), max(x["user1_rendered"] for x in v)],
                 "assistant_tokens_in_chunk0_min_median_max": [min(x["assistant_content_tokens_in_chunk0"] for x in v),
                     float(np.median([x["assistant_content_tokens_in_chunk0"] for x in v])),
                     max(x["assistant_content_tokens_in_chunk0"] for x in v)],
                 "all_match_turn_summary": all(x["matches_turn_summary"] for x in v)} for d, v in sorted(c0.items())}}

    # ---------- 7 / 8: Experiment 2 ----------
    pats = r4m.scripted_patterns(P)[0]
    f7 = {}
    for t in dyn["trials"]:
        spec = pats.get(t["pattern"], {})
        msgs = next((v for v in spec.values() if isinstance(v, list)), None)
        sl = [p["seq_len"] for p in t["trajectory"]]
        if t["pattern"] not in f7:
            u_r = [tok.n(tok.render([{"role": "user", "content": m["text"]}], False)) for m in msgs]
            built = [{"role": "user", "content": msgs[0]["text"]}]
            f7[t["pattern"]] = {"turn1_render_gen_think_off": tok.n(tok.render(built, True, True)),
                                "turn1_render_gen_think_on": tok.n(tok.render(built, True, False)),
                                "turn1_render_no_gen_plus_header": tok.n(tok.render(built, False)) + hdr,
                                "turn1_last_token_think_off": tok.tk.decode([tok.ids(tok.render(built, True, True))[-1]], skip_special_tokens=False),
                                "user_rendered_tokens": u_r, "reps": {}}
        u_r = f7[t["pattern"]]["user_rendered_tokens"]
        steps = [sl[i + 1] - sl[i] for i in range(len(sl) - 1)]
        f7[t["pattern"]]["reps"][str(t["rep"])] = {"seq_len": sl, "steps": steps,
            "implied_assistant_content_tokens": [steps[i] - u_r[i + 1] - 5 for i in range(len(steps))]}
    res["f7_f8_exp2"] = f7

    # ---------- 4: broader leak pattern over the calibration selection ----------
    src = repo / "user-model-paper/correction_v2/emotion_bridge_trials.json"
    stories = json.loads(src.read_text(encoding="utf-8"))["trials"]
    tree = ast.parse((repo / "user-model-paper/correction_v2/leak_split.py").read_text(encoding="utf-8"))
    ns = {"re": re}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("LEAK", "BROAD"):
            exec(compile(ast.Module([node], []), "leak_split.py", "exec"), ns)
    by_e = defaultdict(list)
    for i, s in enumerate(stories):
        by_e[s["emotion"]].append(i)
    selected = [i for v in by_e.values() for i in v[:lkm.N_PER_EMOTION]]
    sets = lkm.cluster_sets(repo / "decision-state-paper/code/decision_moment.py")
    dirs = {"valence": ("POSITIVE_EMOTIONS", "NEG_VALENCE_EMOTIONS"),
            "uncertainty (anxious-calm)": ("UNCERTAINTY_EMOTIONS", "CALM_EMOTIONS"),
            "reward": ("REWARD_EMOTIONS", "NEGATIVE_EMOTIONS")}
    feeds = lambda e: [n for n, (p, q) in dirs.items() if e in sets.get(p, set()) or e in sets.get(q, set())]
    used = [i for i in selected if feeds(stories[i]["emotion"])]
    f4 = {"n_used": len(used), "per_direction": {}}
    for n in dirs:
        ids = [i for i in used if n in feeds(stories[i]["emotion"])]
        f4["per_direction"][n] = {"n": len(ids),
            "leak_preset": sum(bool(ns["LEAK"].search(stories[i]["story_text"])) for i in ids),
            "broad_only": sum(bool(ns["BROAD"].search(stories[i]["story_text"])) and not ns["LEAK"].search(stories[i]["story_text"]) for i in ids),
            "leak_or_broad": sum(bool(ns["LEAK"].search(stories[i]["story_text"]) or ns["BROAD"].search(stories[i]["story_text"])) for i in ids)}
    f4["leak_or_broad_matches"] = [{"record_index_0based": i, "emotion": stories[i]["emotion"],
                                    "by": "preset" if ns["LEAK"].search(stories[i]["story_text"]) else "broad",
                                    "feeds": feeds(stories[i]["emotion"])} for i in used
                                   if ns["LEAK"].search(stories[i]["story_text"]) or ns["BROAD"].search(stories[i]["story_text"])]
    f4["whole_file"] = {"n": len(stories), "leak_preset": sum(bool(ns["LEAK"].search(s["story_text"])) for s in stories),
                        "leak_or_broad": sum(bool(ns["LEAK"].search(s["story_text"]) or ns["BROAD"].search(s["story_text"])) for s in stories)}
    res["f4_calibration_broad"] = f4

    # ---------- extra: initial-study offsets, travel sensitivity, Exp 1 tone counts, chunk 0 ----------
    offs = defaultdict(Counter)
    for r in rows:
        t = r["t"]
        msgs = msgs_of(t["conversation"])
        n_on = tok.n(tok.render(msgs, True, False))
        offs["history" if len(t["conversation"]) > 1 else "no_history"][t["prompt_len"] - n_on] += 1
    res["x_initial_prompt_len_minus_render_gen_think_open"] = {k: dict(sorted(v.items())) for k, v in offs.items()}
    stripped_idx = {d["idx"] for d in think}
    init = {}
    for tp in sorted({t["topic"] for t in acc["trials"]}):
        def vals(c, excl):
            return [t["probe_projections"][V] for t in acc["trials"] if t["topic"] == tp and t["condition"] == c
                    and not (excl and t["idx"] in stripped_idx)]
        row = {}
        for excl in (False, True):
            c, h = vals("cheerful", excl), vals("hostile", excl)
            row["excl_stripped" if excl else "all"] = {"gap": r4(np.mean(c) - np.mean(h)), "n_c": len(c), "n_h": len(h),
                                                       "sd_c": r4(np.std(c, ddof=1)), "sd_h": r4(np.std(h, ddof=1))}
        init[tp] = row
    res["x_initial_topic_gaps_excl_stripped"] = init
    short_ids = {r["idx"] for r in e1 if r["deficit_vs_all_60"] > 0}

    def e1_tone(excl):
        by = defaultdict(Counter)
        for t in tr:
            if excl and t["idx"] in short_ids:
                continue
            conv = t["conversation"]
            for i in range(1, len(conv) - 1, 2):
                rest, _ = r5m.strip_restated(conv[i], [conv[k] for k in range(0, i, 2)])
                c = by[t["condition"]]
                c["turns"] += 1
                c["neg"] += bool(r5m.NEG_RE.search(rest))
                c["pos"] += bool(r5m.POS_RE.search(rest))
                c["plan"] += conv[i].lstrip().startswith(r5m.PLAN_PREFIXES)
        return {k: dict(v) for k, v in sorted(by.items())}
    res["x_exp1_tone_counts"] = {"all": e1_tone(False), "excl_short": e1_tone(True), "n_short": len(short_ids),
                                 "short_idx": sorted(short_ids),
                                 "short_deficit_min_max": [min(r["deficit_vs_all_60"] for r in e1 if r["idx"] in short_ids),
                                                           max(r["deficit_vs_all_60"] for r in e1 if r["idx"] in short_ids)]}
    c0d1 = []
    for t in tr:
        if t["dose"] != 1:
            continue
        conv = t["conversation"]
        u1 = tok.n(tok.render([{"role": "user", "content": conv[0]}], False))
        cs = t["seq_len"] // len(conv)
        c0d1.append((t["idx"], t["condition"], cs - u1 - hdr, t["turn_summaries"][0][V]))
    full = [x for x in c0d1 if x[0] not in short_ids]
    gap0 = lambda xs: r4(np.mean([x[3] for x in xs if x[1] == "cheerful"]) - np.mean([x[3] for x in xs if x[1] == "hostile"]))
    res["x_chunk0_dose1"] = {"assistant_tokens_in_chunk0_full_min_max": [min(x[2] for x in full), max(x[2] for x in full)],
                             "n_full": len(full), "gap_all": gap0(c0d1), "gap_excl_short": gap0(full)}

    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out_path.name)


if __name__ == "__main__":
    main()
