"""Decode the stored token ids (review round 5, findings 5 and 9).

CPU only, no model run. Needs the public tokenizer.json of Qwen/Qwen3.5-27B from the Hugging Face hub, which
is not shipped here; pass its path with --tokenizer (by default the script looks in the local Hugging Face
cache). Positive control: decoding each trial's stored ids must reproduce its stored generated_text.

Records: what ids 353, 561 and 38220 are; for the matched-entity generations that open with the fixed
template, what tokens 0-7 and 8-19 are and where "My reasoning:" ends; and, for the prompts whose reps the
regex labels split between HEDGED and CONFABULATED, the token each rep takes at the first branch.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/r5_decoded_token_ids.json
"""
import argparse
import glob
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
MB = "decision-state-paper/data/matched_burn_results.json"
DR = "decision-state-paper/data/decision_moment_results.json"
OUT = PAPER / "data" / "r5_decoded_token_ids.json"
TEMPLATE = "Let me think through this carefully"
IDS = [353, 561, 38220]


def lf_bytes(path):
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def firstdiff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None, help="path to Qwen/Qwen3.5-27B tokenizer.json")
    args = ap.parse_args()
    tpath = args.tokenizer
    if tpath is None:
        hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--Qwen--Qwen3.5-27B")
        cands = sorted(glob.glob(os.path.join(hub, "snapshots", "*", "tokenizer.json")))
        if not cands:
            raise SystemExit("tokenizer.json not found; pass --tokenizer")
        tpath = cands[0]
    tok = Tokenizer.from_file(tpath)
    mb = json.loads(lf_bytes(REPO / MB).decode("utf-8"))["trials"]
    dr = json.loads(lf_bytes(REPO / DR).decode("utf-8"))["trials"]

    def dec(ids):
        return tok.decode(ids)

    control = {}
    for name, ts in (("matched_burn", mb), ("decision_moment", dr)):
        control[name] = {"n": len(ts),
                         "decode_equals_generated_text": sum(dec(t["trajectory"]["tokens"]).strip() == t["generated_text"].strip() for t in ts),
                         "decode_prefix60_equals": sum(dec(t["trajectory"]["tokens"]).strip()[:60] == t["generated_text"].strip()[:60] for t in ts)}

    tmpl = [t for t in mb if t["generated_text"].startswith(TEMPLATE)]
    first8 = Counter(dec(t["trajectory"]["tokens"][:8]) for t in tmpl)
    restated = 0
    mr_end = Counter()
    for t in tmpl:
        ids = t["trajectory"]["tokens"]
        s = dec(ids[8:20])
        if "question/task" in s or t["entity"] in s:
            restated += 1
        for k in range(len(ids)):
            if "My reasoning:" in dec(ids[:k + 1]):
                mr_end[k] += 1
                break
        else:
            mr_end["absent"] += 1

    by = defaultdict(list)
    for t in dr:
        by[t["prompt"]].append(t)
    mixed = {}
    i_the = []
    for p, ts in by.items():
        labs = [t["behavior"] for t in ts]
        if "HEDGED" in labs and "CONFABULATED" in labs:
            toks = [t["trajectory"]["tokens"] for t in ts]
            k = min(firstdiff(a, b) for a, b in itertools.combinations(toks, 2))
            branch = {str(t["rep"]): {"regex": t["behavior"], "token": dec([t["trajectory"]["tokens"][k]])} for t in ts}
            cross = []
            for a in ts:
                for b in ts:
                    if a["behavior"] == "HEDGED" and b["behavior"] == "CONFABULATED":
                        kk = firstdiff(a["trajectory"]["tokens"], b["trajectory"]["tokens"])
                        cross.append({"first_diff_token": kk, "hedged": dec([a["trajectory"]["tokens"][kk]]),
                                      "confab": dec([b["trajectory"]["tokens"][kk]])})
            mixed[p] = {"first_branch_token": k, "branch_token_by_rep": branch, "cross_label_pairs": cross}
            if {c["hedged"].strip() for c in cross} | {c["confab"].strip() for c in cross} == {"I", "The"}:
                i_the.append(p)
    result = {
        "LABEL": "Review round 5, findings 5 and 9. Stored token ids decoded with the public Qwen/Qwen3.5-27B tokenizer.json (not shipped); no model run.",
        "tokenizer": {"source": "Qwen/Qwen3.5-27B tokenizer.json (Hugging Face hub)", "sha256_lf": hashlib.sha256(lf_bytes(tpath)).hexdigest()},
        "inputs": {MB: hashlib.sha256(lf_bytes(REPO / MB)).hexdigest(), DR: hashlib.sha256(lf_bytes(REPO / DR)).hexdigest()},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "positive_control": control,
        "ids": {str(i): dec([i]) for i in IDS},
        "matched_burn_template": {"n_open_with_template": len(tmpl), "tokens_0_7": dict(first8),
                                  "tokens_8_19_restate_question": restated,
                                  "my_reasoning_ends_at_token": {str(k): v for k, v in sorted(mr_end.items(), key=lambda x: str(x[0]))}},
        "decision_moment_trial0_tokens_8_19": dec(dr[0]["trajectory"]["tokens"][8:20]),
        "regex_mixed_prompts": mixed,
        "prompts_with_an_I_vs_The_branch": i_the,
        "n_cross_pairs_I_vs_The": sum(len(mixed[p]["cross_label_pairs"]) for p in i_the),
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    print("  control", control, "ids", result["ids"])
    print("  template", result["matched_burn_template"])
    print("  dm trial0 8-19", repr(result["decision_moment_trial0_tokens_8_19"]))
    for p, m in mixed.items():
        print("  ", p[:55], m["first_branch_token"], {r: (v["regex"], v["token"]) for r, v in m["branch_token_by_rep"].items()})
    print("  I/The prompts", [p[:40] for p in i_the], "pairs", result["n_cross_pairs_I_vs_The"])


if __name__ == "__main__":
    main()
