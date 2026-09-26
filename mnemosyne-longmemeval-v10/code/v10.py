#!/usr/bin/env python3
"""Mnemosyne v10 retrieval and its evidence-recall evaluation on LongMemEval_S-cleaned.

See DESIGN.md. Retrieval only ever sees view(item). Evidence labels are read only by evidence(),
which is used for scoring, never for retrieval.

  v10.py eval  --split dev  --config CFG.json            # dev: free to iterate
  v10.py eval  --split held --config CFG.json            # held: CFG must be read-only; logged
  v10.py sweep --split dev  --grid GRID.json             # dev only
"""
import argparse, collections, functools, hashlib, itertools, json, math, os, re, sys, time
import datetime as dt
import numpy as np
import tiktoken
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = "/mnt/data1/lme_v2/v10"
DATA = "/mnt/data1/datasets/longmemeval/longmemeval_s.json"
V9_AUTOPSY = "/mnt/data1/lme_v2/review/evidence_recall.json"
ENC = tiktoken.get_encoding("o200k_base")
SESSION_HEADER_TOKENS = 16  # "### Session N:\nSession Date: ...\nSession Content:\n" in the official builder

DEFAULT = {
    "channels": ["bm25", "dense"],  # any subset
    "rrf_k": 60,
    "w_bm25": 1.0, "w_dense": 1.0,
    "w_assistant": 1.0,             # multiplier on assistant-turn fused scores
    "second_weight": 0.5,           # session score = best turn + second_weight * second-best
    "budget": 16000,                # o200k tokens of rendered history
    "whole_top": 99,                # top-N sessions go in whole (if they fit); the rest as windows
    "window": 2,                    # +/- turns around a session's best turns when not whole
    "window_seeds": 2,              # how many best turns seed windows
    "bm25_k1": 1.2, "bm25_b": 0.75,
    "w_time": 1.0,                  # session-level boost for sessions dated inside a parsed time window
}

# ---------------------------------------------------------------- inputs

def turn_key(role, content):
    return hashlib.sha1((role + "\x00" + content).encode()).hexdigest()

def view(item):
    """The ONLY input retrieval gets. Built field by field; no ids, no labels, no type."""
    v = {"question": str(item["question"]), "question_date": str(item["question_date"]),
         "sessions": [{"date": str(d),
                       "turns": [{"role": str(t["role"]), "content": str(t["content"])} for t in s]}
                      for d, s in zip(item["haystack_dates"], item["haystack_sessions"])]}
    assert set(v) == {"question", "question_date", "sessions"}
    assert all(set(s) == {"date", "turns"} and all(set(t) == {"role", "content"} for t in s["turns"])
               for s in v["sessions"])
    return v

def evidence(item):
    """Scoring only: {session_index: [turn indexes with has_answer]}."""
    ev = {}
    for si, s in enumerate(item["haystack_sessions"]):
        ts = [ti for ti, t in enumerate(s) if t.get("has_answer")]
        if ts:
            ev[si] = ts
    return ev

# ---------------------------------------------------------------- per-turn caches

_stem = functools.lru_cache(maxsize=None)(PorterStemmer().stem)
_TOK = re.compile(r"[a-z0-9]+")
_tf_cache, _len_cache = {}, {}

def terms(text):
    return [_stem(w) for w in _TOK.findall(text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS]

def turn_tf(key, content):
    if key not in _tf_cache:
        _tf_cache[key] = collections.Counter(terms(content))
    return _tf_cache[key]

def turn_tokens(key, turn):
    if key not in _len_cache:
        # disallowed_special=(): a turn can contain the literal text '<|endoftext|>' (one held-out
        # haystack does); count it as the ordinary text it is, instead of raising.
        _len_cache[key] = len(ENC.encode(json.dumps(turn, ensure_ascii=False), disallowed_special=())) + 1
    return _len_cache[key]

class Dense:
    def __init__(self):
        meta = json.load(open(f"{ROOT}/emb/meta.json"))
        self.meta = meta
        keys = json.load(open(f"{ROOT}/emb/turn_index.json"))
        self.row = {k: i for i, k in enumerate(keys)}
        self.turns = np.load(f"{ROOT}/emb/turns.f16.npy", mmap_mode="r")
        self.questions = np.load(f"{ROOT}/emb/questions.f16.npy").astype(np.float32)

    def scores(self, qidx, keys):
        """Cosine per turn; NaN for turns without an embedding (assistant turns: see embed_turns.py)."""
        out = np.full(len(keys), np.nan, dtype=np.float32)
        have = [i for i, k in enumerate(keys) if k in self.row]
        if have:
            m = np.asarray(self.turns[[self.row[keys[i]] for i in have]], dtype=np.float32)
            out[have] = m @ self.questions[qidx]
        return out

# ---------------------------------------------------------------- retrieval

def bm25(q_terms, docs, k1, b):
    n = len(docs)
    avgdl = sum(sum(d.values()) for d in docs) / max(n, 1) or 1.0
    df = collections.Counter(t for d in docs for t in d)
    qs = set(q_terms)
    out = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(docs):
        dl = sum(d.values())
        s = 0.0
        for t in qs:
            f = d.get(t, 0)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out[i] = s
    return out

# ---------------------------------------------------------------- the time channel
# Relative-time phrases in the question become date windows around the question date. General
# English patterns, written once; nothing here looks at question types or at any dev question.

_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "couple of": 2,
         "a few": 3, "few": 3, "several": 4}
_NUMRE = r"(\d+|a couple of|couple of|a few|few|several|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT = {"day": 1.0, "week": 7.0, "month": 30.44, "year": 365.25}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]

def parse_date(s):
    return dt.datetime.strptime(s[:10], "%Y/%m/%d")

def time_windows(question, question_date):
    q, qd, D = question.lower(), parse_date(question_date), lambda x: dt.timedelta(days=x)
    num = lambda w: int(w) if w.isdigit() else _NUMW[w]
    wins = []
    for m in re.finditer(_NUMRE + r"\s+(day|week|month|year)s?\s+ago\b", q):
        u = _UNIT[m.group(2)]; c = qd - D(num(m.group(1)) * u); tol = max(1.0, 0.5 * u)
        wins.append((c - D(tol), c + D(tol)))
    for m in re.finditer(r"\b(?:past|last)\s+" + _NUMRE + r"\s+(day|week|month|year)s\b", q):
        wins.append((qd - D(num(m.group(1)) * _UNIT[m.group(2)] + 1), qd))
    for m in re.finditer(r"\blast (week|weekend|month|year)\b", q):
        span = {"week": 14, "weekend": 9, "month": 62, "year": 730}[m.group(1)]
        wins.append((qd - D(span), qd))
    if re.search(r"\byesterday\b", q):
        wins.append((qd - D(2), qd))
    for m in re.finditer(r"\b(?:last|this past) (" + "|".join(_WEEKDAYS) + r")\b", q):
        back = (qd.weekday() - _WEEKDAYS.index(m.group(1))) % 7 or 7
        d = qd - D(back); wins.append((d - D(1), d + D(1)))
    for m in re.finditer(r"\bthis (week|month)\b", q):
        wins.append((qd - D(7 if m.group(1) == "week" else 31), qd))
    for m in re.finditer(r"\b(?:in|during|since) (" + "|".join(_MONTHS) + r")\b", q):
        mi = _MONTHS.index(m.group(1)) + 1
        y = qd.year if mi <= qd.month else qd.year - 1
        start = dt.datetime(y, mi, 1)
        end = dt.datetime(y + (mi == 12), mi % 12 + 1, 1) - D(1)
        wins.append((start, qd if "since" in m.group(0) else end))
    return wins

def rrf(scores, k, positive_only, tiebreak):
    """Reciprocal-rank weights. NaN scores (no signal for that turn) get no rank and no weight."""
    valid = ~np.isnan(scores)
    filled = np.where(valid, scores, -np.inf)
    order = np.lexsort((tiebreak, -filled))  # ties broken by content hash, never by haystack position
    r = np.zeros(len(scores), dtype=np.float32)
    rank = 0
    for i in order:
        if not valid[i] or (positive_only and scores[i] <= 0):
            break
        r[i] = 1.0 / (k + rank + 1)
        rank += 1
    return r

def retrieve(v, qidx, cfg, dense=None):
    """Returns (chosen {session_idx: None|[turn idxs]}, tokens_used, diagnostics)."""
    flat = [(si, ti, t) for si, s in enumerate(v["sessions"]) for ti, t in enumerate(s["turns"])]
    keys = [turn_key(t["role"], t["content"]) for _, _, t in flat]
    tb = np.argsort(np.argsort(np.array(keys)))  # rank of each turn's content hash
    fused = np.zeros(len(flat), dtype=np.float32)
    if "bm25" in cfg["channels"]:
        docs = [turn_tf(k, t["content"]) for k, (_, _, t) in zip(keys, flat)]
        s = bm25(terms(v["question"]), docs, cfg["bm25_k1"], cfg["bm25_b"])
        fused += cfg["w_bm25"] * rrf(s, cfg["rrf_k"], positive_only=True, tiebreak=tb)
    if "dense" in cfg["channels"]:
        fused += cfg["w_dense"] * rrf(dense.scores(qidx, keys), cfg["rrf_k"], positive_only=False, tiebreak=tb)
    if "random" in cfg["channels"]:  # control: the metric must be able to fail
        fused += np.random.default_rng(qidx).random(len(flat)).astype(np.float32)
    asst = np.array([t["role"] == "assistant" for _, _, t in flat])
    fused = np.where(asst, fused * cfg["w_assistant"], fused)

    per_sess = collections.defaultdict(list)
    for (si, ti, _), f in zip(flat, fused):
        per_sess[si].append((f, ti))
    sess_score = {}
    for si, lst in per_sess.items():
        top = sorted(lst, reverse=True)
        sess_score[si] = top[0][0] + cfg["second_weight"] * (top[1][0] if len(top) > 1 else 0.0)
    shash = {si: turn_key("session", json.dumps(v["sessions"][si], sort_keys=True)) for si in sess_score}
    wins = time_windows(v["question"], v["question_date"]) if "time" in cfg["channels"] else []
    if wins:
        dist = {}
        for si in sess_score:
            d = parse_date(v["sessions"][si]["date"])
            ds = [abs((d - (a + (b - a) / 2)).days) for a, b in wins if a <= d <= b]
            if ds:
                dist[si] = min(ds)
        for rank, si in enumerate(sorted(dist, key=lambda si: (dist[si], shash[si]))):
            sess_score[si] += cfg["w_time"] / (cfg["rrf_k"] + rank + 1)
    order = sorted(sess_score, key=lambda si: (-sess_score[si], shash[si]))

    cost = lambda si, tis: SESSION_HEADER_TOKENS + sum(
        turn_tokens(turn_key(v["sessions"][si]["turns"][ti]["role"], v["sessions"][si]["turns"][ti]["content"]),
                    v["sessions"][si]["turns"][ti]) for ti in tis)
    chosen, used = {}, 0
    for rank, si in enumerate(order):
        n = len(v["sessions"][si]["turns"])
        if rank < cfg["whole_top"]:
            c = cost(si, range(n))
            if used + c <= cfg["budget"]:
                chosen[si], used = None, used + c
                continue
        seeds = [ti for _, ti in sorted(per_sess[si], reverse=True)[:cfg["window_seeds"]]]
        tis = sorted({j for ti in seeds for j in range(max(0, ti - cfg["window"]), min(n, ti + cfg["window"] + 1))})
        c = cost(si, tis)
        if used + c <= cfg["budget"]:
            chosen[si], used = (None if len(tis) == n else tis), used + c
    return chosen, used, {"n_sessions": len(chosen), "whole": sum(x is None for x in chosen.values()),
                          "time_windows": len(wins)}

def render(v, chosen):
    """Chronological, dated, official-builder-like JSON history (used for the like-for-like matcher)."""
    parts = []
    for si in sorted(chosen, key=lambda si: (v["sessions"][si]["date"], si)):
        s = v["sessions"][si]
        turns = s["turns"] if chosen[si] is None else [s["turns"][ti] for ti in chosen[si]]
        parts.append(f"\n### Session:\nSession Date: {s['date']}\nSession Content:\n{json.dumps(turns, ensure_ascii=False)}\n")
    return "".join(parts)

# ---------------------------------------------------------------- the v9 autopsy's matcher, verbatim

norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()

def windows(text, w=50, step=80):
    t = norm(text)
    if len(t) <= w: return [t] if len(t) >= 20 else []
    return [t[i:i + w] for i in range(0, len(t) - w + 1, step)]

# ---------------------------------------------------------------- evaluation

def evaluate(data, idxs, cfg, dense, text_match=False):
    rows = []
    for qi in idxs:
        item = data[qi]
        v = view(item)
        chosen, used, diag = retrieve(v, qi, cfg, dense)
        ev = evidence(item)
        inc = lambda si, ti: si in chosen and (chosen[si] is None or ti in chosen[si])
        cov = {si: any(inc(si, ti) for ti in tis) for si, tis in ev.items()}
        row = dict(idx=qi, qid=item["question_id"], type=item["question_type"],
                   abstention=item["question_id"].endswith("_abs"), n_ev=len(ev),
                   covered=sum(cov.values()),
                   all_sessions=bool(ev) and all(cov.values()),
                   all_turns=bool(ev) and all(inc(si, ti) for si, tis in ev.items() for ti in tis),
                   tokens=used, **diag)
        if text_match:
            c = norm(render(v, chosen))
            tm = {}
            for si, tis in ev.items():
                hits = []
                for ti in tis:
                    ws = windows(item["haystack_sessions"][si][ti]["content"])
                    h = sum(w in c for w in ws)
                    hits.append(bool(ws) and h / len(ws) >= 0.5)
                tm[si] = any(hits)
            row["tm_all_sessions_strict"] = bool(ev) and all(tm.values())
        rows.append(row)
    return rows

def summarise(rows, v9=None):
    scored = [r for r in rows if not r["abstention"] and r["n_ev"] > 0]
    by = collections.defaultdict(list)
    for r in scored:
        by[r["type"]].append(r)
    out = {"n_scored": len(scored), "n_unscorable": len(rows) - len(scored),
           "all_sessions": np.mean([r["all_sessions"] for r in scored]),
           "all_turns": np.mean([r["all_turns"] for r in scored]),
           "median_tokens": float(np.median([r["tokens"] for r in rows])),
           "by_type": {t: (len(rs), float(np.mean([r["all_sessions"] for r in rs]))) for t, rs in sorted(by.items())}}
    if "tm_all_sessions_strict" in scored[0]:
        out["tm_all_sessions_strict"] = np.mean([r["tm_all_sessions_strict"] for r in scored])
    if v9 is not None:
        vs = [v9[r["qid"]] for r in scored if r["qid"] in v9]
        out["v9_all_sessions_strict"] = np.mean([x["covered_strict"] == x["n_ev_sessions"] and x["n_ev_sessions"] > 0 for x in vs])
        out["v9_all_sessions_lenient"] = np.mean([x["covered_lenient"] == x["n_ev_sessions"] and x["n_ev_sessions"] > 0 for x in vs])
        out["v9_by_type"] = {}
        for t in by:
            vt = [v9[r["qid"]] for r in by[t] if r["qid"] in v9]
            out["v9_by_type"][t] = float(np.mean([x["covered_strict"] == x["n_ev_sessions"] for x in vt])) if vt else None
    return out

def load_split(name):
    split = json.load(open(f"{ROOT}/split.json"))
    data = json.load(open(split["data_file"]))
    assert hashlib.sha256(open(split["data_file"], "rb").read()).hexdigest() == split["data_sha256"]
    idxs = split[name]
    assert [data[i]["question_id"] for i in idxs] == split[f"{name}_qids"]
    return data, idxs

def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "sweep"])
    ap.add_argument("--split", choices=["dev", "held"], required=True)
    ap.add_argument("--config")
    ap.add_argument("--grid")
    ap.add_argument("--text-match", action="store_true")
    a = ap.parse_args()

    if a.split == "held":
        assert a.cmd == "eval", "sweeps are dev-only"
        assert a.config and not os.access(a.config, os.W_OK), "held-out needs a FROZEN (read-only) config file"
    data, idxs = load_split(a.split)
    v9 = {r["qid"]: r for r in json.load(open(V9_AUTOPSY))}
    if a.split == "held":   # log the ATTEMPT before running: a crashed held-out run must leave a trace too
        with open(f"{ROOT}/heldout_log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y%m%dT%H%M%S"), "status": "started",
                                "config": os.path.abspath(a.config)}) + "\n")

    base = dict(DEFAULT)
    if a.config:
        base.update(json.load(open(a.config)))
    dense = Dense() if os.path.exists(f"{ROOT}/emb/meta.json") else None
    grid = [{}]
    if a.cmd == "sweep":
        g = json.load(open(a.grid))
        grid = [dict(zip(g, vals)) for vals in itertools.product(*g.values())]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    results = []
    for over in grid:
        cfg = {**base, **over}
        if "dense" in cfg["channels"] and dense is None:
            print("skip (no embeddings yet):", over); continue
        rows = evaluate(data, idxs, cfg, dense, text_match=a.text_match)
        s = summarise(rows, v9)
        results.append({"cfg": cfg, "hash": cfg_hash(cfg), "summary": s, "rows": rows})
        bt = " ".join(f"{t.split('-')[0][:5]}={p:.2f}" for t, (n, p) in s["by_type"].items())
        print(f"{cfg_hash(cfg)} {json.dumps(over)[:90]:90} all_sess={s['all_sessions']:.3f} all_turns={s['all_turns']:.3f} "
              f"tok={s['median_tokens']:.0f} | {bt}" + (f" | tm={s['tm_all_sessions_strict']:.3f}" if a.text_match else ""), flush=True)
    if results:
        s0 = results[0]["summary"]
        print(f"v9 on the same questions: all_sess strict={s0['v9_all_sessions_strict']:.3f} lenient={s0['v9_all_sessions_lenient']:.3f} | "
              + " ".join(f"{t.split('-')[0][:5]}={p:.2f}" for t, p in sorted(s0["v9_by_type"].items()) if p is not None))
    out = f"{ROOT}/runs/{stamp}_{a.cmd}_{a.split}.json"
    json.dump(results, open(out, "w"), default=float)
    if a.split == "held":
        with open(f"{ROOT}/heldout_log.jsonl", "a") as f:
            f.write(json.dumps({"time": stamp, "status": "completed", "config": os.path.abspath(a.config), "hash": results[0]["hash"],
                                "all_sessions": float(results[0]["summary"]["all_sessions"]), "out": out}) + "\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
