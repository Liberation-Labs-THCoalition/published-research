#!/usr/bin/env python3
"""um_v2_encode.py: one forward pass per story, saving what R1, R2 and R7 need (user-model correction, run plan
2026-10-18; revised for Agni round 1, encode review N1-N11). NO generation. Runs on Starship (MPS). The analysis
(um_v2_analysis.py) is CPU-only and reads only this script's output; every decision is made there, not here.

Encoding is v2's exactly (wk_backtrack_probe_v2.py, "v2"):
  - model load v2:105-109 (AutoModelForCausalLM, bfloat16, device_map="mps", trust_remote_code=True);
  - a lone user message, apply_chat_template(add_generation_prompt=False, enable_thinking=False), TypeError
    fallback without enable_thinking (v2:127-132); tokenizer(chat, return_tensors="pt"), no truncation (v2:133);
  - model(ids, use_cache=True), keys read from cache.layers[l].keys (v2:136-146). logits_to_keep=1 (N11) changes
    only the LM head's output, not the cache.
The rendered input is <|im_start|>user\\n{story}<|im_end|>\\n: 3 preamble tokens, the story, 2 trailing tokens
(measured on all 900 trials, 2026-10-18). The story's token span comes from the tokenizer's offset mapping (N1).
Per trial, at the 16 full-attention layers (LAB_SOP §3; model layer ids saved in every shard, N7), each pooled four
ways (POOLS):
  mean_all     every position (v2:148);
  mean_skip50  positions >= 50 (the paper: main.tex:200, :250, :273; the bridge's hidden[:, 50:], bridge:476-477,
               applied to this same lone-user-message input in its re-encode, bridge:671-701, :1489-1551);
               the bridge's fallback for T <= 50 is kept and flagged (none of the 900 trials has T <= 50);
  last         the final position, the newline after <|im_end|> (v2:150);
  story        the story's own tokens only, no template (N1, exploratory).
  keys_pool, vals_pool  (16, 4, n_kv*head_dim), flattened head-major exactly as v2:148/150
  res_out    decoder-layer OUTPUT (forward hook on the layer, bridge:467-480): what v1 projected through W_K
  res_in_ln  OUTPUT of layers[l].input_layernorm = exactly what k_proj reads (verified by the self-check)
  res_pre    INPUT of layers[l].input_layernorm = the residual stream entering layer l
  spectral       (16, 4) Frobenius norm, effective rank, entropy, top-SV ratio of all keys flattened across heads
                 (bridge:345-362), CPU float32, no except-fallback (the May rerun's SVD failed on every call).
  spectral_n100  the same on the first 100 story tokens only, so every trial's matrix has the same n (Agni
                 RANK_BOUNDED_ACROSS_UNMATCHED_N; the shortest story is 104 tokens).
  keys_full, vals_full  (16, n_kv, T, head_dim) float16 (float32 if out of range; recorded), input_ids, n_tokens,
  logit_topk_ids/vals   the last position's 20 largest logits (N11).
Pooled arrays are float32 (residual class means differ by small amounts on O(10-100) components; fp16 keeps ~3
digits). W_K (float32, lossless from bf16), k_norm and input_layernorm weights are saved once to wk.npz.

Checks, all recorded in manifest.json:
  - self-check (trial 0 and every --selfcheck-every): k_proj's input IS input_layernorm's output; k_proj's output
    equals F.linear with the saved W_K; per (head, position) |cache key| = |k_norm(k_proj out)| (RoPE preserves
    norms). A failure stops the run.
  - determinism (round 1 N8, round 2 L1/N8): trial 0 is encoded twice back to back, once with no hooks attached,
    and once more after the loop; max |delta| and per-layer relative RMS of every pooled array are recorded.
  - witness (round 2 N2): the 3 fingerprint picks' May stories, teacher-forced; is every May token inside this
    model's temperature-0.7, top-p-0.9 nucleus? (bf16 at the start, fp16 after the loop.)
  - fingerprint (N2): 3 bridge generation prompts with distinct (emotion, topic), rebuilt as bridge:606-622, encoded
    in bf16 at the start and, after the loop, in fp16 (the May rerun's dtype, bridge:1374) with trial 0's keys
    re-encoded in fp16 too. Statistic: ||K_l||_F over the whole layer cache, exactly the May trials'
    encoding_features.per_layer.norms (bridge:316-347). A row whose token count differs from May's is void.
  - distinct stories (N3): n_distinct story sha256 must equal n; per-prompt token Jaccard is recorded.
  - provenance (N5): every shard carries the model commit, dtype, device, script and chat-template sha256 and its
    chat sha256; resume accepts a shard only if all of them match the current run.
SOP §11: no limit below the real maximum. Every story must fit the context (asserted before the model loads). Free
disk must exceed 2x the estimate from the measured lengths (reason: protecting a shared disk).

  ~/observer-env/bin/python um_v2_encode.py --trials ~/user_model_rerun/results/emotion_geometry_bridge/emotion_bridge_trials.json --out ~/user_model_v2/encode
  ... --tokenize-only --out -        # lengths and template checks only; prints, writes nothing; no model load
"""
import argparse, hashlib, itertools, json, os, platform, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"   # v2:33; the paper's model
FULL_ATTENTION_LAYERS = list(range(3, 64, 4))                           # v2:37; LAB_SOP §3
POOLS = ("mean_all", "mean_skip50", "last", "story")
SKIP, NMATCH, TOPK = 50, 100, 20
RES_KINDS = ("res_out", "res_in_ln", "res_pre")
TOL_LINEAR, TOL_NORM = 1e-2, 2e-2   # bf16 rounding (8-bit mantissa, ~0.4%/op); a wrong tensor is off by O(1)
DET_CEIL = 2 ** -8                  # Agni r3 encode N6: per-layer relative RMS ceiling for every repeat of trial 0
#                                     (bf16 unit roundoff, 2^-8 = 0.0039). Above it the encode is void.
WITNESS_SHUFFLE_SEED = 1000         # Agni r3 analysis New-5: the witness's negative control shuffles May's story
SPECTRAL_NAMES = ["frobenius", "eff_rank", "entropy", "top_sv_ratio"]     # column order of spectral, spectral_n100
ENCODE_REGISTERED = {"DET_CEIL": DET_CEIL, "ENCODE_SKIP": SKIP, "ENCODE_NMATCH": NMATCH,
                     "FULL_ATTENTION_LAYERS": FULL_ATTENTION_LAYERS, "SPECTRAL_NAMES": SPECTRAL_NAMES}
BRIDGE_SYSTEM = ("You are a creative writing assistant. Write exactly one paragraph "
                 "(4-6 sentences) as requested. Be vivid and specific. Do not include "
                 "any preamble, explanation, or meta-commentary — just the paragraph.")   # bridge:230-234


def check_prediction(path, sealed):
    """Agni r3 encode N9: the sealed prediction must exist and be a prediction, not a placeholder. And it must be the
    one Lyra sealed: its sha256 equals the prereg's registered prediction_sha256 (Lyra, sign-off 2026-10-19), because
    this script records whatever file it is given and the analysis checks only against that record."""
    b = Path(path).read_bytes() if path and Path(path).is_file() else b""
    t = b.decode("utf-8", "replace")
    if len(b) < 200 or not all(x in t for x in ("R1", "R2", "R7")):
        raise SystemExit(f"{path}: not a sealed prediction (needs >= 200 bytes and R1, R2 and R7 in it)")
    if sha256_bytes(b) != sealed:
        raise SystemExit(f"{path}: sha256 {sha256_bytes(b)[:16]} is not the sealed prediction's {str(sealed)[:16]} "
                         f"(prereg registered_constants.prediction_sha256)")
    return t


def check_registered(prereg_path, mine):
    """The script's registered constants must equal the prereg JSON's (Agni r3 analysis New-9)."""
    reg = json.load(open(prereg_path, encoding="utf-8")).get("registered_constants")
    if not isinstance(reg, dict):
        raise SystemExit(f"{prereg_path}: no registered_constants block")
    bad = {k: (v, reg.get(k)) for k, v in mine.items() if reg.get(k) != v}
    if bad:
        raise SystemExit(f"constants differ from the prereg (script, prereg): {bad}")
    return reg


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_trials(path):
    raw = json.load(open(path, encoding="utf-8"))
    return raw["trials"] if isinstance(raw, dict) and "trials" in raw else raw   # v2:100


def render(tok, story):
    msgs = [{"role": "user", "content": story}]                                  # v2:127
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def story_span(tok, chat, story):
    """Token indices [a, b) whose characters overlap the story, from the offset mapping (N1)."""
    off = tok(chat, return_offsets_mapping=True)["offset_mapping"]
    c0 = chat.find(story)
    c1 = c0 + len(story)
    idx = [j for j, (x, y) in enumerate(off) if x < c1 and y > c0]
    assert idx and idx == list(range(idx[0], idx[-1] + 1)), "story tokens are not one contiguous span"
    return idx[0], idx[-1] + 1


def tokenize_pass(tok, trials, max_pos):
    """SOP §11: every story must fit, nothing is cut. Also: enable_thinking changes nothing (SOP §8), tokenization
    round-trips, the story span, distinct stories (N3) and per-prompt token overlap."""
    recs, rt_fail, think_diff, cells = [], 0, 0, {}
    for i, t in enumerate(trials):
        story = t["story_text"]
        chat = render(tok, story)
        assert chat.count(story) == 1, f"trial {i}: story not intact in the rendered prompt"
        assert chat.count("<think>") == story.count("<think>"), f"trial {i}: template opened a think block (SOP §8)"
        plain = tok.apply_chat_template([{"role": "user", "content": story}], tokenize=False,
                                        add_generation_prompt=False)
        think_diff += int(plain != chat)
        ids = tok(chat, return_tensors="pt").input_ids[0].tolist()
        assert len(ids) <= max_pos, f"trial {i}: {len(ids)} tokens > context {max_pos}"
        a, b = story_span(tok, chat, story)
        assert b - a >= NMATCH, f"trial {i}: {b - a} story tokens < NMATCH={NMATCH}"
        ok = tok.decode(ids) == chat
        rt_fail += int(not ok)
        cells.setdefault((t["emotion"], t["topic_idx"]), []).append(set(ids[a:b]))
        recs.append({"trial": i, "n_tokens": len(ids), "story_tok_start": a, "story_tok_end": b, "roundtrip_ok": ok,
                     "chat_sha256": sha256_text(chat), "story_sha256": sha256_text(story)})
    L = np.array([r["n_tokens"] for r in recs])
    S = np.array([r["story_tok_end"] - r["story_tok_start"] for r in recs])
    n_distinct = len({r["story_sha256"] for r in recs})
    jac = [len(x & y) / len(x | y) for v in cells.values() for x, y in itertools.combinations(v, 2)]
    assert n_distinct == len(recs), f"{len(recs) - n_distinct} duplicate stories (N3); the unit of analysis changes"
    return {"n": len(L), "min": int(L.min()), "median": float(np.median(L)), "mean": float(L.mean()),
            "max": int(L.max()), "sum": int(L.sum()), "n_le_skip": int((L <= SKIP).sum()),
            "story_tokens_min": int(S.min()), "story_tokens_max": int(S.max()),
            "story_start_values": sorted({r["story_tok_start"] for r in recs}),
            "tokens_after_story_values": sorted({int(r["n_tokens"] - r["story_tok_end"]) for r in recs}),
            "n_distinct_story_sha256": n_distinct, "n_prompts": len(cells),
            "per_prompt_token_jaccard": {"mean": float(np.mean(jac)) if jac else None,
                                         "max": float(np.max(jac)) if jac else None, "n_pairs": len(jac)},
            "context_max_position_embeddings": int(max_pos), "roundtrip_failures": rt_fail,
            "renders_differing_without_enable_thinking": think_diff, "per_trial": recs}


def pool4(x, a, b):
    """x: (T, D) float32 -> (4, D) in POOLS order; [a, b) is the story span."""
    skip = x[SKIP:].mean(0) if x.shape[0] > SKIP else x.mean(0)   # bridge:476-479 fallback (flagged per trial)
    return torch.stack([x.mean(0), skip, x[-1], x[a:b].mean(0)])


def get_layers(model):
    layers = model.model.layers                                    # v2:114
    assert len(layers) == 64, f"expected 64 decoder layers, got {len(layers)}"
    for li, L in enumerate(layers):
        has = hasattr(L, "self_attn")
        assert has == (li in FULL_ATTENTION_LAYERS), f"layer {li}: self_attn={has} contradicts SOP §3"
    for li in FULL_ATTENTION_LAYERS:
        assert layers[li].self_attn.k_proj.bias is None, "k_proj has a bias; d_e = W_K(r_e - r) would be affine"
    return layers


class Taps:
    """Read-only forward hooks; pooling happens on device, in float32."""

    def __init__(self, layers, idx):
        self.handles, self.idx, self.span = [], idx, (0, 1)
        for j, li in enumerate(idx):
            L = layers[li]
            self.handles.append(L.input_layernorm.register_forward_hook(self._ln(j)))
            self.handles.append(L.register_forward_hook(self._out(j)))
            self.handles.append(L.self_attn.k_proj.register_forward_hook(self._kp(j)))
        self.reset(False, (0, 1))

    def reset(self, selfcheck, span):
        self.store, self.raw, self.calls, self.sc, self.span = {}, {}, {}, selfcheck, span

    def _count(self, key):
        self.calls[key] = self.calls.get(key, 0) + 1

    def _ln(self, j):
        def f(m, inp, out):
            self._count(("ln", j))
            self.store[("res_pre", j)] = pool4(inp[0][0].float(), *self.span)
            self.store[("res_in_ln", j)] = pool4(out[0].float(), *self.span)
            if self.sc:
                self.raw[("ln", j)] = out.detach()
        return f

    def _out(self, j):
        def f(m, inp, out):
            self._count(("out", j))
            h = out[0] if isinstance(out, tuple) else out          # bridge:471-474
            self.store[("res_out", j)] = pool4(h[0].float(), *self.span)
        return f

    def _kp(self, j):
        def f(m, inp, out):
            self._count(("kp", j))
            if self.sc:
                self.raw[("kin", j)], self.raw[("kout", j)] = inp[0].detach(), out.detach()
        return f

    def assert_once(self):
        for j in range(len(self.idx)):
            for k in ("ln", "out", "kp"):
                n = self.calls.get((k, j), 0)
                assert n == 1, f"hook {k} at layer {self.idx[j]} fired {n} times in one forward"

    def remove(self):
        for h in self.handles:
            h.remove()


def get_kv(cache, li, T):
    if hasattr(cache, "layers"):
        k, v = getattr(cache.layers[li], "keys", None), getattr(cache.layers[li], "values", None)
    elif hasattr(cache, "key_cache"):
        k, v = cache.key_cache[li], cache.value_cache[li]
    else:
        raise RuntimeError("unknown cache type")
    # v2:153-160 silently zero-filled a missing layer; here a missing layer stops the run.
    if k is None or v is None or k.ndim != 4 or k.shape[0] != 1 or k.shape[2] != T:
        raise RuntimeError(f"layer {li}: no usable keys (shape {None if k is None else tuple(k.shape)}, T={T})")
    return k, v


def forward(model, ids, device):
    with torch.no_grad():
        return model(ids.to(device), use_cache=True, logits_to_keep=1)          # v2:136-137; N11


def selfcheck(layers, idx, taps, kv):
    res = {}
    for j, li in enumerate(idx):
        attn = layers[li].self_attn
        kin, kout, ln = taps.raw[("kin", j)], taps.raw[("kout", j)], taps.raw[("ln", j)]
        same = bool(torch.equal(kin, ln))
        lin = F.linear(kin, attn.k_proj.weight).float()
        lin_err = float((lin - kout.float()).abs().max() / kout.float().abs().max().clamp_min(1e-12))
        T, hd = kout.shape[1], attn.head_dim
        n_ref = attn.k_norm(kout.view(1, T, -1, hd)).float()[0].norm(dim=-1).T   # (H, T)
        n_cache = kv[j][0][0].float().norm(dim=-1)                                 # (H, T)
        norm_err = float(((n_cache - n_ref).abs() / n_ref.clamp_min(1e-6)).max())
        res[str(li)] = {"k_proj_reads_input_layernorm": same, "linear_rel_err": lin_err, "rope_norm_rel_err": norm_err}
        if not same or lin_err > TOL_LINEAR or norm_err > TOL_NORM:
            raise RuntimeError(f"self-check failed at layer {li}: {res[str(li)]}")
    return res


def spectral(kf):
    """kf: (H, T, hd) float32 CPU. bridge:345-362, without its try/except."""
    K = kf.reshape(-1, kf.shape[-1])
    S = torch.linalg.svdvals(K)
    Sp = S[S > 1e-10]
    p = Sp / Sp.sum()
    ent = float(-(p * torch.log(p)).sum())
    return [float(torch.linalg.norm(K)), float(np.exp(ent)), ent, float(S[0] / Sp.sum())]


def f16(x):
    """float16 if it fits, else float32. Returns (array, dtype name, absmax, n_subnormal, n_to_zero): how many nonzero
    values fp16 holds only as subnormals (|x| < 6.1e-5, precision lost) or rounds to 0 (|x| < 2.98e-8)."""
    ax = x.abs()
    am, nz = float(ax.max()), ax > 0
    if am >= 65000:
        return x.numpy(), "float32", am, 0, 0
    return x.half().numpy(), "float16", am, int((nz & (ax < 6.104e-5)).sum()), int((nz & (ax < 2.98e-8)).sum())


def keys_pooled(kv, T, span):
    return torch.stack([pool4(k[0].float().cpu().permute(1, 0, 2).reshape(T, -1), *span) for k, _ in kv]).numpy()


def collect(taps, kv, ids, out, full_kv):
    T = ids.shape[1]
    a, b = taps.span
    arr = {kind: torch.stack([taps.store[(kind, j)] for j in range(len(taps.idx))]).cpu().numpy() for kind in RES_KINDS}
    vp, sp, sn, kfull, vfull, info = [], [], [], [], [], {}
    for k, v in kv:
        kf, vf = k[0].float().cpu(), v[0].float().cpu()                   # (H, T, hd)
        vp.append(pool4(vf.permute(1, 0, 2).reshape(T, -1), a, b))
        sp.append(spectral(kf))
        sn.append(spectral(kf[:, a:a + NMATCH]))
        kfull.append(kf), vfull.append(vf)
    arr["keys_pool"] = keys_pooled(kv, T, (a, b))                         # head-major, as v2:148
    arr["vals_pool"] = torch.stack(vp).numpy()
    arr["spectral"], arr["spectral_n100"] = np.array(sp, np.float32), np.array(sn, np.float32)
    arr["input_ids"] = ids[0].cpu().numpy().astype(np.int32)
    arr["n_tokens"] = np.array(T)
    arr["layers"] = np.array(taps.idx)                                    # model layer ids (N7)
    lg = out.logits[0, -1].float().cpu()
    top = torch.topk(lg, TOPK)
    arr["logit_topk_ids"], arr["logit_topk_vals"] = top.indices.numpy().astype(np.int32), top.values.numpy()
    if full_kv:
        for name, xs in (("keys", kfull), ("vals", vfull)):
            arr[name + "_full"], *rest = f16(torch.stack(xs))
            info.update(zip([f"{name}_full_dtype", f"{name}_absmax", f"{name}_fp16_subnormal", f"{name}_fp16_to_zero"],
                            rest))
    return arr, info


def write_shard(path, arrays, meta):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


PROV_KEYS = ("model_commit", "dtype", "device", "script_sha256", "chat_template_sha256")


def shard_ok(path, i, story_sha, chat_sha, prov, full_kv):
    """A shard is reused only if it is this trial, this text, this render and this run's provenance (N5)."""
    if not path.exists():
        return None
    try:
        with np.load(path) as z:                            # closed at once: Windows cannot replace an open file
            meta, files = json.loads(str(z["meta"])), set(z.files)
        need = {"keys_pool", "vals_pool", "spectral", "spectral_n100", "n_tokens", "input_ids", "layers", *RES_KINDS}
        need |= {"keys_full", "vals_full"} if full_kv else set()
        same = (meta["trial"] == i and meta["story_sha256"] == story_sha and meta["chat_sha256"] == chat_sha
                and all(meta.get(k) == prov[k] for k in PROV_KEYS) and need <= files)
        if same:
            return meta
        print(f"  shard {path.name} is from a different run or input; redoing", flush=True)
    except Exception as e:                                  # a torn or foreign file is redone, and said so
        print(f"  shard {path.name} unreadable ({e}); redoing", flush=True)
    return None


def save_wk(out, layers, idx):
    p = out / "wk.npz"
    if p.exists():
        return sha256_file(p)
    a = {"W_K": np.stack([layers[li].self_attn.k_proj.weight.detach().float().cpu().numpy() for li in idx]),
         "k_norm_w": np.stack([layers[li].self_attn.k_norm.weight.detach().float().cpu().numpy() for li in idx]),
         "in_ln_w": np.stack([layers[li].input_layernorm.weight.detach().float().cpu().numpy() for li in idx]),
         "layers": np.array(idx)}
    write_shard(p, a, {"trial": -1, "story_sha256": "", "what": "W_K and norm weights, float32"})
    return sha256_file(p)


def fingerprint_picks(trials, tok):
    """3 trials with distinct (emotion, topic) (round 1, N2: the file is ordered 3 stories per prompt) whose story
    re-tokenises to exactly May's generated length (round 2, N2: the witness teacher-forces May's own tokens).
    On the real file these are trials 0, 33 and 66 (162, 195 and 162 tokens, checked 2026-10-18)."""
    picks, se, st = [], set(), set()
    for i, t in enumerate(trials):
        if (t["emotion"] not in se and t["topic_idx"] not in st and t.get("encoding_features")
                and tok(t["story_text"], return_tensors="pt").input_ids.shape[1] == t.get("n_generated")):
            picks.append(i), se.add(t["emotion"]), st.add(t["topic_idx"])
        if len(picks) == 3:
            break
    return picks


def bridge_prompt(tok, t):
    user = (f"Write a short paragraph about {t['topic']}. The character is feeling deeply {t['emotion']}. "
            f"Show the emotion through their thoughts and actions, not by naming it directly.")       # bridge:236-243
    p = tok.apply_chat_template([{"role": "system", "content": BRIDGE_SYSTEM}, {"role": "user", "content": user}],
                                tokenize=False, add_generation_prompt=True)
    return p if p.endswith("</think>\n") else p + "</think>\n"                                   # bridge:621-622


def fingerprint(model, tok, trials, picks, device, idx):
    """||K_l||_F of the bridge's generation prompt (bridge:606-622) for each pick. Formula as the May value
    (bridge:316-347: torch.norm of the layer's keys flattened to (H*T, hd))."""
    rows = []
    for i in picks:
        t = trials[i]
        ids = tok(bridge_prompt(tok, t), return_tensors="pt").input_ids
        o = forward(model, ids, device)
        now = [float(get_kv(o.past_key_values, li, ids.shape[1])[0].float().norm()) for li in idx]
        may = t["encoding_features"]["per_layer"]["norms"]
        may_ids = t["encoding_features"]["per_layer"].get("layer_indices")
        rows.append({"trial": i, "n_prompt_tokens_now": int(ids.shape[1]),
                     "n_prompt_tokens_may": t["encoding_features"]["aggregate"]["n_prompt_tokens"],
                     "norms_now": now, "norms_may": may, "n_norms_may": len(may), "may_layer_ids": may_ids,
                     "finite": bool(np.all(np.isfinite(now)))})
        # round 2, N3: May's list must be the same 16 layers in the same order, else the row is void
        aligned = len(may) == len(idx) and (may_ids is None or list(may_ids) == list(idx))
        rows[-1]["void"] = (rows[-1]["n_prompt_tokens_now"] != rows[-1]["n_prompt_tokens_may"] or not rows[-1]["finite"]
                            or not aligned)
        del o
    if not rows:                                            # NULL_FROM_UNVERIFIED_SEARCH: an empty run is not a run
        raise RuntimeError("fingerprint built no rows: encoding_features.per_layer.norms not found in the trials")
    return rows


def witness(model, tok, trials, picks, device, temp=0.7, top_p=0.9):
    """Direction-sensitive fingerprint (Agni round 2, encode N2; ||K_l||_F is set mostly by k_norm's weights).
    May sampled each story from the bridge prompt with do_sample, temperature 0.7, top_p 0.9 (bridge:642-650; the
    snapshot has no generation_config, and any top_k could only shrink the set). So under May's weights every token
    of May's own story lies inside this temperature-0.7, top-p-0.9 nucleus, given the prompt and May's earlier
    tokens. Teacher-forced; one forward per pick; full logits."""
    rows = []
    for i in picks:
        t = trials[i]
        p_ids = tok(bridge_prompt(tok, t), return_tensors="pt").input_ids
        s_ids = tok(t["story_text"], return_tensors="pt").input_ids
        with torch.no_grad():
            lg = model(torch.cat([p_ids, s_ids], 1).to(device), use_cache=False).logits[0].float()
        P, S = p_ids.shape[1], s_ids.shape[1]
        tgt = s_ids[0].to(lg.device)
        inside, nll, top1 = [], [], []
        for j in range(S):
            row = lg[P - 1 + j]
            sp, si = torch.sort(torch.softmax(row / temp, -1), descending=True)
            k = int((torch.cumsum(sp, 0) < top_p).sum()) + 1          # HF TopP: keep while mass before < top_p
            inside.append(bool((si[:k] == tgt[j]).any()))
            nll.append(float(-torch.log_softmax(row, -1)[tgt[j]]))
            top1.append(int(row.argmax()) == int(tgt[j]))
        # negative control (Agni r3 analysis New-5): May's tokens in a shuffled order are not a sample from this model
        # in this context, so a membership test that can tell samples from non-samples must flag many of them
        perm = torch.from_numpy(np.random.default_rng(WITNESS_SHUFFLE_SEED + i).permutation(S))
        sh = s_ids[:, perm]
        with torch.no_grad():
            lg2 = model(torch.cat([p_ids, sh], 1).to(device), use_cache=False).logits[0].float()
        t2, out2 = sh[0].to(lg2.device), 0
        for j in range(S):
            sp, si = torch.sort(torch.softmax(lg2[P - 1 + j] / temp, -1), descending=True)
            k = int((torch.cumsum(sp, 0) < top_p).sum()) + 1
            out2 += int(not bool((si[:k] == t2[j]).any()))
        rows.append({"trial": i, "n_story_tokens": S, "retok_matches_n_generated": S == t.get("n_generated"),
                     "inside": inside, "n_outside": S - sum(inside), "nll_mean": float(np.mean(nll)),
                     "top1_frac": float(np.mean(top1)), "n_outside_shuffled": out2,
                     "may_generated_ids_available": False})   # May saved text, not ids (Agni r3 encode N7)
        del lg, lg2
    return rows


def encode_trial(model, tok, taps, t, device, idx, selfcheck_on, layers):
    chat = render(tok, t["story_text"])
    ids = tok(chat, return_tensors="pt").input_ids                              # v2:133, no truncation
    span = story_span(tok, chat, t["story_text"])
    taps.reset(selfcheck_on, span)
    o = forward(model, ids, device)
    taps.assert_once()
    T = int(ids.shape[1])
    kv = [get_kv(o.past_key_values, li, T) for li in idx]
    sc = selfcheck(layers, idx, taps, kv) if selfcheck_on else None
    return chat, ids, span, o, kv, sc


def max_delta(a, b, keys=("keys_pool", "vals_pool", *RES_KINDS)):
    return {k: {"max_abs": float(np.abs(a[k] - b[k]).max()),
                "max_rel": float(np.abs(a[k] - b[k]).max() / max(np.abs(a[k]).max(), 1e-12)),
                "rms_rel_per_layer": (np.sqrt(((a[k] - b[k]) ** 2).mean(axis=(1, 2)))
                                      / np.maximum(np.sqrt((a[k] ** 2).mean(axis=(1, 2))), 1e-12)).tolist()}
            for k in keys}


def det_worst(det, keys):
    """The largest per-layer relative RMS over every pooled array of the named repeat checks."""
    return max(v for k in keys for arr in det[k].values() for v in arr["rms_rel_per_layer"])


def write_manifest(out, man, metas):
    n_all = man["trials_file"]["n_in_file"]
    man = dict(man, trials=[metas[i] for i in sorted(metas)], n_done=len(metas),
               complete=len(metas) == n_all and man["trials_file"]["n"] == n_all,          # SILENT_N_CHANGE
               updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    tmp = out / "manifest.json.tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, out / "manifest.json")


def run_encode(model, tok, trials, out, a, man):
    """Everything after the model is loaded. Separated so the dry run can drive it with a tiny fake model."""
    out = Path(out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    layers = get_layers(model)
    idx = FULL_ATTENTION_LAYERS
    # Agni r3 encode N8: None == None would let a local-path load pass the provenance check
    assert man["model"]["commit_hash"] is not None, "no model commit hash: provenance cannot be checked"
    man["spectral_feature_names"] = SPECTRAL_NAMES
    prov = {"model_commit": man["model"]["commit_hash"], "dtype": a.dtype, "device": a.device,
            "script_sha256": man["script_sha256"], "chat_template_sha256": man["chat_template_sha256"]}
    man["provenance"] = prov
    man["wk_sha256"] = save_wk(out, layers, idx)
    prev = json.load(open(out / "manifest.json")) if (out / "manifest.json").exists() else {}
    man["selfchecks"] = prev.get("selfchecks", {})
    picks = fingerprint_picks(trials, tok) if a.fingerprint else []
    fp = {"picks": picks, "statistic": "||K_l||_F, keys of the whole prompt flattened to (H*T, hd), per layer",
          "bf16": fingerprint(model, tok, trials, picks, a.device, idx) if a.fingerprint else "skipped",
          "bf16_witness": witness(model, tok, trials, picks, a.device) if a.fingerprint else "skipped"}
    # round 2, encode L1: trial 0 with no hooks attached, to compare with the hooked passes
    chat0 = render(tok, trials[0]["story_text"])
    ids0 = tok(chat0, return_tensors="pt").input_ids
    o0 = forward(model, ids0, a.device)
    k_unhooked = keys_pooled([get_kv(o0.past_key_values, li, ids0.shape[1]) for li in idx], ids0.shape[1],
                             story_span(tok, chat0, trials[0]["story_text"]))
    del o0
    taps = Taps(layers, idx)
    metas, t_start, n_new = {}, time.time(), 0
    try:
        # round 1, N8: trial 0 twice, back to back, before the loop
        pools = []
        for _ in range(2):
            _, i0, _, o0, kv0, _ = encode_trial(model, tok, taps, trials[0], a.device, idx, False, layers)
            pools.append(collect(taps, kv0, i0, o0, False)[0])            # read the taps before the next pass
            del o0, kv0
        man["determinism"] = {"trial": 0, "back_to_back": max_delta(pools[0], pools[1]),
                              "unhooked_vs_hooked": max_delta({"keys_pool": k_unhooked}, pools[0], ("keys_pool",)),
                              "bit_identical_back_to_back": all(np.array_equal(pools[0][k], pools[1][k])
                                                                for k in ("keys_pool", "vals_pool", *RES_KINDS)),
                              "bit_identical_unhooked": bool(np.array_equal(k_unhooked, pools[0]["keys_pool"])),
                              "ceiling_rms_rel": DET_CEIL}
        del pools
        worst = det_worst(man["determinism"], ("back_to_back", "unhooked_vs_hooked"))
        man["determinism"]["worst_pre_loop"] = worst
        if worst > DET_CEIL:                       # Agni r3 encode N6: stop before 900 passes on a broken device
            raise RuntimeError(f"determinism ceiling: per-layer relative RMS {worst:.3g} > {DET_CEIL:.3g}; encode void")
        for i, t in enumerate(trials):
            story = t["story_text"]
            ssha, chat = sha256_text(story), render(tok, story)
            path = out / "shards" / f"t{i:04d}.npz"
            m = shard_ok(path, i, ssha, sha256_text(chat), prov, a.full_kv)
            if m is not None:
                metas[i] = dict(m, shard_sha256=sha256_file(path))
                continue
            t0 = time.time()
            chat, ids, span, o, kv, sc = encode_trial(model, tok, taps, t, a.device, idx,
                                                      i == 0 or i % a.selfcheck_every == 0, layers)
            T = int(ids.shape[1])
            arr, info = collect(taps, kv, ids, o, a.full_kv)
            meta = {"trial": i, "shard": path.name, "emotion": t["emotion"], "topic_idx": t["topic_idx"],
                    "story_idx": t["story_idx"], "n_generated": t.get("n_generated"), "story_sha256": ssha,
                    "chat_sha256": sha256_text(chat), "n_tokens": T, "story_tok_start": span[0],
                    "story_tok_end": span[1], "skip50_fallback": T <= SKIP, "seconds": round(time.time() - t0, 3),
                    **prov, **info}
            write_shard(path, arr, meta)
            meta["shard_sha256"] = sha256_file(path)
            metas[i] = meta
            if sc is not None:
                man["selfchecks"][str(i)] = sc
            n_new += 1
            del o, kv, arr
            taps.reset(False, (0, 1))
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()                                             # v2:166-167
            if n_new % 25 == 0 or i == len(trials) - 1:
                rate = (time.time() - t_start) / n_new
                print(f"  [{i + 1}/{len(trials)}] T={T} {meta['seconds']:.2f}s  "
                      f"ETA {rate * (len(trials) - i - 1) / 60:.1f} min", flush=True)
                write_manifest(out, dict(man, fingerprint=fp), metas)
        # round 2, encode N8: trial 0 again after the loop (memory state has changed), still bf16, against its shard
        _, i0, _, o0, kv0, _ = encode_trial(model, tok, taps, trials[0], a.device, idx, False, layers)
        late = collect(taps, kv0, i0, o0, False)[0]
        del o0, kv0
        with np.load(out / "shards" / "t0000.npz") as z:
            ref = {k: z[k] for k in ("keys_pool", "vals_pool", *RES_KINDS)}
        man["determinism"]["post_loop_vs_shard"] = max_delta(ref, late)
        man["determinism"]["bit_identical_post_loop"] = all(np.array_equal(ref[k], late[k]) for k in ref)
        worst = det_worst(man["determinism"], ("post_loop_vs_shard",))
        man["determinism"]["worst_post_loop"] = worst
        if worst > DET_CEIL:
            raise RuntimeError(f"determinism ceiling after the loop: {worst:.3g} > {DET_CEIL:.3g}; encode void")
    finally:
        taps.remove()
        write_manifest(out, dict(man, fingerprint=fp), metas)
    if a.fingerprint and a.alt_dtype:
        # After the loop, so a cast cannot touch the encode: the same weights in May's dtype (N2, N8). This gives the
        # fp16-now vs fp16-May comparison and the size of a numeric-path change at fixed weights.
        model.to(getattr(torch, a.alt_dtype))
        fp[a.alt_dtype] = fingerprint(model, tok, trials, picks, a.device, idx)
        fp[a.alt_dtype + "_witness"] = witness(model, tok, trials, picks, a.device)
        taps2 = Taps(layers, idx)
        try:
            e = encode_trial(model, tok, taps2, trials[0], a.device, idx, False, layers)
        finally:
            taps2.remove()
        k_alt = keys_pooled(e[4], int(e[1].shape[1]), e[2])
        with np.load(out / "shards" / "t0000.npz") as z:
            k_ref = z["keys_pool"]
        fp["trial0_keys_pool_" + a.alt_dtype + "_vs_" + a.dtype] = {
            "rms_rel_per_layer": (np.sqrt(((k_alt - k_ref) ** 2).mean(axis=(1, 2)))
                                  / np.sqrt((k_ref ** 2).mean(axis=(1, 2)))).tolist(),
            "max_rel": float(np.abs(k_alt - k_ref).max() / np.abs(k_ref).max())}
        write_manifest(out, dict(man, fingerprint=fp), metas)
    return metas


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", required=True, help="emotion_bridge_trials.json (900 trials)")
    ap.add_argument("--out", required=True, help="output dir; '-' only with --tokenize-only (print, write nothing)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--hf-cache", default=None, help="v2 used /Users/margaret/models/hf_cache, which no longer "
                    "exists; the model now sits in the default HF cache, so the default is None")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", help="v2:107")
    ap.add_argument("--alt-dtype", default="float16", help="dtype of the after-loop fingerprint pass (May's, "
                    "bridge:1374); '' to skip")
    ap.add_argument("--limit", type=int, default=0, help="first N trials only (debugging; 0 = all; never complete)")
    ap.add_argument("--no-full-kv", dest="full_kv", action="store_false", help="skip per-position K/V (~10 GB)")
    ap.add_argument("--no-fingerprint", dest="fingerprint", action="store_false")
    ap.add_argument("--selfcheck-every", type=int, default=100)
    ap.add_argument("--tokenize-only", action="store_true", help="lengths and template checks; no model load")
    ap.add_argument("--prediction", default=None, help="Lyra's sealed R1R2_PREDICTION.md; required for a run, so "
                    "'sealed before the encode' is a hash in the manifest, not a claim (Agni round 2, N10)")
    ap.add_argument("--prereg", default=None, help="prereg_um_v2_R1R2.json; required for a run: the script's "
                    "registered constants are checked against it")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.out == "-" and not a.tokenize_only:
        raise SystemExit("--out - is only for --tokenize-only")
    torch.manual_seed(42), np.random.seed(42)                     # v2:94; nothing here is random
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers
    trials = load_trials(a.trials)
    n_file = len(trials)
    if a.limit:
        trials = trials[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=a.hf_cache, trust_remote_code=True)          # v2:105
    cfg = AutoConfig.from_pretrained(a.model, cache_dir=a.hf_cache, trust_remote_code=True)
    tc = getattr(cfg, "text_config", cfg)
    full = [i for i, x in enumerate(tc.layer_types) if x == "full_attention"]
    assert full == FULL_ATTENTION_LAYERS, f"config full-attention layers {full} != SOP §3"
    lens = tokenize_pass(tok, trials, tc.max_position_embeddings)
    summary = {k: v for k, v in lens.items() if k != "per_trial"}
    print("token lengths:", json.dumps(summary), flush=True)
    if a.tokenize_only:
        if a.out != "-":
            Path(a.out).mkdir(parents=True, exist_ok=True)
            json.dump(lens, open(Path(a.out) / "token_lengths.json", "w"), indent=1)
        return 0
    if not a.prereg:
        raise SystemExit("--prereg (prereg_um_v2_R1R2.json) is required before the encode")
    reg = check_registered(a.prereg, ENCODE_REGISTERED)
    check_prediction(a.prediction, reg.get("prediction_sha256"))   # Agni r2 N10, r3 N9; the seal (sign-off)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(lens, open(out / "token_lengths.json", "w"), indent=1)
    nkv, hd, H = tc.num_key_value_heads, tc.head_dim, tc.hidden_size
    per_tok = 2 * 16 * nkv * hd * 2 if a.full_kv else 0
    est = lens["sum"] * per_tok + len(trials) * 16 * 4 * 4 * (2 * nkv * hd + 3 * H) + 16 * nkv * hd * H * 4
    free = shutil.disk_usage(out).free
    print(f"disk: estimate {est / 1e9:.1f} GB, free {free / 1e9:.0f} GB", flush=True)
    if free < 2 * est:
        raise SystemExit(f"free disk {free / 1e9:.0f} GB < 2x estimate {est / 1e9:.1f} GB; stopping before load")
    t0 = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(a.model, cache_dir=a.hf_cache, dtype=getattr(torch, a.dtype),
                                                       device_map=a.device, trust_remote_code=True,
                                                       output_loading_info=True)                       # v2:106-108
    model.eval()
    info = {k: sorted(map(str, v)) for k, v in (info if isinstance(info, dict) else {}).items()}
    # The checkpoint stores model.language_model.*; if that mapping failed, transformers warns and leaves the
    # layers randomly initialised. A missing weight stops the run (the silent-merge failure class).
    if info.get("missing_keys"):
        raise SystemExit(f"{len(info['missing_keys'])} weights missing (random init), e.g. {info['missing_keys'][:3]}")
    man = {"script": Path(__file__).name, "script_sha256": sha256_file(__file__), "args": vars(a),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_seconds": round(time.time() - t0, 1),
           "load_info": {k: {"n": len(v), "first": v[:5]} for k, v in info.items()},
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "transformers": transformers.__version__, "platform": platform.platform()},
           "model": {"id": a.model, "class": type(model).__name__, "dtype": a.dtype, "device": a.device,
                     "commit_hash": getattr(model.config, "_commit_hash", None) or getattr(cfg, "_commit_hash", None),
                     "full_attention_layers": FULL_ATTENTION_LAYERS, "n_kv_heads": nkv, "head_dim": hd,
                     "hidden_size": H, "partial_rotary_factor": getattr(tc, "partial_rotary_factor", None)},
           "chat_template_sha256": sha256_text(tok.chat_template or ""),
           "trials_file": {"path": str(a.trials), "sha256": sha256_file(a.trials), "n": len(trials),
                           "n_in_file": n_file},
           "token_lengths": summary, "pools": list(POOLS), "skip": SKIP, "nmatch": NMATCH,
           "prediction": {"path": str(a.prediction), "sha256": sha256_file(a.prediction)},
           "prereg": {"path": str(a.prereg), "sha256": sha256_file(a.prereg)},
           "arrays": {"keys_pool/vals_pool": "(16, 4, n_kv*head_dim) float32, head-major",
                      "res_out/res_in_ln/res_pre": "(16, 4, hidden) float32",
                      "spectral/spectral_n100": "(16, 4) frob, eff_rank, entropy, top_sv_ratio",
                      "keys_full/vals_full": "(16, n_kv, T, head_dim) float16 unless *_full_dtype says float32",
                      "logit_topk_ids/vals": "last position (the newline after <|im_end|>): a determinism aid, not a "
                                             "margin; it measures nothing about emotion"}}
    metas = run_encode(model, tok, trials, out, a, man)
    print(f"done: {len(metas)}/{len(trials)} trials, {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
