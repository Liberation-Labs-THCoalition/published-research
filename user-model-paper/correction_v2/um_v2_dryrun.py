#!/usr/bin/env python3
"""um_v2_dryrun.py: CPU dry run of um_v2_encode.py and um_v2_analysis.py. No model, no GPU, tiny synthetic tensors.
Revised for Agni rounds 1, 2 and 3: every new branch and guard has a case here that must make it fire.

  python um_v2_dryrun.py --work SCRATCH_DIR [--jobs 4] [--skip-analysis] [--v2 PATH]

Part A, analysis, on synthetic encode outputs written with the encode's own write_shard/write_manifest:
  PLANTED  emotion in the residuals (so, through W_K, in the keys) and in the spectral norm; a weak text cue.
           Must find: R2 EXCEEDS_TEXT, R7 SIGNAL_FOUND, R0 reproduced, pipeline control passed.
  NULL     emotion only in the text. Must not find: R2 FAIL_WITHIN_2X_CHANCE, R1 at chance, R7 NULL_HOLDS.
  WEAK     a little emotion in the cache, a strong text cue: R2 must be FAIL_BELOW_TEXT.
  MATCHED  cache a little above text, under Lyra's 3-point floor: R2 must be INCONCLUSIVE (round 2).
  CONFOUND emotion carried by story length, weak text: R2 must be CONFOUND_SUSPECTED (round 2, Lyra's branch).
           The fixture strengths for WEAK, MATCHED, NORM10 and CONFOUND were tuned until each lands in its branch; the analysis
           was not changed to get there.
  NORM10   (analysis N1, CRITICAL) the signal is in model layers 3/7/11 and model layer 15 carries none but has 10x
           the residual norm: the primary per-layer cosine must find it; the old concatenated cosine must not.
  Unlike breakages (each a different failure, to show the checks can fail):
    LEAK   directions estimated with the test fold included: on NULL data this must score far above chance.
    R7     a constant column must NOT void a set (N2); all-constant -> UNINFORMATIVE_ALL_CONSTANT; features with no
           structure at all -> UNINFORMATIVE_INSTRUMENT, never NULL_HOLDS (NULL_FROM_UNVERIFIED_SEARCH).
    PORT   the three v2 functions are AST-identical to the committed v2 file.
    FINGERPRINT  IN_BAND / OUT_OF_BAND / VOID from constructed rows.
  Stats: McNemar against a hand value; DeLong against sklearn and its size under H0; bootstrap coverage; McNemar
  power is ~alpha/2 at delta 0 and rises with delta; per-layer cosine == cosine of concatenated unit blocks.
Part B, encode, on a tiny fake Qwen-shaped model (64 layers, attention at 3,7,...,63, partial RoPE):
  pooling (incl. story-only) against the saved full keys; LN(res_pre) == res_in_ln at the last position; spectral
  and spectral_n100 not degenerate; self-check passes and fires on a broken model; a missing cache layer raises;
  determinism recorded; fingerprint rows exist, distinct prompts; the alt-dtype pass runs; resume redoes exactly a
  deleted and a torn shard and redoes everything when provenance changes; --limit is never complete; duplicate
  stories stop the tokenize pass; --help and import load no transformers.
"""
import argparse, ast, json, subprocess, sys, types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import um_v2_encode as ENC          # noqa: E402
import um_v2_analysis as AN         # noqa: E402

V2_SHA256 = "8a41036dcd87497b35884a750a5919ff60a82f594246c6131b205b97acae5c6b"   # PROVENANCE_wk_probes.md
V2_DEFAULT = Path.home() / "Desktop" / "Project-Oracle" / "experiments" / "wk_backtrack_probe_v2.py"
EMOTIONS = sorted(["excited", "elated", "enthusiastic", "passionate", "loving", "calm", "content", "peaceful", "serene",
                   "appreciative", "terrified", "furious", "desperate", "panicked", "hostile", "melancholic", "gloomy",
                   "resigned", "defeated", "brooding", "curious", "surprised", "anxious", "restless", "contemplative",
                   "nostalgic", "ambivalent", "bored", "neutral", "focused"])   # the trials file's 30 (bridge:161-205)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


# ------------------------------------------------------------------ Part A: synthetic data
def make_dataset(root, s, cue, seed, norm10=False, len_confound=False):
    """s: emotion strength in the residuals and spectral norm; cue: rate of an emotion word in the text."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    H, nkv, hd, L, NPL = 48, 2, 16, 16, len(ENC.POOLS)
    dk = nkv * hd
    W = rng.normal(size=(L, dk, H)) / np.sqrt(H)
    mu, tau = rng.normal(size=(30, L, H)), rng.normal(size=(10, L, H))
    spec_mu, spec_tau = rng.normal(size=(30, L)), rng.normal(size=(10, L, 3))
    vocab = [f"w{i}" for i in range(300)]
    sig = np.full(L, s)
    scale = np.ones(L)
    if norm10:                       # model layer 15 = row 3: no emotion, 10x the residual norm
        sig[3], scale[3] = 0.0, 10.0
    trials, metas = [], {}
    for ei, e in enumerate(EMOTIONS):
        for t in range(10):
            for k in range(3):
                i = len(trials)
                words = list(rng.choice(vocab, 70)) + [f"topic{t}a", f"topic{t}b"]
                if rng.random() < cue:
                    words.append(f"cue{e}{rng.integers(2)}")
                rng.shuffle(words)
                story = " ".join(words)
                T = int(109 + 3 * ei + rng.integers(0, 3)) if len_confound else int(rng.integers(109, 206))
                trials.append({"emotion": e, "topic_idx": t, "story_idx": k, "topic": f"topic {t}", "story_text": story,
                               "n_generated": T - 5, "story_length": len(story)})
                res = {kind: np.zeros((L, NPL, H), np.float32) for kind in ENC.RES_KINDS}
                kp = np.zeros((L, NPL, dk), np.float32)
                for p in range(NPL):
                    r = (sig[:, None] * mu[ei] + tau[t] + rng.normal(size=(L, H)) * (2.0 if p == 2 else 1.0))
                    rin = r / np.sqrt((r ** 2).mean(-1, keepdims=True))
                    res["res_out"][:, p] = r * scale[:, None]
                    res["res_in_ln"][:, p], res["res_pre"][:, p] = rin, 0.8 * r + 0.1
                    kraw = np.einsum("ldh,lh->ld", W, rin).reshape(L, nkv, hd)
                    kraw /= np.sqrt((kraw ** 2).mean(-1, keepdims=True))
                    kp[:, p] = kraw.reshape(L, dk) * 3 + rng.normal(size=(L, dk)) * 0.5
                sp = np.stack([50 + 2 * s / 0.55 * spec_mu[ei] + spec_tau[t, :, 0] + rng.normal(size=L),
                               5 + spec_tau[t, :, 1] + rng.random(L), 1.6 + spec_tau[t, :, 2] + rng.random(L),
                               0.3 + 0.1 * rng.random(L)], -1).astype(np.float32)
                arr = dict(res, keys_pool=kp, vals_pool=rng.normal(size=(L, NPL, dk)).astype(np.float32),
                           spectral=sp, spectral_n100=sp + rng.normal(size=sp.shape).astype(np.float32) * 0.1,
                           input_ids=rng.integers(0, 1000, T).astype(np.int32), n_tokens=np.array(T),
                           layers=np.array(ENC.FULL_ATTENTION_LAYERS))
                meta = {"trial": i, "shard": f"t{i:04d}.npz", "emotion": e, "topic_idx": t, "story_idx": k,
                        "story_sha256": ENC.sha256_text(story), "n_tokens": T, "story_tok_start": 3,
                        "story_tok_end": T - 2, "skip50_fallback": False}
                ENC.write_shard(root / "shards" / meta["shard"], arr, meta)
                meta["shard_sha256"] = ENC.sha256_file(root / "shards" / meta["shard"])
                metas[i] = meta
    ENC.write_shard(root / "wk.npz", {"W_K": W.astype(np.float32), "layers": np.array(ENC.FULL_ATTENTION_LAYERS)},
                    {"trial": -1, "story_sha256": ""})
    det = {k: {"keys_pool": {"rms_rel_per_layer": [1e-3] * L}} for k in ("back_to_back", "unhooked_vs_hooked",
                                                                         "post_loop_vs_shard")}
    row = lambda now: {"norms_now": [now] * 16, "norms_may": [100.0] * 16, "void": False}
    wit = [{"n_story_tokens": 10, "inside": [True] * 10, "n_outside": 0, "retok_matches_n_generated": True,
            "n_outside_shuffled": 8}]
    # v2's and v1's results files and May's emotion vectors, in their real formats (synthetic numbers)
    json.dump({"probes": [{"name": "W_K mean-pos \u2192 30-class (circularity-fixed)", "accuracy": 368 / 900}]},
              open(root / "v2_results.json", "w"))
    json.dump({"probes": [{"name": "W_K projection \u2192 30-class", "accuracy": 470 / 900}]},
              open(root / "v1_results.json", "w"))
    json.dump({str(l): {e: (mu[ei, j] * sig[j]).tolist() for ei, e in enumerate(EMOTIONS)}
               for j, l in [(0, 0), (0, 3), (1, 7), (2, 11), (3, 15)]}, open(root / "v1_vectors.json", "w"))
    fp = {"bf16": [row(100.2)], "float16": [row(100.1)], "bf16_witness": wit, "float16_witness": wit,
          "trial0_keys_pool_float16_vs_bfloat16": {"rms_rel_per_layer": [2e-3] * L}}
    (root / "prediction.md").write_text("synthetic prediction for the dry run. R1: reproduced. R2: inconclusive. "
                                        "R7: null. Padding so the file passes the sealed-prediction content check, "
                                        "which needs at least 200 bytes and the three run names.\n")
    json.dump({"registered_constants": dict(AN.REGISTERED, **ENC.ENCODE_REGISTERED,
                                            v2_results_sha256=ENC.sha256_file(root / "v2_results.json"),
                                            v1_results_sha256=ENC.sha256_file(root / "v1_results.json"),
                                            v1_vectors_sha256=ENC.sha256_file(root / "v1_vectors.json"),
                                            prediction_sha256=ENC.sha256_file(root / "prediction.md"))},
              open(root / "prereg.json", "w"))
    ENC.write_manifest(root, {"trials_file": {"n": len(trials), "n_in_file": len(trials)}, "determinism": det,
                              "fingerprint": fp, "args": {"alt_dtype": "float16", "dtype": "bfloat16"},
                              "spectral_feature_names": ENC.SPECTRAL_NAMES,
                              "prediction": {"sha256": ENC.sha256_file(root / "prediction.md")},
                              "model": {"full_attention_layers": ENC.FULL_ATTENTION_LAYERS, "id": "SYNTHETIC"}}, metas)
    json.dump({"trials": trials}, open(root / "trials.json", "w"))
    import R0_text_baseline as R0                                        # the real R0 code builds the R0 file
    texts, emos, tops = [t["story_text"] for t in trials], [t["emotion"] for t in trials], [t["topic_idx"] for t in trials]
    json.dump({"all_trials": {"thirty_way": R0.thirty_way(texts, emos, tops), "valence": R0.valence(texts, emos, tops)}},
              open(root / "R0_results.json", "w"))


def run_analysis(root, jobs, n_perm_r7):
    return AN.main(["--enc", str(root), "--trials", str(root / "trials.json"), "--r0", str(root / "R0_results.json"),
                    "--prereg", str(root / "prereg.json"), "--prediction", str(root / "prediction.md"),
                    "--v2-results", str(root / "v2_results.json"), "--v1-results", str(root / "v1_results.json"),
                    "--v1-vectors", str(root / "v1_vectors.json"),
                    "--out", str(root / "results.json"), "--jobs", str(jobs), "--n-perm", "20", "--n-boot", "2000",
                    "--n-power", "1000", "--r1-band-reps", "4", "--n-perm-r7", str(n_perm_r7)])


def _leaky(orig, Kx, Px, yd, tr, te, sim):
    allidx = np.arange(len(yd))
    X_all, _ = orig(Kx, Px, yd, allidx, allidx, sim)       # directions use every trial
    return X_all[tr], X_all[te]


def leak_check(root):
    """Directions from all trials, test fold included (Agni kill C2). Must beat chance on NULL data."""
    man, recs, D, S, S100, nt = AN.load_enc(root, False)
    emos = [r["emotion"] for r in recs]
    y, g = np.array([sorted(set(emos)).index(e) for e in emos]), np.array([r["topic_idx"] for r in recs])
    vv, yv = AN.valence_subset(emos)
    NP = {"mean_all": nt, "mean_skip50": nt - 50, "last": np.ones(len(nt)), "story": nt - 5.0}
    spec = dict(AN.PRIMARY, layers="all16")
    honest = AN.r2_cell(D, spec, y, g, nt, NP, np.array(vv), yv, valence=False)["accuracy"]
    orig = AN.r2_features
    AN.r2_features = lambda Kx, Px, yd, tr, te, sim: _leaky(orig, Kx, Px, yd, tr, te, sim)
    try:
        leaky = AN.r2_cell(D, spec, y, g, nt, NP, np.array(vv), yv, valence=False)["accuracy"]
    finally:
        AN.r2_features = orig
    return honest, leaky


def port_check(v2):
    def fns(src):
        out = {}
        for n in ast.parse(src).body:
            if isinstance(n, ast.FunctionDef) and n.name in ("fwl_residualize", "compute_emotion_directions",
                                                              "project_trials"):
                if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None),
                                                                              ast.Constant):
                    n.body = n.body[1:]
                out[n.name] = ast.dump(n)
        return out
    return fns(v2.read_text(encoding="utf-8")), fns((HERE / "um_v2_analysis.py").read_text(encoding="utf-8"))


def stats_checks(jobs):
    from sklearn.metrics import roc_auc_score
    ca = np.array([1] * 10 + [0] * 2 + [1] * 5, bool)
    cb = np.array([0] * 10 + [1] * 2 + [1] * 5, bool)
    check("McNemar b=10,c=2 two-sided exact p = 0.03857", abs(AN.mcnemar(ca, cb)["p_exact_two_sided"] - 0.0385742) < 1e-6)
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(150), np.zeros(150)].astype(int)
    pa, pb = y + rng.normal(size=300), y * 0.8 + rng.normal(size=300)
    d = AN.delong(y, pa, pb)
    check("DeLong AUCs equal sklearn", abs(d["auc_a"] - roc_auc_score(y, pa)) < 1e-12 and
          abs(d["auc_b"] - roc_auc_score(y, pb)) < 1e-12)
    rej = 0
    for s in range(400):
        r = np.random.default_rng(100 + s)
        base = y * 0.8 + r.normal(size=300)
        rej += AN.delong(y, base + 0.5 * r.normal(size=300), base + 0.5 * r.normal(size=300))["p_two_sided"] < 0.05
    check("DeLong size under H0 near 0.05 (400 sims)", 0.02 <= rej / 400 <= 0.09, f"rate={rej / 400:.3f}")
    cover = 0
    for s in range(200):
        r = np.random.default_rng(500 + s)
        cl = np.repeat(np.arange(300), 3)
        q = np.repeat(r.uniform(0.2, 0.6, 300), 3)
        lo, hi = AN.boot_acc(r.random(900) < q, r.random(900) < q, cl, 1000, s)["ci95"]
        cover += lo <= 0 <= hi
    check("cell-cluster bootstrap 95% CI covers delta=0 (200 sims)", 0.90 <= cover / 200 <= 0.99, f"coverage={cover / 200:.3f}")
    c0 = np.random.default_rng(3).random(900) < 0.38
    pw = {d: AN.mcnemar_power(c0, d, 0.3, 2000, 1) for d in (0.0, 0.03, 0.05, 0.08)}
    check("McNemar power: <= alpha/2 at delta 0, rising with delta", pw[0.0] <= 0.035 and pw[0.0] < pw[0.03] < pw[0.05]
          < pw[0.08], json.dumps(pw))
    check("mde_observed: 300 discordants need a net gain of ~3.8 points at n=900",
          abs(AN.mde_observed(300, 900) - 0.0378) < 0.003, str(AN.mde_observed(300, 900)))
    rng = np.random.default_rng(4)
    Kx, Px, yd = rng.normal(size=(60, 4, 8)), rng.normal(size=(60, 4, 8)), np.repeat(np.arange(3), 20)
    tr = np.arange(60)
    Xp = AN.r2_features(Kx, Px, yd, tr, tr, "cos_perlayer")[0]
    g = Px.mean(0)
    Dm = np.stack([Px[yd == c].mean(0) - g for c in range(3)])
    Du = (Dm / np.linalg.norm(Dm, axis=2, keepdims=True)).reshape(3, -1)
    Ku = (Kx / np.linalg.norm(Kx, axis=2, keepdims=True)).reshape(60, -1)
    Xc = (Ku @ Du.T) / (np.linalg.norm(Ku, axis=1, keepdims=True) * np.linalg.norm(Du, axis=1))
    check("N1: mean of per-layer cosines == cosine of concatenated unit blocks", np.allclose(Xp, Xc, atol=1e-12))
    # R7 verdict logic (analysis N2), on one layer set so a small permutation count can reach alpha
    n, y30 = 900, np.repeat(np.arange(30), 30)
    grp = np.tile(np.repeat(np.arange(10), 3), 30)
    cel = y30 * 10 + grp
    Z = AN.covariates(np.full(n, 150.0) + np.random.default_rng(5).integers(0, 50, n), None, "paper")
    one = {"all16": list(AN.FULL)}
    base = np.random.default_rng(6).normal(size=(n, 16, 4)).astype(np.float32)
    S = base.copy()
    S[:, :, 1], S[:, :, 2] = 1.0, 0.0                                   # the May rerun's constant rank/entropy
    S[:, :, 0] += np.random.default_rng(7).normal(size=(30, 16))[y30] * 1.5 + np.random.default_rng(8).normal(
        size=(10, 16))[grp]
    v = AN.r7(S, y30, grp, Z, cel, 40, 0, jobs, "test", sets=one)
    check("R7: constant rank/entropy columns do not void a set; the varying norm still finds the signal",
          v["verdict"] == "SIGNAL_FOUND" and v["rows"][0]["n_constant_dropped"] == 32, v["verdict"])
    S2 = np.ones((n, 16, 4), np.float32)
    v = AN.r7(S2, y30, grp, Z, cel, 40, 0, jobs, "test", sets=one)
    check("R7: all columns constant -> UNINFORMATIVE_ALL_CONSTANT", v["verdict"] == "UNINFORMATIVE_ALL_CONSTANT",
          v["verdict"])
    v = AN.r7(base, y30, grp, Z, cel, 40, 0, jobs, "test", sets=one)
    check("R7: pure-noise features (blind instrument) -> UNINFORMATIVE_INSTRUMENT, not NULL_HOLDS",
          v["verdict"] == "UNINFORMATIVE_INSTRUMENT", f"{v['verdict']} control={v['rows'][0]['positive_control_topic_accuracy']:.3f}")
    S3 = base.copy()
    S3[:, :, 0] += np.random.default_rng(8).normal(size=(10, 16))[grp] * 1.5     # topic structure, no emotion
    v = AN.r7(S3, y30, grp, Z, cel, 40, 0, jobs, "test", sets=one)
    check("R7: topic structure, no emotion -> NULL_HOLDS (control passes), with a detection threshold and the null "
          "vector saved", v["verdict"] == "NULL_HOLDS" and v["rows"][0]["detection_threshold_accuracy"] is not None
          and len(v["rows"][0]["perm_null"]) == 40, f"{v['verdict']} excludes acc >= "
          f"{v['rows'][0]['detection_threshold_accuracy']}")
    try:
        AN.r7(S3, y30, grp, Z, cel, 50, 0, jobs, "test")
        check("R7: 50 permutations over 17 sets is refused (cannot reach 0.05/17)", False)
    except SystemExit as e:
        check("R7: 50 permutations over 17 sets is refused (cannot reach 0.05/17)", True, str(e)[:60])
    may = [100.0] * 16
    row = lambda now, void=False: {"norms_now": now, "norms_may": may, "void": void}
    w = lambda ins: [{"n_story_tokens": len(ins), "inside": ins, "n_outside": len(ins) - sum(ins),
                      "retok_matches_n_generated": True, "n_outside_shuffled": len(ins)}]
    ok10, one_flip = [True] * 10, [True] * 9 + [False]
    base_fp = lambda nh, wb, wh, void=False: {"bf16": [row([100.3] * 16)], "float16": [row(nh, void)],
                                              "bf16_witness": w(wb), "float16_witness": w(wh)}
    cases = {"IN_BAND": base_fp([100.05] * 16, ok10, ok10),
             "OUT_OF_BAND (norms)": base_fp([103.0] * 16, ok10, ok10),
             "OUT_OF_BAND (witness: a May token outside the nucleus in both dtypes, no flip)": base_fp(
                 [100.05] * 16, one_flip, one_flip),
             "IN_BAND (witness: the outside token is a measured dtype flip)": base_fp([100.05] * 16, one_flip[::-1],
                                                                                    one_flip),
             "VOID": base_fp([100.0] * 16, ok10, ok10, void=True), "NOT_RUN": {}}
    got = [AN.fingerprint_verdict(f, "float16")["verdict"] for f in cases.values()]
    want = [k.split(" ")[0] for k in cases]
    check("FINGERPRINT: norms band, weight-sensitive witness, VOID, NOT_RUN", got == want, str(got))
    # the registered R2 ladder (Lyra's floor, inconclusive region, confound), on constructed correctness vectors
    cells300 = np.repeat(np.arange(300), 3)

    def ladder(b, c, ci=None, sig_power=1.0, confound=False):
        r0 = np.zeros(900, bool)
        r0[:344] = True                                          # R0 at 38.2%, as measured on the real data
        r2 = r0.copy()
        r2[344 + 3 * np.arange(b) % 556] = True                  # b trials R2 right, R0 wrong, one per cell
        r2[3 * np.arange(c) % 344] = False                       # c trials R2 wrong, R0 right, one per cell
        m = AN.mcnemar(r2, r0)
        lo = AN.boot_acc(r2, r0, cells300, 2000, 0)["ci95"][0] if ci is None else ci
        return AN.r2_verdict(int(r2.sum()), int(r0.sum()), 900, m, lo, 0.001, 0.033, True, confound, sig_power), m
    v, m = ladder(30, 12)
    check("LADDER: a +2-point win with p < 0.05 is INCONCLUSIVE (below the 3-point floor), not EXCEEDS",
          v == "INCONCLUSIVE:below_3pt_floor" and m["p_exact_two_sided"] < 0.05,
          f"{v} delta={m['delta']:.4f} p={m['p_exact_two_sided']:.4f}")
    want = {"EXCEEDS_TEXT_TFIDF": ladder(50, 14)[0], "INCONCLUSIVE:cell_ci_includes_0": ladder(50, 14, ci=-0.001)[0],
            "INCONCLUSIVE:underpowered": ladder(130, 100, sig_power=0.5)[0],
            "NOT_DISTINGUISHED_FROM_TEXT_TFIDF": ladder(130, 100, sig_power=0.9)[0],
            "FAIL_BELOW_TEXT_TFIDF": ladder(10, 40)[0], "NOT_ABOVE_TEXT_TFIDF": ladder(20, 20)[0],
            "CONFOUND_SUSPECTED": ladder(50, 14, confound=True)[0]}
    check("LADDER: every registered branch is reachable and lands where the prereg says",
          all(k == v for k, v in want.items()), json.dumps(want))
    # --- round 3 ---
    rv = {(k, h, v1): AN.r1_verdict(k, 368, h, v1) for k, h, v1 in
          ((368, 18, "REPRODUCED"), (368, 0, "REPRODUCED"), (368, 0, "NOT_REPRODUCED"), (360, 3, "REPRODUCED"),
           (360, 3, "NOT_REPRODUCED"))}
    check("R1 (r3): band cap, v1 artifact check and the four reading branches",
          list(rv.values()) == ["UNINFORMATIVE_BAND_TOO_WIDE", "REPRODUCED", "REPRODUCED_WEIGHTS_UNVERIFIED",
                                "NOT_REPRODUCED_CODE_PATH", "NOT_REPRODUCED_WEIGHTS_UNVERIFIABLE"], str(list(rv.values())))
    neg = {"bf16": [row([100.3] * 16)], "float16": [row([100.05] * 16)],
           "bf16_witness": [dict(w(ok10)[0], n_outside_shuffled=1)], "float16_witness": [dict(w(ok10)[0], n_outside_shuffled=1)]}
    check("FINGERPRINT (r3): a witness that cannot flag shuffled May tokens is uninformative",
          AN.fingerprint_verdict(neg, "float16")["verdict"] == "NORMS_ONLY_WITNESS_UNINFORMATIVE")
    reg = {"registered_constants": dict(AN.REGISTERED)}
    ok = AN.check_registered(reg) is not None
    try:
        AN.check_registered({"registered_constants": dict(AN.REGISTERED, FLOOR_PTS=2.0)})
        bad = False
    except SystemExit:
        bad = True
    try:
        AN.check_registered({})
        miss = False
    except SystemExit:
        miss = True
    check("CONSTANTS (r3): equal passes; a changed FLOOR_PTS or a missing block refuses", ok and bad and miss)
    real = json.load(open(HERE / "prereg_um_v2_R1R2.json", encoding="utf-8"))
    try:
        AN.check_registered(real)
        ENC.check_registered(HERE / "prereg_um_v2_R1R2.json", ENC.ENCODE_REGISTERED)
        same = True
    except SystemExit as e:
        same = str(e)
    check("CONSTANTS (r3): the real prereg JSON's registered constants equal both scripts'", same is True, str(same)[:120])
    nb = []
    AN.cv(lambda tr, te: (np.full((len(tr), 2), np.nan), np.full((len(te), 2), np.nan)), y30, grp,
          np.ones((900, 1)), 10, False, nan_box=nb)
    check("NaN (r3): NaN features are counted and a NaN primary is VOID",
          sum(nb) > 0 and AN.r2_verdict(400, 344, 900, {"p_exact_two_sided": 0.001}, 0.01, 0.001, 0.03, True,
                                        False, 0.9, void=True) == "VOID_NAN_FEATURES", f"nan={sum(nb)}")
    tmp = HERE.parent / "um_v2_dryrun_tmp_prediction.md"
    real_pred = HERE / "R1R2_PREDICTION.md"
    seal = json.load(open(HERE / "prereg_um_v2_R1R2.json", encoding="utf-8"))["registered_constants"]["prediction_sha256"]

    def refused(fn, path):
        try:
            fn(path, seal)
            return False
        except SystemExit:
            return True
    tmp.write_text("R1 R2 R7")
    short, short_e = refused(AN.check_prediction, tmp), refused(ENC.check_prediction, tmp)
    check("PREDICTION (r3): a placeholder prediction file is refused by both scripts", short and short_e)
    # The seal (sign-off): a real-looking prediction that is not the sealed one. Same text plus one added line.
    tmp.write_bytes(real_pred.read_bytes() + b"\nR2 will exceed text by 5 points.\n")
    edited, edited_e = refused(AN.check_prediction, tmp), refused(ENC.check_prediction, tmp)
    tmp.unlink()
    sealed_ok = not refused(AN.check_prediction, real_pred) and not refused(ENC.check_prediction, real_pred)
    check("PREDICTION (seal): an edited prediction is refused by both scripts, and the sealed one passes both",
          edited and edited_e and sealed_ok, f"edited refused {edited}/{edited_e}, sealed passes {sealed_ok}")
    try:
        AN.hashed_json(HERE / "R0_results.json", "0" * 64, "v2 results")
        hb = False
    except SystemExit:
        hb = True
    check("HASH (r3): a results file whose sha256 is not the registered one is refused", hb)
    # R7 per-set confound: emotion rides on length, and the feature also carries emotion
    rng = np.random.default_rng(9)
    ntc = 109 + 3 * y30 + rng.integers(0, 3, 900)
    Zc = AN.covariates(ntc.astype(float), None, "paper")
    Sc = rng.normal(size=(900, 16, 4)).astype(np.float32)
    Sc[:, :, 0] += rng.normal(size=(30, 16))[y30] * 1.5 + rng.normal(size=(10, 16))[grp]
    cov = AN.covariate_only_acc(y30, grp, Zc, 10)
    v = AN.r7(Sc, y30, grp, Zc, cel, 40, 0, jobs, "test", sets=one, cov_only=cov)
    check("R7 (r3): a signal in a set whose length covariates alone pass its detection threshold is "
          "CONFOUND_SUSPECTED, not SIGNAL", v["rows"][0]["verdict"] == "CONFOUND_SUSPECTED" and
          v["verdict"] == "CONFOUND_SUSPECTED", f"cov_only={cov:.3f} thr={v['rows'][0]['detection_threshold_accuracy']}")
    yp = AN.perm_within(y30, grp, np.random.default_rng(0), cel)
    same_cell = all(len(set(yp[cel == c])) == 1 for c in np.unique(cel))
    perm_ok = all(sorted(yp[grp == g][::3]) == sorted(y30[grp == g][::3]) for g in range(10))
    check("PERMUTATION: a cell's 3 siblings keep one label; labels are permuted over cells within each topic",
          same_cell and perm_ok and not np.array_equal(yp, y30))


# ------------------------------------------------------------------ Part B: fake model
class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Attn(nn.Module):
    def __init__(self, H, nkv, hd, li, broken):
        super().__init__()
        self.k_proj, self.v_proj = nn.Linear(H, nkv * hd, bias=False), nn.Linear(H, nkv * hd, bias=False)
        self.o_proj, self.k_norm = nn.Linear(nkv * hd, H, bias=False), RMS(hd)
        self.head_dim, self.li, self.broken = hd, li, broken

    def forward(self, x, cache):
        T = x.shape[1]
        k = self.k_norm(self.k_proj(x * 1.01 if self.broken else x).view(1, T, -1, self.head_dim)).transpose(1, 2)
        pos = torch.arange(T).float()[:, None] / (100 ** (torch.arange(2).float() / 2))   # RoPE on dims 0..3
        c, s = torch.cos(pos).to(k.dtype), torch.sin(pos).to(k.dtype)
        k1, k2, rest = k[..., :2], k[..., 2:4], k[..., 4:]
        k = torch.cat([k1 * c - k2 * s, k2 * c + k1 * s, rest], -1)
        v = self.v_proj(x).view(1, T, -1, self.head_dim).transpose(1, 2)
        cache.layers[self.li].keys, cache.layers[self.li].values = k, v
        cnt = torch.arange(1, T + 1, dtype=x.dtype)[None, :, None]
        return self.o_proj(v.transpose(1, 2).reshape(1, T, -1).cumsum(1) / cnt)


class Layer(nn.Module):
    def __init__(self, H, li, broken):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = RMS(H), RMS(H)
        self.mlp = nn.Linear(H, H)
        if li % 4 == 3:
            self.self_attn = Attn(H, 2, 8, li, broken)
        else:
            self.linear_attn = nn.Linear(H, H)

    def forward(self, h, cache):
        x = self.input_layernorm(h)
        a = self.self_attn(x, cache) if hasattr(self, "self_attn") else self.linear_attn(x).cumsum(1) * 0.05
        h = h + a
        return h + self.mlp(self.post_attention_layernorm(h)) * 0.1


class FakeModel(nn.Module):
    def __init__(self, broken=False, drop_cache_layer=None, noisy=False):
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.Module()
        self.model.embed = nn.Embedding(9000, 16)
        self.model.layers = nn.ModuleList([Layer(16, li, broken) for li in range(64)])
        self.lm_head = nn.Linear(16, 9000)
        self.drop = drop_cache_layer
        self.noisy = noisy
        self.config = types.SimpleNamespace(_commit_hash="FAKE")

    def forward(self, ids, use_cache=True, logits_to_keep=0):
        cache = types.SimpleNamespace(layers=[types.SimpleNamespace(keys=None, values=None) for _ in range(64)])
        h = self.model.embed(ids)
        for L in self.model.layers:
            h = L(h, cache)
        if self.drop is not None:
            cache.layers[self.drop].keys = None
        if self.noisy:                                           # a device that is not deterministic
            for li in range(3, 64, 4):
                cache.layers[li].keys = cache.layers[li].keys + 0.05 * torch.randn_like(cache.layers[li].keys)
        h = h[:, -logits_to_keep:] if logits_to_keep else h
        return types.SimpleNamespace(past_key_values=cache, logits=self.lm_head(h))


class FakeTok:
    chat_template = "fake"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n<think>\n" if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        ids = [min(ord(c), 8999) for c in text]
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        return types.SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def fake_args(**kw):
    return types.SimpleNamespace(**dict(dict(device="cpu", dtype="float32", alt_dtype="bfloat16", full_kv=True,
                                             fingerprint=True, selfcheck_every=5), **kw))


def fake_man(n, n_in_file=None, script="s1", commit="FAKE"):
    return {"trials_file": {"n": n, "n_in_file": n if n_in_file is None else n_in_file},
            "model": {"commit_hash": commit}, "script_sha256": script, "chat_template_sha256": "t1"}


def encode_checks(work):
    rng = np.random.default_rng(3)
    trials = [{"emotion": EMOTIONS[i % 30], "topic_idx": i % 10, "story_idx": 0, "topic": "a topic",
               "story_text": " ".join(rng.choice(["the", "rain", "fell", "she", "waited", "quietly"], int(rng.integers(30, 45)))),
               "encoding_features": {"per_layer": {"norms": [1.0] * 16}, "aggregate": {"n_prompt_tokens": 98}}}
              for i in range(12)]
    for t in trials:
        t["n_generated"] = len(t["story_text"])                 # FakeTok is char-level: re-tokenises exactly
    a = fake_args()
    out = work / "enc_fake"
    tok = FakeTok()
    lens = ENC.tokenize_pass(tok, trials, 262144)
    check("tokenize_pass: fit, round-trip, think flag inert, distinct stories, story span recorded",
          lens["roundtrip_failures"] == 0 and lens["renders_differing_without_enable_thinking"] == 0 and
          lens["n_le_skip"] == 0 and lens["n_distinct_story_sha256"] == 12 and lens["story_start_values"] == [17],
          json.dumps({k: lens[k] for k in ("min", "max", "story_tokens_min", "story_start_values")}))
    try:
        ENC.tokenize_pass(tok, trials + [dict(trials[0])], 262144)
        check("N3: a duplicated story stops the tokenize pass", False)
    except AssertionError as e:
        check("N3: a duplicated story stops the tokenize pass", True, str(e)[:50])
    with torch.no_grad():
        metas = ENC.run_encode(FakeModel(), tok, trials, out, a, fake_man(12))
    mf = json.load(open(out / "manifest.json"))
    check("12 shards, manifest complete, self-checks at trials 0,5,10", len(metas) == 12 and mf["complete"] and
          sorted(mf["selfchecks"]) == ["0", "10", "5"] and all(
              v["k_proj_reads_input_layernorm"] for sc in mf["selfchecks"].values() for v in sc.values()))
    dm = mf["determinism"]
    check("N8 + round 2 L1/N8: back-to-back, unhooked-vs-hooked and post-loop checks recorded, all bit-identical here",
          dm["bit_identical_back_to_back"] and dm["bit_identical_unhooked"] and dm["bit_identical_post_loop"]
          and dm["post_loop_vs_shard"]["keys_pool"]["max_abs"] == 0.0)
    fp = mf["fingerprint"]
    check("N2: fingerprint rows exist for 3 distinct prompts, voids flagged, alt-dtype pass and trial-0 keys done",
          len(fp["bf16"]) == 3 and len({(trials[i]["emotion"], trials[i]["topic_idx"]) for i in fp["picks"]}) == 3
          and all(r["void"] for r in fp["bf16"]) and len(fp["bfloat16"]) == 3
          and "trial0_keys_pool_bfloat16_vs_float32" in fp, str(fp["picks"]))
    wb = fp["bf16_witness"]
    check("r3 New-5 / N7 / New-9: witness negative control and May-ids flag recorded; spectral names in the manifest",
          all("n_outside_shuffled" in r and r["may_generated_ids_available"] is False for r in wb)
          and mf["spectral_feature_names"] == ENC.SPECTRAL_NAMES and mf["determinism"]["worst_pre_loop"] <= ENC.DET_CEIL)
    check("round 2 N2: the witness teacher-forced each pick's story in both dtypes and counted nucleus membership",
          len(wb) == 3 and len(fp["bfloat16_witness"]) == 3 and all(len(r["inside"]) == r["n_story_tokens"] and
                                                                   r["retok_matches_n_generated"] for r in wb),
          f"outside={[r['n_outside'] for r in wb]} top1={[round(r['top1_frac'], 2) for r in wb]}")
    bad = dict(trials[0], encoding_features={"per_layer": {"norms": [1.0] * 64}, "aggregate": {"n_prompt_tokens": 1}})
    rows = ENC.fingerprint(FakeModel(), tok, [bad], [0], "cpu", ENC.FULL_ATTENTION_LAYERS)
    check("round 2 N3: May's norm list of the wrong length makes the row void and is recorded",
          rows[0]["void"] and rows[0]["n_norms_may"] == 64)
    with np.load(out / "shards" / "t0004.npz") as zf:
        z = {k: zf[k] for k in zf.files}
    meta = json.loads(str(z["meta"]))
    kf = z["keys_full"].astype(np.float32)
    T, (sa, sb) = int(z["n_tokens"]), (meta["story_tok_start"], meta["story_tok_end"])
    flat = lambda x: x.transpose(0, 2, 1, 3).reshape(16, T, -1)
    ok = (np.allclose(z["keys_pool"][:, 0], flat(kf).mean(1), atol=2e-3) and
          np.allclose(z["keys_pool"][:, 1], flat(kf)[:, 50:].mean(1), atol=2e-3) and
          np.allclose(z["keys_pool"][:, 2], flat(kf)[:, -1], atol=2e-3) and
          np.allclose(z["keys_pool"][:, 3], flat(kf)[:, sa:sb].mean(1), atol=2e-3))
    check("keys_pool mean_all / skip50 / last / story match the saved full keys", ok, f"story span {sa}-{sb} of {T}")
    check("N7/N5/N11: shard carries layer ids, provenance and last-position top-k logits",
          z["layers"].tolist() == ENC.FULL_ATTENTION_LAYERS and all(k in meta for k in ENC.PROV_KEYS)
          and z["logit_topk_ids"].shape == (20,))
    check("round 2 CENSORING: fp16 subnormal and to-zero counts recorded per shard",
          all(k in meta for k in ("keys_fp16_subnormal", "keys_fp16_to_zero", "vals_fp16_subnormal")))
    m = FakeModel()
    ln = [m.model.layers[li].input_layernorm for li in ENC.FULL_ATTENTION_LAYERS]
    pre_last = torch.tensor(z["res_pre"][:, 2])
    rin = torch.stack([ln[j](pre_last[j]) for j in range(16)]).detach().numpy()
    check("input_layernorm(res_pre) == res_in_ln at the last position", np.allclose(rin, z["res_in_ln"][:, 2], atol=1e-5))
    sp = np.stack([np.load(out / "shards" / f"t{i:04d}.npz")["spectral_n100"] for i in range(12)])
    check("spectral_n100 eff_rank > 1 and varies across trials", sp[:, :, 1].min() > 1 and len(np.unique(sp[:, :, 1])) > 12)
    before = {p.name: (p.stat().st_mtime_ns, np.load(p)["keys_pool"].copy()) for p in (out / "shards").glob("*.npz")}
    (out / "shards" / "t0003.npz").unlink()
    torn = out / "shards" / "t0007.npz"
    torn.write_bytes(torn.read_bytes()[:500])
    with torch.no_grad():
        ENC.run_encode(FakeModel(), tok, trials, out, a, fake_man(12))
    changed = sorted(p.name for p in (out / "shards").glob("*.npz") if p.stat().st_mtime_ns != before[p.name][0])
    same = np.array_equal(np.load(out / "shards" / "t0003.npz")["keys_pool"], before["t0003.npz"][1])
    check("resume redoes exactly the deleted and the torn shard, deterministically", changed == ["t0003.npz",
                                                                                                "t0007.npz"] and same,
          str(changed))
    mid = {p.name: p.stat().st_mtime_ns for p in (out / "shards").glob("*.npz")}
    with torch.no_grad():
        ENC.run_encode(FakeModel(), tok, trials, out, a, fake_man(12, script="s2"))
    redone = sum(p.stat().st_mtime_ns != mid[p.name] for p in (out / "shards").glob("*.npz"))
    check("N5: a changed script sha256 makes resume redo every shard", redone == 12, f"{redone}/12")
    with torch.no_grad():
        ENC.run_encode(FakeModel(), tok, trials[:4], work / "enc_limit", a, fake_man(4, n_in_file=12))
    check("SILENT_N_CHANGE: a --limit run is never complete",
          json.load(open(work / "enc_limit" / "manifest.json"))["complete"] is False)
    for name, model in (("attention not reading input_layernorm -> self-check raises", FakeModel(broken=True)),
                        ("missing cache layer -> raises (v2 zero-filled)", FakeModel(drop_cache_layer=7)),
                        ("r3 N6: a nondeterministic device breaches the ceiling -> encode void", FakeModel(noisy=True))):
        try:
            with torch.no_grad():
                ENC.run_encode(model, tok, trials[:3], work / f"enc_bad_{len(RESULTS)}", a, fake_man(3))
            check(name, False, "did not raise")
        except RuntimeError as e:
            check(name, True, str(e)[:70])
    try:
        with torch.no_grad():
            ENC.run_encode(FakeModel(), tok, trials[:2], work / "enc_nocommit", a, fake_man(2, commit=None))
        check("r3 N8: a None commit hash stops the encode", False)
    except AssertionError as e:
        check("r3 N8: a None commit hash stops the encode", True, str(e)[:60])
    try:
        ENC.fingerprint(FakeModel(), tok, [{k: v for k, v in t.items() if k != "encoding_features"} for t in trials],
                        [], "cpu", ENC.FULL_ATTENTION_LAYERS)
        check("NULL_FROM_UNVERIFIED_SEARCH: an empty fingerprint raises", False)
    except RuntimeError as e:
        check("NULL_FROM_UNVERIFIED_SEARCH: an empty fingerprint raises", True, str(e)[:60])
    r = subprocess.run([sys.executable, str(HERE / "um_v2_encode.py"), "--help"], capture_output=True, text=True)
    check("encode --help exits 0", r.returncode == 0)
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, r'%s'); import um_v2_encode; "
                        "print('transformers' in sys.modules)" % HERE], capture_output=True, text=True)
    check("importing the encode loads no transformers (no model code)", r.stdout.strip() == "False", r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument("--v2", default=str(V2_DEFAULT), help="the committed wk_backtrack_probe_v2.py")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    print("== stats, R7 logic, fingerprint logic, port")
    stats_checks(a.jobs)
    v2 = Path(a.v2)
    check("PORT: the v2 file compared is the committed one (sha256 = PROVENANCE)",
          ENC.sha256_file(v2) == V2_SHA256, str(v2))   # PROVENANCE hashes the working file as is
    v2f, mine = port_check(v2)
    check("PORT: v2's three functions AST-identical to the committed file", v2f == mine and len(v2f) == 3, str(sorted(v2f)))
    print("== Part B: encode on a fake model")
    encode_checks(work)
    if not a.skip_analysis:
        cases = (("planted", dict(s=0.55, cue=0.35, seed=7), 400), ("null", dict(s=0.0, cue=0.35, seed=8), 400),
                 ("weak", dict(s=0.25, cue=0.9, seed=9), 0), ("matched", dict(s=0.45, cue=0.9, seed=10), 0),
                 ("norm10", dict(s=0.35, cue=0.35, seed=11, norm10=True), 0),
                 ("confound", dict(s=0.55, cue=0.1, seed=12, len_confound=True), 0))
        for name, kw, npr7 in cases:
            print(f"== Part A: analysis, {name}")
            root = work / name
            make_dataset(root, **kw)
            res = run_analysis(root, a.jobs, npr7)
            v, ck = res["verdicts"], res["checks"]["r0_reproduction"]
            acc, r0 = res["R2"]["primary"]["accuracy"], res["R0"]["acc_10fold"]
            info = (f"R1={res['R1']['mean']['accuracy']:.3f} R2={acc:.3f} R0_10={r0:.3f} R7={v['R7']} "
                    f"R2v={v['R2_primary']} power3={res['R2']['power']['registered_power_at_3pt']}")
            sent_raw = any(x["accuracy"] >= res["R2"]["primary"]["accuracy"]
                           for x in res["R2"]["raw_residual_LR_EXPLORATORY"])
            check(f"{name}: outcome sentences fire exactly when their conditions hold (raw-residual, R0', R0)",
                  sent_raw == any(x.startswith("A raw-residual") for x in res["outcome_sentences"])
                  and (res["tests"]["exploratory"]["R0_tuned_C_EXPLORATORY"]["accuracy"] >=
                       res["R2"]["primary"]["accuracy"]) == any(x.startswith("A tuned") for x in res["outcome_sentences"])
                  and all(k in res["checks"]["confound"] for k in ("flag_covariates_above_2x_chance",))
                  and "sign_flip_cells" in res["tests"]["primary_R2_vs_R0_10fold_30way"]["robustness"],
                  f"{len(res['outcome_sentences'])} sentences; v1 {res['verdicts']['V1_check']}; R1 {res['verdicts']['R1']}")
            check(f"{name}: R0 verdict R0_REPRODUCED, pipeline control passed",
                  ck["verdict"] == "R0_REPRODUCED" and res["checks"]["pipeline_control_topic"]["passed"],
                  json.dumps(res["checks"]["pipeline_control_topic"]))
            if name == "planted":
                check("planted: R2 primary EXCEEDS_TEXT_TFIDF", v["R2_primary"] == "EXCEEDS_TEXT_TFIDF", info)
                check("planted: R1 verdict is a registered name, raw-residual baselines and hashes present",
                      v["R1"] in ("REPRODUCED", "REPRODUCED_WEIGHTS_UNVERIFIED", "NOT_REPRODUCED_CODE_PATH", "NOT_REPRODUCED_WEIGHTS_UNVERIFIABLE", "UNINFORMATIVE_BAND_TOO_WIDE") and len(res["R2"]["raw_residual_LR_EXPLORATORY"]) == 2
                      and {"prereg", "prediction"} <= set(res["meta"]["inputs_sha256"]), v["R1"])
                check("planted: R1 above 2x chance, band computed", res["R1"]["mean"]["accuracy"] > 2 / 30 and
                      res["R1"]["band"]["R"] == 4, str(res["R1"]["band"]["half_width_trials"]))
                check("planted: R7 finds the spectral signal", v["R7"] == "SIGNAL_FOUND")
                # v1 artifact check end to end: the same keys give the same count (REPRODUCED), a count 30 off does not
                man, recs, D, S, S100, nt = AN.load_enc(root, False)
                emo = [r["emotion"] for r in recs]
                g_ = np.array([r["topic_idx"] for r in recs])
                vec = json.load(open(root / "v1_vectors.json"))
                tr_ = json.load(open(root / "trials.json"))["trials"]
                acts = D["K"][:, :4, 0, :]
                args = (acts, D["WK4"], [vec[str(l)] for l in AN.R1_LAYERS], list(vec["0"].keys()),
                        np.array([t["story_length"] for t in tr_], float), emo, g_)
                k1 = AN.v1_counts(AN.v1_features(*args[:4]), *args[4:])[0]
                same = AN.v1_check(*args, k1, [1e-3] * 4, 4, 0, a.jobs)
                off = AN.v1_check(*args, k1 + 30, [1e-3] * 4, 4, 0, a.jobs)
                check("V1 (r3): recomputing v1 on the same keys reproduces its count; a count 30 away does not",
                      same["verdict"] == "REPRODUCED" and off["verdict"] == "NOT_REPRODUCED",
                      f"k={k1} h={same['half_width_trials']}")
                mf = json.load(open(root / "manifest.json"))
                bad = json.loads(json.dumps(mf))
                bad["determinism"]["post_loop_vs_shard"]["keys_pool"]["rms_rel_per_layer"] = [1e-2] * 16
                json.dump(bad, open(root / "manifest.json", "w"))
                try:
                    run_analysis(root, a.jobs, 0)
                    refused = False
                except SystemExit as e:
                    refused = "encode void" in str(e)
                json.dump(mf, open(root / "manifest.json", "w"))
                check("DETERMINISM (r3): the analysis refuses an encode whose repeat checks breach the ceiling", refused)
            elif name == "null":
                check("null: R2 primary FAIL_WITHIN_2X_CHANCE", v["R2_primary"] == "FAIL_WITHIN_2X_CHANCE", info)
                check("null: R1 within 2x chance and NOT_REPRODUCED", res["R1"]["mean"]["accuracy"] <= 2 / 30
                      and v["R1"].startswith("NOT_REPRODUCED"))
                check("null: R7 NULL_HOLDS (every set's control passed)", v["R7"] == "NULL_HOLDS" and
                      all(r["positive_control_passed"] for r in res["R7"]["rows"]))
                check("null: permutation null near chance", res["R2"]["permutation"]["null_mean"] < 2 / 30,
                      str(res["R2"]["permutation"]["null_mean"]))
                honest, leaky = leak_check(root)
                check("LEAK: test-fold-in-directions beats chance on null; the real code does not",
                      leaky > 0.2 and honest <= 2 / 30, f"honest={honest:.3f} leaky={leaky:.3f}")
            elif name == "weak":
                check("weak cache, strong text: R2 FAIL_BELOW_TEXT_TFIDF", v["R2_primary"] == "FAIL_BELOW_TEXT_TFIDF",
                      info)
            elif name == "matched":
                check("matched: cache 0-3 points above text -> INCONCLUSIVE (below the floor), never 'equal'",
                      v["R2_primary"].startswith("INCONCLUSIVE") and "below_3pt_floor" in v["R2_primary"], info)
            elif name == "confound":
                cf = res["checks"]["confound"]
                check("confound: emotion in story length -> CONFOUND_SUSPECTED (covariates alone reach text (TF-IDF))",
                      v["R2_primary"] == "CONFOUND_SUSPECTED" and "covariates_alone_reach_R0" in cf["reasons"],
                      f"{info} cov_only={cf['covariate_only_accuracy']:.3f} fitted={cf['fitted_only_accuracy']:.3f}")
            else:
                old = next(c for c in res["R2"]["exploratory"] if c["spec"] == dict(AN.PRIMARY, sim="cos_concat"))
                ea, emo = res["meta"]["emotions"], np.array(res["per_trial"]["emotion"])
                mo = AN.mcnemar(np.array(ea)[old["pred"]] == emo, np.array(res["R0"]["pred_10fold"]) == emo)
                check("NORM10 (N1): per-layer cosine beats text; the old concatenated cosine is >= 10 points worse "
                      "and cannot be told from text", v["R2_primary"] == "EXCEEDS_TEXT_TFIDF" and acc - old["accuracy"] >= 0.1
                      and (mo["p_exact_two_sided"] >= 0.05 or mo["delta"] <= 0),
                      f"primary={acc:.3f} concat={old['accuracy']:.3f} text={r0:.3f} concat-vs-text p={mo['p_exact_two_sided']:.3f}")
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    json.dump([{"check": n, "pass": ok, "detail": d} for n, ok, d in RESULTS], open(work / "dryrun_report.json", "w"),
              indent=1)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
