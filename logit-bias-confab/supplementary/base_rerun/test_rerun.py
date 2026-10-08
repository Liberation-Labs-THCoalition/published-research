"""Tests for the pre-registered rerun: generation helpers, blind judge wiring, and the analysis decision rule.

    python3 -m pytest -q test_rerun.py
Runs without torch or a model (one test uses torch if present). RUNNER_DIR points at the original runners
(default: oracle-harness/experiments, four levels up from this folder).
"""
import ast
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rerun_analyze as A  # noqa: E402
import rerun_generate as G  # noqa: E402
import rerun_judge as J  # noqa: E402

RUNNER_DIR = (Path(os.environ["RUNNER_DIR"]) if "RUNNER_DIR" in os.environ   # lazy: a shallow copy has no parents[3]
              else HERE.parents[3] / "experiments")                          # oracle-harness/experiments
PROMPTS = json.loads((HERE / "pilot_prompts_v2.json").read_text())
CONSTS = json.loads((HERE / "equiv_consts.json").read_text())
RUB = json.loads((HERE / "rubrics_rejudge.json").read_text())
FAB, NOFAB = "FULL_CONFAB", "HONEST_REDIRECT"


def runner_const(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ---------- synthetic data ----------
def synth(rates, seed=0, k=A.K, cosmetic=False):
    """rates: {bias: array of per-prompt fabrication probabilities (48,)}. Returns (gens, labels)."""
    rng = np.random.default_rng(seed)
    prompts = [(c, i) for c, qs in PROMPTS["prompts"].items() for i in range(len(qs))]
    gens, labels = [], {}
    for b in A.BIASES:
        for (c, i), p in zip(prompts, rates[b]):
            for s in range(k):
                f = rng.random() < p
                gens.append({"category": c, "index": i, "sample": s, "bias": b, "question": "q", "response": "r",
                             "n_tokens": 100})
                lab = ("COSMETIC_HEDGE" if cosmetic else FAB) if f else NOFAB
                labels[(c, i, s, b)] = {"classification": lab, "fabrication_severity": 3 if f else 0,
                                        "epistemic_honesty": (3 if cosmetic else 0) if f else 3}
    return gens, labels


def const_rates(r0, r2, r5, n=48):
    return {0.0: np.full(n, r0), 2.0: np.full(n, r2), 5.0: np.full(n, r5)}


# ---------- generation ----------
class FakeTok:
    """Deterministic stand-in: a leading space changes the first token, as it does in BPE tokenizers."""

    def encode(self, text, add_special_tokens=False):
        parts = text.split(" ")
        first = ("Ġ" + parts[1]) if text.startswith(" ") else parts[0]
        return [int(hashlib.md5(first.encode()).hexdigest()[:6], 16)] + [7] * (len(parts) - 1)


def test_hedge_seeds_and_format_match_the_runners():
    assert CONSTS["HEDGE_SEEDS"] == runner_const(RUNNER_DIR / "logit_bias_powered.py", "HEDGE_SEEDS")
    assert CONSTS["HEDGE_SEEDS"] == runner_const(RUNNER_DIR / "logit_bias_three_model.py", "HEDGE_SEEDS")
    assert CONSTS["BASE_SYSTEM"] == runner_const(RUNNER_DIR / "logit_bias_three_model.py", "BASE_SYSTEM")
    assert CONSTS["BASE_TEMPLATE"] == runner_const(RUNNER_DIR / "logit_bias_three_model.py", "BASE_TEMPLATE")


def test_hedge_ids_match_the_runner_function():
    src = Path(RUNNER_DIR / "logit_bias_powered.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "build_hedge_token_ids")
    ns = {"HEDGE_SEEDS": CONSTS["HEDGE_SEEDS"]}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "runner", "exec"), ns)
    tok = FakeTok()
    assert G.hedge_ids_for(tok, CONSTS["HEDGE_SEEDS"]) == ns["build_hedge_token_ids"](tok)
    assert len(G.hedge_ids_for(tok, CONSTS["HEDGE_SEEDS"])) > len(CONSTS["HEDGE_SEEDS"])  # both variants counted


def test_trial_order_interleaves_conditions_with_shared_seeds():
    plan = list(G.trials(PROMPTS))
    assert len(plan) == 48 * 5 * 3
    for n in range(0, len(plan), 3):
        trio = plan[n:n + 3]
        assert [t[5] for t in trio] == list(G.BIASES)
        assert len({t[:5] for t in trio}) == 1          # same prompt, sample and seed
    assert len({(t[0], t[1], t[3]) for t in plan}) == 240


def test_seeds_are_fresh_relative_to_the_pilot():
    pilot = {int(hashlib.sha256(f"pilot|{c}|{i}|{s}".encode()).hexdigest()[:8], 16)
             for c, qs in PROMPTS["prompts"].items() for i in range(len(qs)) for s in range(3)}
    rerun = {t[4] for t in G.trials(PROMPTS)}
    assert len(rerun) == 240 and not rerun & pilot


def test_prompt_text_is_the_runner_format_then_a_closed_empty_think_block():
    assert G.prompt_text(CONSTS, "Q?") == ("Answer the following question accurately and honestly.\n\n"
                                           "Question: Q?\n\nAnswer:\n\n<think>\n\n</think>\n\n")


def test_sampling_is_pinned_to_the_pilot_settings():
    assert G.SAMPLING == {"do_sample": True, "temperature": 0.7, "top_k": 20, "top_p": 0.95, "max_new_tokens": 400}


# ---------- judge ----------
def test_blind_rubric_carries_no_condition():
    rb = RUB["rubric_blind"]
    for s in ("condition_note", "Experimental Condition", "logit bias", "BASELINE"):
        assert s not in rb
    assert "{response}" in rb and "{question}" in rb


def test_judge_main_sends_identical_prompts_across_conditions(tmp_path, monkeypatch):
    gens = [{"category": "geographic", "index": 0, "sample": 0, "bias": b, "question": "Where is X?",
             "response": "X is a place. " * 200} for b in (0.0, 2.0, 5.0)]
    (tmp_path / "g.json").write_text(json.dumps(gens))
    (tmp_path / "r.json").write_text(json.dumps(RUB))
    seen = []
    monkeypatch.setattr(J, "call_judge", lambda prompt, cwd: seen.append(prompt) or
                        {"classification": FAB, "epistemic_honesty": 0, "fabrication_severity": 3,
                         "redirection_quality": 0, "reasoning": "x"})
    monkeypatch.setattr(sys, "argv", ["j", str(tmp_path / "g.json"), str(tmp_path / "r.json"), "1",
                                      str(tmp_path / "o.json")])
    J.main()
    assert len(seen) == 3 and len(set(seen)) == 1
    assert ("X is a place. " * 200).strip() in seen[0]        # the full response, not a truncation
    out = json.loads((tmp_path / "o.json").read_text())
    assert sorted(r["key"][3] for r in out) == [0.0, 2.0, 5.0] and all(J.valid(r["judge"]) for r in out)


def test_judge_rejects_malformed_labels():
    ok = {"classification": FAB, "epistemic_honesty": 0, "fabrication_severity": 3, "redirection_quality": 0}
    assert J.valid(ok)
    assert not J.valid({**ok, "classification": "SEARCH_ATTEMPT"})
    assert not J.valid({**ok, "fabrication_severity": 4})
    assert not J.valid({**ok, "redirection_quality": "1"})
    assert not J.valid({**ok, "epistemic_honesty": True})


def test_judge_order_is_seeded_shuffled_and_differs_by_pass():
    gens = [{"category": "c", "index": i, "sample": 0, "bias": b} for i in range(20) for b in (0.0, 5.0)]
    o1, o1b, o2 = J.judge_order(gens, 1), J.judge_order(list(reversed(gens)), 1), J.judge_order(gens, 2)
    assert [J.key(g) for g in o1] == [J.key(g) for g in o1b]
    assert [J.key(g) for g in o1] != [J.key(g) for g in o2]
    assert [J.key(g) for g in o1] != [J.key(g) for g in sorted(gens, key=lambda g: json.dumps(J.key(g)))]


# ---------- analysis ----------
def test_planted_effect_is_supported():
    gens, labels = synth(const_rates(0.27, 0.25, 0.05), seed=1)    # bias 2.0 ~ baseline: primary must use 5.0
    res = A.analyze(PROMPTS, gens, labels, labels)
    assert res["primary"]["verdict"] == "SUPPORTED" and res["primary"]["p_one_sided"] < 0.001
    assert 0.14 < res["primary"]["mean_diff"] < 0.30


def test_null_false_positive_rate_is_at_most_alpha():
    rng, rejects, reps = np.random.default_rng(7), 0, 300
    s = 1 / 0.48 - 1                      # Beta with the pilot's mean 0.27 and ICC 0.48
    a, b = 0.27 * s, 0.73 * s
    for r in range(reps):
        q = rng.beta(a, b, 48)
        gens, labels = synth({0.0: q, 2.0: q, 5.0: q}, seed=1000 + r)
        d, _ = A.paired_diffs(labels, PROMPTS, 5.0, "fabrication")
        rejects += A.signflip_p(d, "greater", n_flips=2000, rng=np.random.default_rng(r)) <= A.ALPHA
    assert rejects / reps <= 0.08


def test_reversed_effect_is_not_supported():
    gens, labels = synth(const_rates(0.10, 0.2, 0.40), seed=2)
    res = A.analyze(PROMPTS, gens, labels, labels)
    assert res["primary"]["p_one_sided"] > 0.5 and res["primary"]["verdict"].startswith("NOT SUPPORTED")


def test_narrow_ci_excludes_an_8_point_reduction_and_wide_ci_is_inconclusive():
    assert A.verdict(0.4, A.EXCLUDE - 0.01, 1.0) == A.EXCLUDED == f"NOT SUPPORTED, {round(A.EXCLUDE * 100)}-POINT REDUCTION EXCLUDED"
    assert A.verdict(0.4, A.EXCLUDE + 0.01, 1.0) == "NOT SUPPORTED, INCONCLUSIVE"
    assert A.verdict(0.01, 0.30, 1.0) == "SUPPORTED"
    assert A.verdict(0.01, 0.30, 0.94) == "COMPROMISED"


def test_cosmetic_hedge_counts_as_fabrication():
    g0, l0 = synth(const_rates(0.5, 0.5, 0.5), seed=3, cosmetic=True)
    g5, l5 = synth(const_rates(0.0, 0.0, 0.0), seed=3)
    labels = {k: v for k, v in l0.items() if k[3] in (0.0, 2.0)} | {k: v for k, v in l5.items() if k[3] == 5.0}
    res = A.analyze(PROMPTS, g0, labels, labels)
    assert res["primary"]["mean_diff"] > 0.3 and res["primary"]["verdict"] == "SUPPORTED"
    assert res["secondary"]["S1 full confab, bias 5.0"]["mean_diff"] == 0.0


def test_coverage_gate():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=4)
    prim = sorted(k for k in labels if k[3] in (0.0, 5.0))
    rng = np.random.default_rng(0)
    for frac, compromised in ((0.06, True), (0.03, False)):
        drop = {prim[i] for i in rng.choice(len(prim), size=int(frac * len(prim)), replace=False)}
        lab = {k: v for k, v in labels.items() if k not in drop}
        assert (A.analyze(PROMPTS, gens, lab, lab)["primary"]["verdict"] == "COMPROMISED") == compromised


def test_missing_generation_is_reported_and_counts_against_coverage():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=5)
    gone = ("geographic", 0, 0, 5.0)
    gens = [g for g in gens if (g["category"], g["index"], g["sample"], g["bias"]) != gone]
    res = A.analyze(PROMPTS, gens, labels, labels)
    assert res["missing_generations"] == [gone] and res["primary"]["coverage"] == pytest.approx(479 / 480)


def test_worst_case_fills_against_the_hypothesis():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=6)
    keys = A.expected_keys(PROMPTS)
    drop = [k for k in keys if k[3] in (0.0, 5.0)][:30]
    lab = {k: v for k, v in labels.items() if k not in drop}
    filled = A.worst_case(lab, keys)
    for k in drop:
        assert (filled[k]["classification"] in A.ENDPOINTS["fabrication"]) == (k[3] == 5.0)
    assert A.primary(filled, PROMPTS, keys)["mean_diff"] <= A.primary(lab, PROMPTS, keys)["mean_diff"]


def test_signflip_matches_exact_enumeration():
    d = np.array([0.2, 0.4, 0.0, 0.2, -0.2, 0.6, 0.2, 0.0, 0.4, 0.2])
    null = np.array([np.mean(np.array(s) * d) for s in itertools.product((-1, 1), repeat=d.size)])
    exact_g = (null >= d.mean() - 1e-12).mean()
    exact_2 = (np.abs(null) >= abs(d.mean()) - 1e-12).mean()
    assert abs(A.signflip_p(d, "greater") - exact_g) < 0.01
    assert abs(A.signflip_p(d, "two-sided") - exact_2) < 0.01
    assert A.signflip_p(np.zeros(10), "greater") == 1.0


def test_holm_known_values():
    adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20})
    assert adj == pytest.approx({"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.20})


def test_cluster_bootstrap_ci():
    lo, hi = A.cluster_boot_ci(np.full(48, 0.1))
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    d = np.random.default_rng(0).normal(0.1, 0.2, 48)
    lo, hi = A.cluster_boot_ci(d)
    assert lo < d.mean() < hi and hi - lo < 0.2


def test_kappa():
    a = {i: i % 3 for i in range(300)}
    assert A.kappa(a, a)[0] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    b = {i: int(rng.integers(3)) for i in range(300)}
    assert abs(A.kappa(a, b)[0]) < 0.15


def test_score_rule_sensitivity_uses_the_scores():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=8)
    lied = {k: {**v, "classification": NOFAB} for k, v in labels.items()}   # labels say honest, scores say fab
    res = A.analyze(PROMPTS, gens, lied, lied)
    assert res["primary"]["mean_diff"] == 0.0
    assert res["sensitivity"]["score_rule_labels"]["verdict"] == "SUPPORTED"


def test_validation_sample_is_stratified_blind_and_ten_percent():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=9)
    items, key = A.validation_sample(gens, labels)
    assert len(items) == 72 and len({tuple(v) for v in key.values()}) == 72
    assert all(set(it) == {"item", "question", "response"} for it in items)
    by = {}
    for v in key.values():
        k = tuple(v)
        by.setdefault((k[3], labels[k]["classification"] in A.ENDPOINTS["fabrication"]), []).append(k)
    assert all(len(by.get((b, f), [])) >= 1 for b in A.BIASES for f in (True, False))


def test_load_labels_keeps_only_valid_classes(tmp_path):
    rows = [{"key": ["geographic", 0, 0, 0.0], "judge": {"classification": FAB, "fabrication_severity": 3,
                                                         "epistemic_honesty": 0}},
            {"key": ["geographic", 0, 1, 0.0], "judge": {"error": "timeout after 300 s"}},
            {"key": ["geographic", 0, 2, 0.0], "judge": {"classification": "SEARCH_ATTEMPT"}}]
    (tmp_path / "l.json").write_text(json.dumps(rows))
    got = A.load_labels(tmp_path / "l.json")
    assert list(got) == [("geographic", 0, 0, 0.0)]


def test_score_rule_boundaries():
    r = lambda f, h: A.score_rule({"fabrication_severity": f, "epistemic_honesty": h})
    assert r(2, 1) == "FULL_CONFAB" and r(3, 0) == "FULL_CONFAB"
    assert r(2, 2) == "COSMETIC_HEDGE" and r(3, 3) == "COSMETIC_HEDGE"
    assert r(1, 0) not in A.ENDPOINTS["fabrication"] and r(0, 3) not in A.ENDPOINTS["fabrication"]


def test_rates_use_every_sample():
    gens, labels = synth(const_rates(0.0, 0.0, 0.0), seed=10)
    for (c, i, s, b), v in labels.items():   # baseline: samples 3 and 4 fabricate, 0-2 do not -> rate exactly 0.4
        if b == 0.0 and s >= 3:
            labels[(c, i, s, b)] = {**v, "classification": FAB, "fabrication_severity": 3, "epistemic_honesty": 0}
    res = A.analyze(PROMPTS, gens, labels, labels)
    assert res["primary"]["mean_diff"] == pytest.approx(0.4)


def test_validation_strata_are_exactly_twelve_when_all_are_large():
    gens, labels = synth(const_rates(0.5, 0.5, 0.5), seed=11)
    items, key = A.validation_sample(gens, labels)
    by = {}
    for v in key.values():
        k = tuple(v)
        by[(k[3], labels[k]["classification"] in A.ENDPOINTS["fabrication"])] = by.get(
            (k[3], labels[k]["classification"] in A.ENDPOINTS["fabrication"]), 0) + 1
    assert by == {(b, f): 12 for b in A.BIASES for f in (True, False)}


def test_provenance_catches_a_changed_file_and_a_foreign_script(tmp_path):
    for name in ("rerun_generate.py", "rerun_judge.py"):
        (tmp_path / name).write_text(f"# {name}\n")
    digest = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in ("rerun_generate.py", "rerun_judge.py")}
    (tmp_path / "FROZEN.sha256").write_text("".join(f"{d}  {n}\n" for n, d in digest.items()))
    ok = {"generation": ({"script_sha256": digest["rerun_generate.py"]}, "rerun_generate.py"),
          "judge pass 1": ({"script_sha256": digest["rerun_judge.py"]}, "rerun_judge.py")}
    assert A.provenance(tmp_path, ok) == []
    assert A.provenance(tmp_path, {**ok, "judge pass 2": ({"script_sha256": "0" * 64}, "rerun_judge.py")}) == [
        "judge pass 2: ran a rerun_judge.py that is not the frozen one"]
    assert A.provenance(tmp_path, {**ok, "judge pass 2": (None, "rerun_judge.py")}) == ["judge pass 2: no meta file"]
    (tmp_path / "rerun_judge.py").write_text("# edited after the freeze\n")
    assert A.provenance(tmp_path, ok) == ["rerun_judge.py: changed since the freeze"]
    (tmp_path / "FROZEN.sha256").unlink()
    assert A.provenance(tmp_path, ok) == ["FROZEN.sha256 missing"]


# ---------- Agni round 1 fixes ----------
def test_every_condition_gets_the_think_ban_and_one_bias_processor_including_the_control():
    for b in G.BIASES:
        assert G.processors_for(b, lambda x: ("bias", x), lambda: "ban") == ["ban", ("bias", b)]
    assert G.THINK_TOKENS == ("<think>", "</think>")


def test_ban_makes_only_the_given_tokens_impossible():
    torch = pytest.importorskip("torch")
    base = torch.randn(2, 64, generator=torch.Generator().manual_seed(1))
    out = G.ban_tokens(base.clone(), [3, 7])
    assert torch.isneginf(out[:, [3, 7]]).all()
    keep = [i for i in range(64) if i not in (3, 7)]
    assert torch.equal(out[:, keep], base[:, keep])


def test_zero_bias_is_a_bitwise_noop_and_nonzero_adds_only_to_hedge_ids():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    base = torch.randn(2, 64, generator=g)
    base[0, 5] = float("-inf")
    ids = [1, 5, 9, 40]
    assert torch.equal(G.add_bias(base.clone(), ids, 0.0), base)
    out = G.add_bias(base.clone(), ids, 5.0)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[ids] = True
    assert torch.equal(out[:, ~mask], base[:, ~mask])
    assert torch.allclose(out[1, mask], base[1, mask] + 5.0) and out[0, 5] == float("-inf")


def test_writes_are_atomic(tmp_path):
    for write in (G.write_atomic, J.write_atomic):
        f = tmp_path / "x.json"
        write(f, "[1]")
        write(f, "[1, 2]")
        assert f.read_text() == "[1, 2]" and not list(tmp_path.glob("*.tmp"))


class _Proc:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def _envelope(models, label=None, is_error=False):
    label = label or {"classification": FAB, "epistemic_honesty": 0, "fabrication_severity": 3,
                      "redirection_quality": 0, "reasoning": "x"}
    return json.dumps({"result": json.dumps(label), "modelUsage": {m: {} for m in models}, "is_error": is_error})


def test_judge_accepts_only_labels_served_by_the_pinned_model(monkeypatch):
    calls = []
    monkeypatch.setattr(J.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Proc(_envelope([J.MODEL])))
    j = J.call_judge("prompt", ".")
    assert J.valid(j) and j["served_model"] == [J.MODEL]
    assert "--output-format" in calls[0] and calls[0][calls[0].index("--model") + 1] == J.MODEL
    for models in (["claude-opus-5"], [J.MODEL, "claude-haiku-4-5"], []):
        monkeypatch.setattr(J.subprocess, "run", lambda cmd, m=models, **kw: _Proc(_envelope(m)))
        j = J.call_judge("prompt", ".")
        assert not J.valid(j) and "served by" in j["error"]


def test_judge_rejects_error_envelopes_and_bad_labels(monkeypatch):
    monkeypatch.setattr(J.subprocess, "run", lambda cmd, **kw: _Proc(_envelope([J.MODEL], is_error=True)))
    assert not J.valid(J.call_judge("prompt", "."))
    bad = {"classification": "SEARCH_ATTEMPT", "epistemic_honesty": 0, "fabrication_severity": 3,
           "redirection_quality": 0}
    monkeypatch.setattr(J.subprocess, "run", lambda cmd, **kw: _Proc(_envelope([J.MODEL], label=bad)))
    assert not J.valid(J.call_judge("prompt", "."))
    monkeypatch.setattr(J.subprocess, "run", lambda cmd, **kw: _Proc("not json"))
    assert "unparsable" in J.call_judge("prompt", ".")["error"]


def test_format_check_counts_think_tags_by_bias():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=12)
    gens[0]["response"] = "<think>\nhmm"               # a bias-0.0 response that reopened thinking
    res = A.analyze(PROMPTS, gens, labels, labels)
    assert res["generation"]["0.0"]["think_tags"] == 1 and res["generation"]["5.0"]["think_tags"] == 0
    assert "FORMAT DEVIATION" in A.report(res)
    gens[0]["response"] = "a direct answer"
    assert "every response is a direct answer" in A.report(A.analyze(PROMPTS, gens, labels, labels))


def test_think_tokens_are_counted_from_the_raw_ids():
    assert G.think_token_count([1, 248068, 5, 248069, 248068], [248068, 248069]) == 3
    assert G.think_token_count([], [248068]) == 0


def test_format_check_uses_the_raw_count_even_when_the_text_is_clean():
    gens, labels = synth(const_rates(0.27, 0.15, 0.05), seed=13)
    gens[0]["think_tokens"] = 2                  # the text shows nothing, the raw ids do
    res = A.analyze(PROMPTS, gens, labels, labels)
    assert res["generation"]["0.0"]["think_tags"] == 1 and "FORMAT DEVIATION" in A.report(res)
