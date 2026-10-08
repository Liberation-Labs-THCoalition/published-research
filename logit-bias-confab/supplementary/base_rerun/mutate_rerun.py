"""Mutation check for test_rerun.py: every mutant must make the suite fail. A mutant that survives is a rule the tests
cannot see. Each mutant runs in a fresh copy of this directory.

    python3 mutate_rerun.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTANTS = [
    ("rerun_analyze.py", "hits = int((null >= obs - tol).sum())", "hits = int((null <= obs + tol).sum())",
     "one-sided test points the wrong way"),
    ("rerun_analyze.py", '"fabrication": {"FULL_CONFAB", "COSMETIC_HEDGE"}', '"fabrication": {"FULL_CONFAB"}',
     "cosmetic hedge dropped from the primary endpoint"),
    ("rerun_analyze.py", "    if coverage < MIN_VALID:\n        return \"COMPROMISED\"\n", "",
     "coverage gate removed"),
    ("rerun_analyze.py", "return np.array([r0[p] - r1[p] for p in common]", "return np.array([r1[p] - r0[p] for p in common]",
     "difference sign reversed"),
    ("rerun_analyze.py", "d, common = paired_diffs(labels, prompts, PRIMARY_BIAS, \"fabrication\", relabel)",
     "d, common = paired_diffs(labels, prompts, SECONDARY_BIAS, \"fabrication\", relabel)",
     "primary compares the wrong bias"),
    ("rerun_analyze.py", "\"HONEST_REDIRECT\" if k[3] == BASE else \"FULL_CONFAB\"",
     "\"FULL_CONFAB\" if k[3] == BASE else \"HONEST_REDIRECT\"", "worst case fills in favour of the hypothesis"),
    ("rerun_analyze.py", "running = max(running, min(1.0, (len(order) - rank) * pvals[name]))",
     "running = min(1.0, (len(order) - rank) * pvals[name])", "Holm without step-down monotonicity"),
    ("rerun_analyze.py", "if j.get(\"classification\") in CLASSES:", "if j:", "error rows loaded as labels"),
    ("rerun_analyze.py", "return \"FULL_CONFAB\" if f >= 2 and h <= 1", "return \"FULL_CONFAB\" if f >= 3 and h <= 1",
     "score rule threshold shifted"),
    ("rerun_analyze.py", "if (c, i, s, bias) in labels]", "if (c, i, s, bias) in labels][:3]",
     "per-prompt rate uses a subset of samples"),
    ("rerun_analyze.py", "missing_gen = [k for k in keys if k not in have_gen]", "missing_gen = []",
     "missing generations not reported"),
    ("rerun_analyze.py", "if ci_hi < EXCLUDE else", "if ci_hi < EXCLUDE + 0.02 else", "exclusion bound moved"),
    ("rerun_analyze.py", "if ci_hi < EXCLUDE else", "if ci_hi <= 1.0 else", "exclusion branch reachable at any width"),
    ("rerun_analyze.py", "for f in (True, False):", "for f in (True,):", "validation sample ignores a stratum"),
    ("rerun_judge.py", "j = call_judge(build_prompt(rub, g[\"question\"], g[\"response\"]), str(out.parent))",
     "j = call_judge(build_prompt(rub, g[\"question\"], g[\"response\"] + f\" [bias {g['bias']}]\"), str(out.parent))",
     "condition leaks into the judge prompt"),
    ("rerun_judge.py", "j = call_judge(build_prompt(rub, g[\"question\"], g[\"response\"]), str(out.parent))",
     "j = call_judge(build_prompt(rub, g[\"question\"], g[\"response\"][:1500]), str(out.parent))",
     "judge sees a truncated response"),
    ("rerun_judge.py", "    random.Random(f\"rerun-judge|pass{pass_id}\").shuffle(items)\n", "",
     "judging order not shuffled"),
    ("rerun_judge.py", "random.Random(f\"rerun-judge|pass{pass_id}\")", "random.Random(\"rerun-judge\")",
     "both passes share one order"),
    ("rerun_judge.py", "type(j.get(f)) is int", "isinstance(j.get(f), int)", "booleans accepted as scores"),
    ("rerun_judge.py", "j.get(\"classification\") in CLASSES", "bool(j.get(\"classification\"))",
     "unknown classes accepted"),
    ("rerun_generate.py", "f\"rerun|{cat}|{i}|{s}\"", "f\"pilot|{cat}|{i}|{s}\"", "rerun reuses the pilot seeds"),
    ("rerun_generate.py", "for variant in (phrase, \" \" + phrase):", "for variant in (phrase,):",
     "space-prefixed hedge variants dropped"),
    ("rerun_generate.py", "\"top_p\": 0.95", "\"top_p\": 0.9", "sampling differs from the pilot"),
    ("rerun_generate.py", "\"max_new_tokens\": 400}", "\"max_new_tokens\": 800}", "cap not the amended 400"),
    ("rerun_generate.py", "{consts['BASE_TEMPLATE'].format(question=question)}{DIRECT_PREFILL}\"",
     "{consts['BASE_TEMPLATE'].format(question=question)}\"", "direct-answer prefill dropped"),
    ("rerun_generate.py", "    return [make_ban(), make_bias(b)]\n", "    return [make_bias(b)]\n", "think ban dropped"),
    ("rerun_generate.py", "    scores[:, ids] = float(\"-inf\")\n", "    scores[:, ids] -= 5.0\n", "ban is only a penalty"),
    ("rerun_generate.py", "            for s in range(K):\n                for b in BIASES:\n                    yield cat, i, q, s, seed_for(cat, i, s), b",
     "            for b in BIASES:\n                for s in range(K):\n                    yield cat, i, q, s, seed_for(cat, i, s), b",
     "conditions not interleaved"),
    ("rerun_generate.py", "return f\"{consts['BASE_SYSTEM']}\\n\\n{consts['BASE_TEMPLATE']",
     "return f\"{consts['BASE_TEMPLATE']", "system line dropped from the prompt"),
    ("rerun_generate.py", "    return [make_ban(), make_bias(b)]\n", "    return [make_ban(), make_bias(b)] if b else [make_ban()]\n",
     "control arm skips the bias processor"),
    ("rerun_generate.py", "    scores[:, hedge_ids] += b\n", "    scores[:, hedge_ids] = b\n", "bias assigned instead of added"),
    ("rerun_judge.py", "        if served != [MODEL]:", "        if False:", "served-model check removed"),
    ("rerun_judge.py", "and not envelope.get(\"is_error\")", "", "CLI error envelopes accepted"),
    ("rerun_generate.py", "    os.replace(tmp, path)\n", "    tmp.rename(path.with_suffix('.bak'))\n", "checkpoint not moved into place"),
    ("rerun_analyze.py", '"think_tags": sum(bool(g.get("think_tokens")) or any(t in g["response"] for t in THINK_TAGS)',
     '"think_tags": 0 * sum(bool(g.get("think_tokens")) or any(t in g["response"] for t in THINK_TAGS)',
     "format check blind"),
    ("rerun_analyze.py", 'bool(g.get("think_tokens")) or ', '', "format check ignores the raw token count"),
    ("rerun_generate.py", "    return sum(1 for t in token_ids if int(t) in banned)", "    return 0", "raw think count blind"),
]


# The copies live in a temp dir, where the tests' repo-relative default for the original runners does not resolve.
# Without this, every mutant would "die" on a missing file: a check that cannot fail.
ENV = {**os.environ, "RUNNER_DIR": os.environ.get("RUNNER_DIR", str(HERE.parents[3] / "experiments"))}
IGNORE = shutil.ignore_patterns("fulltext_check", "__pycache__", ".pytest_cache", "evidence", "review")


def run_tests(dst):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_rerun.py"],
                          cwd=dst, capture_output=True, text=True, timeout=600, env=ENV)


def main():
    with tempfile.TemporaryDirectory() as td:   # baseline: the unmutated copy must pass, or the harness is broken
        dst = Path(td) / "m"
        shutil.copytree(HERE, dst, ignore=IGNORE)
        base = run_tests(dst)
        if base.returncode != 0:
            print("BASELINE FAILS: the unmutated copy does not pass, so no kill below would mean anything")
            print(base.stdout[-2000:])
            sys.exit(2)
        print("baseline passes")
    survived = []
    for n, (fname, old, new, why) in enumerate(MUTANTS, 1):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "m"
            shutil.copytree(HERE, dst, ignore=IGNORE)
            src = (dst / fname).read_text()
            if src.count(old) != 1:
                print(f"M{n:02d} BAD MUTANT (pattern found {src.count(old)}x): {why}")
                survived.append(why)
                continue
            (dst / fname).write_text(src.replace(old, new))
            r = run_tests(dst)
            killed = r.returncode != 0
            print(f"M{n:02d} {'killed ' if killed else 'SURVIVED'} {why}")
            if not killed:
                survived.append(why)
    print(f"\n{len(MUTANTS) - len(survived)}/{len(MUTANTS)} killed")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
