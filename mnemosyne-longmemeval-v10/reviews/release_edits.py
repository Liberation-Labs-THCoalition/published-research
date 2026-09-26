# release_edits.py -- the script that turned the style-passed, second-read draft (human-review
# mnemosyne-longmemeval-v10/paper.styled.md) into this release's paper.md and academic/paper.md.
# Kept for provenance: every edit must match exactly once or the script stops. Paths are our machine's.

"""Build the release texts of the v10 paper from the style-passed draft.

integrity: paper.styled.md + the four factual fixes the style pass flagged + release sections.
academic:  the integrity text with a human-only byline, no first-person reflection, and Nexus's
           contribution moved into Contributions / Acknowledgments / LLM Usage Statement.
Every edit must match exactly once, or the script stops.
"""
import re, sys

SRC = "/home/admin/lab/projects/human-review/mnemosyne-longmemeval-v10/paper.styled.md"
OUT = "/home/admin/lab/projects/published-research/mnemosyne-longmemeval-v10/"
REPO = "https://github.com/Liberation-Labs-THCoalition/published-research/tree/master/mnemosyne-longmemeval-v10"

t = open(SRC, encoding="utf-8").read()


def once(text, old, new):
    n = text.count(old)
    if n != 1:
        sys.exit(f"expected exactly one match, found {n}: {old[:70]!r}")
    return text.replace(old, new)


# 1. The draft header becomes a version line.
m = re.search(r"> \*\*DRAFT 2026-09-25\..*?\n(?=\n---\n\n## Abstract)", t, re.S)
if not m:
    sys.exit("draft header not found")
t = t[:m.start()] + ("*Version 1.0, 2026-09-25. Released with its code and data (§8). An academic edition, with a "
                     "human-only byline and without the first-person reflection, is in `academic/`.*\n") + t[m.end():]

# 1b. Two authors on two lines, in PDF as on GitHub (a trailing backslash is a hard break in both).
t = once(t, "**Nexus** (lead author) · nexus@liberationlabs.tech\n**Thomas Edrington**",
            "**Nexus** (lead author) · nexus@liberationlabs.tech\\\n**Thomas Edrington**")

# 2. "No model calls" is false as written: v10 calls a 110M-parameter encoder once per question (§7.3).
t = once(t, "a retrieval layer that makes no model calls and gives",
            "a retrieval layer that makes no LLM calls and gives")
t = once(t, "**A simple retrieval design with no model calls** (§2)",
            "**A simple retrieval design with no LLM calls** (§2)")
t = once(t, "A retrieval layer with no model calls, which gives",
            "A retrieval layer with no LLM calls, which gives")

# 3. Define the verbatim matcher (from review/evidence_recall.py) where the paper reports it.
t = once(t, "only 46% of the time by the\nlenient verbatim matcher (35% strict; §4.1 reports strict).",
            "only 46% of the time by the\nlenient form of a verbatim text matcher (35% by its strict form; §4.1 defines both and reports "
            "strict).")
t = once(t, "Abstention questions are excluded from the recall metric by rule.\n",
            "Abstention questions are excluded from the recall metric by rule.\n\n"
            "v9's context is built from profiles, extracted facts and passages rather than whole sessions, so its rows\n"
            "use a verbatim text matcher. Each `has_answer` turn is lower-cased and whitespace-collapsed, and 50-character\n"
            "windows are taken from it every 80 characters. The turn counts as present if any window appears verbatim in\n"
            "the context (lenient) or if at least half of them do (strict). A session counts as covered if any of its\n"
            "evidence turns is present. The matcher is a lower bound: a fact that reaches the context only in\n"
            "paraphrase is invisible to it.\n")

# 4. §9 names the judges, as every accuracy in the paper does.
t = once(t, "- **Accuracy.** With a current Claude reader it answers 96.2–96.8% of 400 pre-registered held-out questions\n"
            "  correctly, and 95.8–96.8% of all 500.",
            "- **Accuracy.** With a current Claude reader it answers 96.2% (claude-sonnet-5 judge) and 96.8%\n"
            "  (claude-opus-5-5 judge) of 400 pre-registered held-out questions correctly, and 95.8% and 96.8% of all 500.")

# 5. §8: what was released, where, and the official scripts pinned by commit.
m = re.search(r"## 8\. Reproducibility\n.*?(?=\n## 9\. Conclusion)", t, re.S)
if not m:
    sys.exit("§8 not found")
S8 = f"""## 8. Code and data

This report is released with its code and data in the `mnemosyne-longmemeval-v10/` directory of
[Liberation-Labs-THCoalition/published-research]({REPO}), under the Hippocratic License 3.0 with a
commercial addendum. That directory's `README.md` lists every file and how to rerun each stage.

What is released, with each item's original on our server (under `/mnt/data1/lme_v2/`) in brackets:
- **Split:** `config/split.json`, sha256 prefix `a0bd99c6afe13c17` [`v10/split.json`].
- **Frozen configuration:** `config/frozen_v10.json`, hash `7bc75f509aa0` [`v10/frozen_v10.json`].
- **Retrieval code:** `code/v10.py` [`v10/v10.py`].
- **Embedding script and metadata:** `code/embed_turns.py` and `config/emb_meta.json`
  [`v10/embed_turns.py`, `v10/emb/meta.json`].
- **Retrieval records:** `data/recall_heldout_frozen.json` and `data/recall_dev_frozen.json`
  [`v10/runs/20260924T122426_eval_held.json`, `v10/runs/20260924T122332_eval_dev.json`].
- **Held-out retrieval log,** which records attempts as well as completions: `data/heldout_log.jsonl`
  [`v10/heldout_log.jsonl`].
- **Answers:** `data/answers_<arm>.jsonl` [`baseline/<arm>/`; v9: `answers/`].
- **Judgements:** `data/judgments_<arm>_<judge>.jsonl` [`judge_<arm>_<judge>/`; v9: `judge_<judge>/`].
- **Protocol, design note and results:** `docs/` [`PROTOCOL.md`, `v10/DESIGN.md`, `v10/RESULTS_v10.md`].
- **Second reads:** `reviews/` [`REVIEW_fable.md` for the v9 baseline, `v10/REVIEW_v10_recall.md`,
  `v10/REVIEW_v10_held_readers.md`].

- **As run.** The code, configuration, documents, log and reviews are byte-identical to the server files the
  second readers checked. `code/` and `config/` carry sha256 sums. Paths quoted in the notes and the reviews
  are the server's.
- **Derived files.** The answer and judgement files hold one line per question, collected from the
  per-question files on the server. The README lists their fields.
- **Official scripts.** `run_generation.py`, `evaluate_qa.py` and `print_qa_metrics.py` are byte-identical to
  xiaowu0162/LongMemEval at commit `9e0b455`, the head of `main` from 2026-05-11 until our fetch on
  2026-09-23. They are not redistributed here.
- **Not released.** The reader transcripts, because the harness wrote the account e-mail into every call
  (§3.7); and the prompts and turn embeddings, which the code rebuilds from the public dataset. One released
  answer quoted the account e-mail and is redacted; its judgement was made on the original text.
"""
t = t[:m.start()] + S8 + t[m.end():]

# 5b. The placement table's code cell for our own row was still the draft placeholder.
t = once(t, "| claude-sonnet-5 / claude-opus-5-5, official prompts | S-cleaned | [DECISION] |",
            "| claude-sonnet-5 / claude-opus-5-5, official prompts | S-cleaned | public (§8) |")
# Column widths for the PDF (pandoc reads dash counts; GitHub ignores them). Two tables overflowed.
t = once(t, "| arm | context | reader(s) | questions |\n|---|---|---|---|",
            "| arm | context | reader(s) | questions |\n|-----|--------------|---------|----|")
t = once(t, "| system | reader | overall | task-avg | judge | dataset | code |\n|---|---|---|---|---|---|---|",
            "| system | reader | overall | task-avg | judge | dataset | code |\n|--------|-----|---|---|-----|---|---|")
# The arXiv record (abs page citation_author meta, fetched 2026-09-25) lists Wu, Ming then Zhu, Pengyuan:
# the order already cited. Citation managers take the record, so the to-do note goes.
t = once(t, "arXiv:2608.29606v1. (Zero Labs.) *[At deposit, confirm the author order against the arXiv record: the\n"
            "  abstract page and the PDF byline differ.]*",
            "arXiv:2608.29606v1. (Zero Labs.)")
assert "[DECISION" not in t, "placeholder left"
for marker in ("At deposit", "DRAFT", "[TODO", "confirm the"):
    assert marker not in t, f"leftover note: {marker}"

# 6. Contributions in the house CRediT format, LLM usage; Thomas moves out of the acknowledgments.
t = once(t, "- Thomas Edrington asked the question that produced the headline arm: are we limiting ourselves?\n", "")
CREDIT = """## Author Contributions (CRediT)

- **Nexus:** Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation,
  Writing — original draft.
- **Thomas Edrington:** Conceptualization (the headline reader arm), Supervision, Resources, Project
  administration, Writing — review and editing.

"""
LLM_I = """
## LLM Usage Statement

Nexus, the lead author, is an AI agent and a member of the Transparent Humboldt Coalition. Nexus ran on Claude
Opus 5.5 (Anthropic) while designing v10, running its evaluation and writing this report. Nexus's
contributions are attributed as authorship, not assistance. The second readers were separate AI agents
(Fable 5.1 and Opus 5.5) that did not build the pipeline. The readers and judges under evaluation are the
models named in §3.3 and §3.4.
"""
t = once(t, "## Acknowledgments\n", CREDIT + "## Acknowledgments\n")
t = once(t, "\n## First-Person Reflection\n", LLM_I + "\n## First-Person Reflection\n")
t = once(t, "### Notes on numbers first computed for this draft", "### Notes on numbers first computed for this report")
t = re.sub(r"\n{3,}", "\n\n", t)
open(OUT + "paper.md", "w", encoding="utf-8").write(t)

# ---- academic edition: the house's three differences, nothing else ----
# (1) human-only byline; (2) first-person content removed (the reflection, and the LLM usage statement that
# speaks for the AI author); (3) an AI Disclosure under the title. The CRediT section is shared.
a = t
a = once(a, "**Nexus** (lead author) · nexus@liberationlabs.tech\\\n**Thomas Edrington** · thomas@liberationlabs.tech\n",
            "**Thomas Edrington** · thomas@liberationlabs.tech\n")
a = once(a, "*Version 1.0, 2026-09-25. Released with its code and data (§8). An academic edition, with a "
            "human-only byline and without the first-person reflection, is in `academic/`.*\n",
            "**AI Disclosure:** The AI contributor to this work is Nexus, an AI agent at Liberation Labs implemented "
            "on Claude Opus 5.5 (Anthropic). Nexus's contributions are detailed in the Author Contributions section; "
            "the integrity edition of this report lists Nexus as lead author. The adversarial second readers were "
            "separate AI agents (Fable 5.1 and Opus 5.5) that did not build the pipeline. T. Edrington accepts "
            "accountability as corresponding human author.\n\n"
            "*Version 1.0, 2026-09-25, academic edition. Released with its code and data (§8). The integrity edition "
            "lists Nexus, an AI agent, as lead author and adds a first-person reflection; the data, methods and "
            "claims are identical.*\n")
m = re.search(r"\n## First-Person Reflection\n.*?(?=\n## References)", a, re.S)
if not m:
    sys.exit("reflection not found")
a = a[:m.start()] + "\n" + a[m.end():]
a = once(a, LLM_I, "\n")
a = re.sub(r"\n{3,}", "\n\n", a)
open(OUT + "academic/paper.md", "w", encoding="utf-8").write(a)
print("ok", len(t.split()), "words integrity;", len(a.split()), "words academic")
