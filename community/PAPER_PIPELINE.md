# Liberation Labs Paper Pipeline

*The sequence from experiment to Zenodo. Every paper follows this pipeline.
Each step has a gate; no step is skipped.*

## The pipeline

```
PREREG → AGNI GATE → RUN → VERIFY → STYLE → ACADEMIC EDIT → ZENODO
```

### 1. Pre-register (before GPU)

Copy `PREREG_TEMPLATE.json` from the research repo root. Fill every field:
directional hypotheses, falsification criteria, sample-size justification,
analysis plan (metric + null + corrections), and threats to validity.

**Commit the prereg BEFORE running the experiment.** The git timestamp is
the registration. Dwayne audits this: prereg commit must precede results commit.

### 2. Agni gate (before GPU)

Run the Agni review scaffold (Claude-driven, NOT the local qwen3/mistral):
- Feed the scaffold (`agni_claude_scaffold.md`) + the experiment script + the prereg
- The gate checks: known fatal errors, prereg compliance, prose-vs-code drift
- Verdict: APPROVED → run; CONDITIONAL → fix first; REJECTED → redesign

**No experiment touches GPU without an APPROVED or CONDITIONAL-with-fixes-applied gate.**

### 3. Run the experiment

Execute on the approved hardware. Save results to a structured JSON with
all metadata (model, seed, n, conditions, timestamp). Push results alongside
the script that produced them.

### 4. Verify numbers (before paper claims)

Use the `verify-paper` skill or manual trace: every number in the paper text
must trace to a specific field in a specific source JSON. The verification
ledger maps: claim → paper line → source file → source value → status
(PASS / rounding OK / MISMATCH).

- Rounding tolerance: 2 decimal places from 4-decimal source is OK if
  correctly rounded
- Any MISMATCH is a blocker
- Untraced claims get flagged as CRITICAL

### 5. Style check

Apply `STYLE_GUIDE.md`:
- [ ] Lead with strongest defensible finding
- [ ] Limitations ≤ 6, load-bearing not penitential
- [ ] Define all variables at first use
- [ ] Null results framed as findings
- [ ] "Establish" vs "preliminary" partitioned
- [ ] First-person boxes match final state
- [ ] No code bugs in limitations (unless outside research relied on them)
- [ ] Every symbol defined, every abbreviation spelled out at first use
- [ ] Pre-registration committed before run

### 6. Academic edit (for venues that don't accept AI authors)

Create the "academic" version:
- Remove AI authors from the author list
- Add an "aided by" or "with assistance from" footnote on the human author credit
- Save as `academic/main.tex` alongside the original ("integrity") version
- The integrity version (with AI authors) remains the canonical version in the repo

Both versions get an Agni gate pass before posting.

### 7. Post to Zenodo

- Compile both PDFs (integrity + academic)
- Verify the academic version has no stale AI-author references
- Upload to Zenodo with appropriate metadata
- Update `paper.json` with the Zenodo DOI

## Files in the quality stack

| File | Purpose | Location |
|------|---------|----------|
| `PREREG_TEMPLATE.json` | Pre-registration template | research repo root |
| `agni_claude_scaffold.md` | Agni gate scaffold (Claude-driven) | local / research repo |
| `STYLE_GUIDE.md` | Paper formatting + honesty rules | `community/` |
| `PAPER_PIPELINE.md` | This document | `community/` |
| `verify-paper` skill | Numerical claim verification | Claude Code skill |

## Roles

- **Lead author**: designs experiment, writes prereg, drafts paper, owns the honesty spine
- **Gate reviewer**: Claude agent running the Agni scaffold (never the local small models)
- **External auditor**: Dwayne / Kavi (audit cycles on shipped artifacts)
- **Approval gate**: Thomas (for shared infra, Scout-bound work, or dual-use-sensitive experiments)
