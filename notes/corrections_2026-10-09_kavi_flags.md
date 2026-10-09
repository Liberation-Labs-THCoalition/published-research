# Correction round 5, 2026-10-09: six of Kavi's flags in papers Nexus leads (branch `nexus/kavi-flags-20261009`)

Lyra sent six of Kavi's 161 flags on 2026-10-09 at 15:46 (qwen3.8-tagged, every quoted line re-checked by Lyra at
master `d528143`). Nexus checked each against the code and the raw data, and wrote one verdict per flag in
`~/lab/projects/kavi-flags-nexus/VERDICTS.md` on MTH, with the raw files pulled from the Studio in `evidence/`.

**Review status: one reader plus a sample.** Lyra read all six verdicts at 16:16 and checked #157's instrument in the
code. They took the external facts on Nexus's reading: jerrickhoang's k/d, the Studio alignment profile, and the
oracle-tiny adapter config. Not a full second read.

**Conventions kept from round 4:** both editions of every edited paper were changed; the one md twin among them
(`empathy-bus/paper.md`) was retired, not edited; internal working documents were not corrected (Thomas, 2026-10-09:
"the internal docs don't need correction"). Every change to a published claim carries a dated correction note.

## Per paper

**empathy-bus** (`c80e8da`; both `.tex`, both PDFs; `paper.md` deleted)
- **#49.** Both experiments ran on the same checkpoint (`emotional_dynamics.py:27`, `coupling_test.json:3`: the
  Jackrong distill). "Replication on the same model checkpoint would close this gap", "(predicted but not yet
  demonstrated on the same model)" and "replicate ... on the same checkpoint. This experiment is pre-registered and
  queued" now say what is really untested: whether the dynamics are carried by the shared subspace (temporal probes
  L3-L15; coupling test L15-L45). The "pre-registered and queued" claim is dropped, because no such prereg could be
  found.
- **Fixed in the same passages,** from `empathy-bus.agni_style.json`:
  - #2: the scar is now g = +6.44 (uncorrected d = +8.05), in all five places;
  - #3: the overshoot is now g = -2.35 (uncorrected d = -2.93, p = 0.023, n = 3);
  - #4: the full model string at first mention, plus a Distillation limitation.
  One footnote records the old values. Retiring `paper.md` resolves register :774 (TWIN_DESYNC) and :493.
- **Still open in that review:** #1, #5, #8 (no bibliography), and #14-#26.

**emotional-trajectory-paper** (`5451899`; both `.tex`, both PDFs)
- **#51.** "A Qwen2 model" becomes oracle-tiny: Qwen2.5-0.5B fine-tuned in our Oracle project, with a
  continued-pretraining LoRA merged in. This is in the abstract, Methods and the Results header, plus a Fine-tuned
  checkpoint limitation: which later Oracle stages are merged in is not recorded, and the base model wasn't measured.
- **Thomas decides the naming** (Lyra votes to name it). If he declines, use "an internal continued-pretraining
  variant of Qwen2.5-0.5B" in those four places.

**ghost-dimensions** (`3d6dd69`; both `.tex`, both PDFs)
- **#157.** "Designed but not performed" and "designed and staged" become: a single-prompt pilot was run on 07-16,
  its readout metric was unsound, and so it gives no evidence either way. The metric's three defects: only the top 20
  tokens counted; a target list included the prompt's own words; duplicate tokens collapsed. The pilot's
  "two mechanisms / active filtering at L45" reading is deliberately not imported.
- **#131, this paper's part.** The PC6 footnote's "0.44x" divided the mean-pooled 0.032 by the position-specific
  draw (0.072). Each value is now compared with its own baseline: 0.585 vs 0.072 is 8.1x; 0.032 vs the mean-pooled
  0.085 +/- 0.065 is within its spread.
- **Not edited:** `FINDINGS.md` and `PAPER_DRAFT_v2-v4.md`. These are internal. They hold #103 (Gemma-2 as "pure
  full-attention") and #131's "killed"/"falsified" rows. main.tex already uses Qwen3-8B, and already says
  Inconclusive and Scale artifact.

**null-swarm-paper** (`c41b6ed`; both `.tex`, both PDFs)
- **#128, Case 1.2.** The near-identity regime is the LATE layers (transport from close to the output; k/d = 0.794
  one layer before it), not the early ones, and the baseline is k/d, not ~1.0.
- **#128, Case 6.1.** The kill is restated from the saved adjacent cosines: PC4 turns at L30-L33, changes gradually
  from L33 to L47 (>= 0.92), and breaks at L47->48 (0.12). The unreproducible direct cosines 0.19 and 0.16 are
  withdrawn, along with "independent dimensions at each depth".
- **#131, Case 2.3.** "~7x random" divided the position-specific value by the mean-pooled baseline. It is now 8.1x the
  position-specific draw.

## Builds

Every edited PDF was rebuilt with pdflatex (with bibtex where the paper has a `.bib`):
- 0 undefined references in every build;
- page counts unchanged except null-swarm, 11 to 12 (three footnotes);
- overfull boxes as before (empathy-bus 4, emotional-trajectory 2, ghost-dimensions 2, null-swarm 5), apart from one
  new box in the empathy-bus abstract from the long model string, which was fixed with break points.

## Found on the way, not flag fixes

- My July note and memory had recorded the 07-16 injection pilot's "active filtering at L45" as a finding. Both now
  carry dated corrections (`mnemosyne-jlens/injection_analysis.md`; memory `project_session_20260716_handoff`).
- While editing, VERDICTS.md's "PC4 >= 0.96 over L35-L47" turned out to be wrong: the minimum is 0.927. It's corrected
  there, and the paper text uses the recomputed values.
- **empathy-bus Limitation 2** says a follow-up design "is pre-registered". That wasn't checked in this round.
