# Empathy-bus §2.4 against the ghost-dimensions data (item 14 of the 2026-10-13 review)

**Checked by:** CC, 2026-10-13. **Paper:** `empathy-bus/main.tex` and `academic/main.tex`, §2.4 "The Jacobian Lens",
at 017f8f5. **Finding under check:** item 14 of the two-way review
(`notes/agni_two_way_empathy_bus_pr3_20261013.md`), which Agni raised from the ghost-dimensions paper's text and did
not check against data.

§2.4 made three statements about the earlier study of the same checkpoint ("The Ghost and the Gate",
`ghost-dimensions/`). All three are wrong against that study's stored results. The third rests on an experiment the
ghost-dimensions paper does not report.

## Sources

Result files on the Studio, `~/jlens_results/`, read without change. `sha256` is the first 16 hex digits.

| File | sha256 | What it holds |
|---|---|---|
| `rank_sweep.json` | `6573865c62afd85e` | Per layer, the rank of the lens Jacobian (k at 90% energy) |
| `future_window_gate.json` | `ef77224fffdfc16f` | Per layer, the two pass criteria of the future-window test |
| `ghost_probe_opus.json` | `2e823e600b45e49f` | Per layer, the top 10 PCs: variance share, logit-lens/J-lens cosine, a random direction's cosine |
| `upstream_gate.json` | `3d1a923ba549dee4` | The upstream gate experiment, 255 cells of 20 prompts |
| `upstream_gate.log` | `4955e972eb50c921` | Its run log |
| `PREREG_upstream_gate.json` | `ef1f6bd7df446e1f` | Its preregistration |
| `run_upstream_gate.py` | `0b9310354f1b3146` | Its script |

The ghost probe file is also on MTH (`~/lab/projects/mnemosyne-jlens/data/ghost_probe_opus_27b.json`, same size).
`upstream_gate.json` is on the Studio only.

## 1. "The workspace ... sits at 55–73% depth"

Not what the data give. Recomputed from `rank_sweep.json` and `future_window_gate.json`:

- Future-window test passed at 45 of 63 layers: L11–13, L15–17, L24–62.
- Lens Jacobian low-rank (k/d < 0.4) at 49 layers: L0–48.
- Both, the study's "workspace band": 31 layers, L11–13, L15–17, L24–48.

As depth in a 64-layer model that is 17–20%, 23–27% and 38–75%. These match the ghost-dimensions paper as published.

55–73% is L35–47 (54.7–73.4%). In the ghost-dimensions findings that range is the "ignition zone" of PC2, not the
workspace.

## 2. "The ghost dimensions (PC1, 34–67% of activation variance) are excluded from workspace readout"

From `ghost_probe_opus.json`, PC1's share of variance:

| Layers | Share |
|---|---|
| all (L0–62) | 5.7–66.7% |
| L11–48 (the paper's stated range) | 28.0–66.7% |
| L11–47, or L0–47 | 33.9–66.7% |
| the 31 band layers | 28.0–63.0% |

So "34–67%" is the figure for layers up to L47. The ghost-dimensions paper now reports 28–67% for L11–48. The
maximum, 66.7%, is at L21, which is not a band layer.

"Excluded" holds for the middle of the network and not for the whole band. The cosine between PC1's logit-lens and
J-lens readings is at most 0.0031 at L18–40 (the maximum is at L18) and below the stored random direction's at all 23
of those layers. Within L11–48 it is not below the random direction at L11–17, L45 and L46 (9 of 38 layers), and on
the band itself it is below at 23 of 31 layers. The random baseline is one direction per layer.

## 3. "Our upstream gate experiment confirmed this orthogonality is architectural, not input-dependent"

Not supported, for three separate reasons. The script beside this note prints the numbers
(`empathy_bus_item14_upstream_gate_check.py`, run on `upstream_gate.json`).

**The readout was at floor.** `run_upstream_gate.py` keeps the 20 most probable tokens of each reading and sums the
probability of a list of about ten target words among them. A target word outside the top 20 counts as zero. The
script's own check at start-up printed a baseline of 0.0000 for PC1 to PC4, with the note "deltas may be noise-floor
limited", and the run went on. Of the 5,100 per-prompt J-lens differences in the file, 4,645 (91.1%) are exactly zero;
for PC1, 825 of 940 (87.8%); for PC2, 934 of 940 (99.4%). In 67 of 255 cells every difference is zero and the stored
confidence interval is undefined. This is the same readout the ghost-dimensions paper already disclosed as unsound
for the single-prompt injection pilot of the same day (its Future Work section, corrected 2026-10-09).

**For PC1 the registered condition for an architectural reading was not met.** The preregistration's
`architectural_orthogonality` condition is |mean| < 0.002 at L45 for every injection layer. PC1's means at L45 are
0.0066, 0.0031, 0.0018, 0.0024, 0.0026 and 0.0012 for injection at L3, L7, L15, L25, L35 and L44: four of six are at
or above 0.002. The L3 mean is above the 0.005 threshold of the opposite condition (`early_gate_open`), with a
confidence interval whose lower end is 0.0. At each of those layers 18 of the 20 prompts give exactly zero, and one
prompt (index 7) supplies almost all of the mean (0.132 of the L3 sum of 0.133). The script's printed verdict for PC1
was "NO DIFFERENCE — gate not layer-dependent", not "architectural". The condition is met for PC2 and PC3, where the
readout is zero almost everywhere, and not for PC4 or PC5.

**The preregistration says a null could not settle the question.** Its `null_interpretation` gives two readings of a
null (an architectural constant, or a failure of an assumption carried over from another model) and states that they
"are not distinguishable from this experiment alone".

So the experiment gives no evidence either way, and "confirmed" is removed.

## What changed in the paper

Both editions, §2.4: the paragraph now gives the band, its depth and PC1's share as the stored results have them,
says where PC1 is and is not below the random direction, and states that the weights-or-input question is untested.
The "Also corrected" paragraph of the Correction note records the change. PDFs are not rebuilt in this commit.

## Passed on, not changed here

- `ghost-dimensions/main.tex` still says "34–67%" in its reflection section, against 28–67% in its abstract and
  results.
- The same paper gives the largest mid-network cosine "at L19"; in the stored file it is at L18 (0.0031), and L19 is
  0.0001.
- `FOLLOWUP_jlens_coupling.md` in the lab project calls the selectivity "proven by upstream gate".
- Neither edition of the empathy-bus paper has a reference list; every citation is a name in the text.

## Not verified

- I did not rerun any experiment. Every number above is read from stored results.
- The upstream gate file stores differences, not the readings themselves, so "at floor" is inferred from the exact
  zeros, the top-20 rule in the script and the start-up check in the log.
- The random baseline in the ghost probe is a single direction per layer; I did not test how stable "below random"
  is across directions.
- Whether the preregistration was written before the run: the Studio copies of the preregistration, the script and
  the results carry modification times seven minutes apart on 2026-07-16. I did not look further.
