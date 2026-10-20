# R1 / R2 / R7 results: user-model v2 correction run

**Run:** Starship, encode 2026-10-19 23:05–23:22 BST, analysis 23:22–23:27 BST. Results `results_20261019/um_v2_results.json`
(sha256 `0aefed36…`). Pre-registration `prereg_um_v2_R1R2.json` as signed off (notes §10.7); sealed prediction
`R1R2_PREDICTION.md` (sha256 `89a578e1…`, unchanged, checked by both scripts).
**Written:** 2026-10-19 15:30 PDT by Lyra, after reading the results file and the registered rules (prereg §5), before any paper edit.

## Integrity: all clean
- **No overrides.** `meta.args` holds the registered values (n_perm 1000, n_perm_r7 5000, n_boot 10000, seed 42,
  sha verification on).
- **Determinism:** worst per-layer relative RMS across the three repeats of trial 0 is **0.0**.
- **R0 reproduced** exactly (5 folds against the file: agreement 1.0).
- **The v2-era witness:** v1, recomputed on CPU from today's encode, gives **470/900, May's exact count**.
- **Pipeline control** passed (topic 0.83). **No confound:** covariates alone 0.051; FWL diagnostics within limits.
- **Fingerprint against May's bridge weights:** IN_BAND (reported, not gating).
- **One operational fault:** the first encode hung at weight loading for 59 minutes and was killed. It could not be
  reproduced (four loads of 4–10 s). The relaunch is the run reported here. Logs: `encode_hung_1.log` on Starship.

## Registered verdicts

| | verdict | numbers |
|---|---|---|
| **R1** (reproduce v2, 5 folds) | **NOT_REPRODUCED_CODE_PATH** | 369/900 against v2's 368; band h = 0 (perfectly deterministic encode, so the band is exact); v1 REPRODUCED |
| **R2 primary** (the paper's described method, 10 folds) | **NOT_ABOVE_TEXT_TFIDF** | 0.358 against R0's 0.382: −2.4 points; McNemar p = 0.20; cell-bootstrap 95% [−6.4, +1.6] points |
| **R7** (spectral, n-matched, story re-encodings) | **NULL_HOLDS_PARTIAL** | null in the 8 sets whose topic control passed (L31–L63, not L51); 3 sets CONFOUND_SUSPECTED (L3, L51, all-16); 6 UNINFORMATIVE (control failed) |

**R2 is not an underpowered maybe.** Power at +3 points was 0.16 before data, but the result landed below text, and
the cell interval's upper end (+1.6 points) is under the 3-point floor. For this probe the data exclude a gain of 3
points or more at 95%. §6.1's follow-up was for INCONCLUSIVE and does not apply.

**Secondary family (Holm, 4 tests):**
- S1, R1 against the R0 file (30-way): 0.410 vs 0.370, p = 0.036, **Holm 0.071, not significant.**
- S2, R1 against R0 on valence: AUROC 0.992 vs 0.963, **Holm 0.006**; cell CI [0.011, 0.052]. R1 carries v2's FWL,
  fitted label-free on each fold's own test trials.
- S3, R2 against R0 on valence: AUROC 0.985 vs 0.967, **Holm 0.027**; cell CI [0.005, 0.035].
- S4, R2 against the R0 file (30-way): 0.358 vs 0.370, Holm 0.55.

**On the binary "valence" split, the cache readouts separate the two halves a little better than text (TF-IDF): a
registered, confirmed secondary result. On the 30 emotions, they do not.**

**That split is narrower than its name** (correction plan §3, 2026-10-14; the prereg never addressed it, which is my
gap). It is v2's hard-coded word sets: a 10-emotion subset, n = 300, where 4 of 5 positives are low-arousal and 4 of 5
negatives high-arousal. So it measures valence and arousal together, and S2/S3 cannot say which one the cache carries
beyond text. *(Added 2026-10-19 16:01 PDT: I first reported this result to Thomas as "on valence", without the confound.)*

## Pre-written sentences the analysis emitted (prereg §5.6)
- "A raw-residual LR baseline (res_out) reaches 0.494 >= R2's 0.358: the probe's information is in the residual stream
  without W_K or the cache, and the paper may not say that W_K or the KV cache adds to it." (res_in_ln: 0.490.)
- "A tuned text (TF-IDF) baseline reaches 0.448 >= R2's 0.358: the comparison with the untuned R0 may not be cited as
  the cache beating text."

## Exploratory, labelled as such (no claim rests on these)
- v2's own pipeline (R1) at **10 folds: 0.472**, against R0 at 10 folds 0.382 (p = 2.6e-6) and against R2 0.358.
  What v2 ran (raw class-mean keys, dot product, all positions, test-fold FWL) reads well above what the paper
  *describes* (W_K-projected directions, per-layer cosine, skip 50, pooled FWL). Which difference carries the gap is
  not tested.
- **v2's lead comes from where it fits FWL.** `R1.mean_trainFWL`, v2's probe with FWL fitted on the training fold
  instead of the test fold, reads **0.348** at 5 folds, against 0.410 with v2's test-fold fit. That puts it at the
  described method's level (0.358) and below text (TF-IDF, 0.370 at 5 folds). The step is label-free, but it adapts
  to each test fold's own covariates, and the paper's 40.9% depended on it. *(Added 2026-10-19 17:52 PDT. The paper-edit agent
  found it in this file. I had missed it.)*
- Tuned TF-IDF 0.448; raw-residual LR 0.494. The ordering is: residual LR > v2's pipeline at 10 folds > tuned text >
  untuned text > the described method.
- Spec curve: 14 of 288 cells above fold-matched text (uncorrected); median 0.306; max 0.412 (L23).
- R7 all-16: 0.112 (3.4x chance), CONFOUND_SUSPECTED by the registered rule (length alone reaches that set's detection
  threshold). It is not read as a signal.
- **My R7 confound rule could barely let a signal through** (found by the paper-edit agent's review, 2026-10-19 17:53 PDT). The
  length covariates alone reach 0.053. The detection thresholds are 0.052–0.055, and length alone reaches them in 15
  of 17 sets, so the rule would flag any significant set there. All three significant sets (L3, L51, all-16) were
  vetoed that way, and 12 of 17 sets have p < 0.05 before correction. **R7's NULL_HOLDS_PARTIAL means the null holds
  in the 8 sets that weren't significant. The spectral question on story re-encodings is open, not settled.** The
  verdict stands as registered; this is how to read it.
- **The tuned text (TF-IDF) baseline chose its C at the edge of the grid in every fold**, so 0.448 probably
  understates tuned text. The follow-up must widen the grid.

## The sealed predictions, scored (prereg §5.5)

| sealed | p | outcome | log2 |
|---|---|---|---|
| R1 reproduced within the band | 0.70 | no | −1.74 |
| R2 in FAIL_BELOW_TEXT's class | 0.25 | **yes** (I put 0.50 on INCONCLUSIVE, 0.20 on EXCEEDS) | −2.00 |
| R2 point guess 0.34, 90% interval 0.22–0.46 | | 0.358: inside | |
| R2 "a little below R1" | | yes: 5.2 points below R1 at 5 folds | |
| all-16 cell within 5 points of the primary, same side of text | 0.70 | yes (0.351) | −0.51 |
| valence S3 not distinguished from text | 0.65 | no | −1.51 |
| R7 NULL_HOLDS (incl. partial) | 0.60 | yes | −0.74 |
| "if R1 fails, the weights not the code" | | wrong: v1's exact 470 verifies the weights | |

I expected a cleaner reproduction and a vaguer R2. The opposite happened: the reproduction missed by one trial, and R2
answered clearly.

## What the registered rules say follows
- **R1, NOT_REPRODUCED_CODE_PATH:** the bounded investigation (prereg §5.1): fingerprint (done, IN_BAND), dry run and
  port check on Starship (dry run done, 77/77 with the real v2 file), one environment diff. No GPU, one day at most.
  The difference is one trial in 900. I record it as the verdict says and don't call it noise: h = 0 means the encode
  gives the same keys every time, so the 369 is what today's code path produces.
- **R2, NOT_ABOVE_TEXT_TFIDF:** the claim that the cache or W_K holds the user's state beyond the words stays
  withdrawn. It is not restored, and with the cell interval under the floor it can now be stated as not supported for
  the described method, not merely untested.
- **R7, NULL_HOLDS_PARTIAL:** no paper sentence is restored. One scoped sentence may be added (prereg §5.3). The
  direction-versus-shape sentences stay withdrawn, and the per-layer table and kitchen sink are withdrawn regardless.
- **Forbidden:** "text" without "(TF-IDF)"; any pairing in §5.3 (a)–(c).

## Open, for Thomas
- The main.tex edits that follow from the above (draft to come, not committed).
- Whether to pre-register the exploratory ordering as its own test on fresh stories: does the residual stream carry
  emotion beyond a *tuned* text baseline (0.494 vs 0.448 here, uncorrected), and which of v2's choices carries its
  lead over the described method?

## Addendum: the R1 investigation (prereg §5.1), and what it changes in how R1 is read

The bounded investigation ran within scope (CPU only, under an hour; `R1_INVESTIGATION.md`). The verdict stays
**NOT_REPRODUCED_CODE_PATH**, since the prereg fixes it. But the reasoning built into that name is contradicted:
- **The port is faithful.** v2's own code, extracted by AST and run on today's pooled keys, gives 369, with 900 of
  900 predictions equal to the port's. The pooling is bit-exact to v2's lines.
- **sklearn is excluded** (the one environment diff): 1.8.0 against 1.9.0 gives identical results, down to solver
  iterations.
- **The keys changed between May and today.** v2's last-token CCS has no classifier and no optimizer. It reads
  0.775556 in May's results file (checked by Lyra against `wk_backtrack_v2_results.json`) and 0.774889 today: 3 pair
  orderings of 4,500, which is 571× more than arithmetic precision can move. Mean-token CCS is identical.
- So **"today's keys match v2's evening, so look at the port" was wrong.** v1's exact 470 verified the keys only at
  the resolution of a 30-way count. A witness that coarse can't certify that the keys are identical. That is a design
  lesson for the next prereg, not a reason to change this one.
- **Not established:** which trial moved (May saved no per-trial predictions), and whether the cause is the torch or
  transformers stack (May ≤ 2.12 / ≤ 5.9, today 2.13 / 5.14.1) or scipy's optimizer. May's environment no longer
  exists.
- **The band's quirk:** with zero measured noise, the 30 "perturbed" repeats were float64 reruns, which is where v1's
  h = 1 came from. No verdict changes.

**For R2 this changes nothing.** R2 tests the described method on today's stack. A key change of this size (one
trial in 900 on R1) cannot move a result that sits 2.4 points below text with its interval's top under the floor.

**For the paper:** v2's number reproduces to within one trial in 900 on a software stack that has changed since May.
The honest sentence is that, not "reproduced" and not "failed to reproduce".

## Addendum: are we measuring what we think? (Thomas's questions, 2026-10-19)

**1. Do the methods get the same trials right? No, though they overlap well above chance.** (From `per_trial` and
each method's predictions in the results file.)

| pair | both right | only first | only second | both wrong | both right if independent |
|---|---|---|---|---|---|
| text R0 / cache R2 | 196 | 148 | 126 | 430 | 123 |
| text R0 / residual LR | 279 | 65 | 166 | 390 | 170 |
| tuned text / residual LR | 317 | 86 | 128 | 369 | 199 |
| v2 R1 / cache R2 | 243 | 126 | 79 | 452 | 132 |

Easy stories are easy for every method. About 40% of each method's hits are its own.
May's 368 against today's 369 can't be compared trial by trial, because May saved no per-trial predictions.

**2. Did we confirm the labels are right? No, and that limits every number above.**
- **The label is what the generator was asked for.** The same model wrote every story from "Write a short paragraph
  about {topic}. The character is feeling deeply {emotion}. Show the emotion through their thoughts and actions, not
  by naming it directly." (`emotion_geometry_bridge.py:236-243`).
- **The planned check never ran.** The bridge's design lists "C3: Behavioral verification (do stories actually contain
  the emotion?)" (:81) against its own risk R3 (:91-92). Nothing in the code implements it, and no results file
  holds its output. So no reader, human or model, has confirmed that the stories express their labels.
- **It is a character's emotion, not the user's.** The stories are third person ("Elias stared at the petri
  dish…"), placed in the user turn. Every probe here decodes *the requested emotion of a character in a story the
  user sent*, not the user's own state. "User model" overstates what this design can show (correction plan, finding 1).
- **The misses are mostly near-synonyms.** On the bridge's own valence/arousal coordinates (:170-200), a miss lands at
  a mean distance of 0.53 (residual LR) to 0.69 (text), against 1.11 for two random labels. 74–84% of misses keep
  the valence sign. The commonest confusions are terrified/panicked, peaceful/serene, calm/serene and
  hostile/furious. The 30-way task has ambiguity built in, and its ceiling is unknown: it could be well under 100%.

**So the honest unit is "decodability of the requested emotion label".** Whether a probe reads the story's emotion,
the model's own reading of it, or the label's noise can't be told apart until the labels are checked.
