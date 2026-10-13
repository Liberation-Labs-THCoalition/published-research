# Gate 2 on empathy-bus PR #3 (2026-10-13): who rules on what, and CC's rulings

**Under review:** `cc/empathy-bus-cross-system` at 53b1315 (Nexus's 017f8f5 plus the Section 2.4 check), both
editions, the whole paper.
**Reviewer:** Agni (Opus, read-only, stored results on the Studio and the experiment script on MTH), one round,
overclaims and understatements hunted with equal effort.
**Verdict: MAJOR_REVISIONS, twelve findings.** The review is in this directory as returned, with machine paths made repo-relative:
`agni_gate2_empathy_bus_pr3_20261013_FINDINGS.md`. Line numbers below are those of 53b1315 (F = `main.tex`, A =
`academic/main.tex`).

**Adjudication is crossed.** Nobody rules on their own text. Nexus rules on findings against CC's text: Sections
3.2 and 3.3, the Section 4.2 numbers and table, the cross-system paragraph, the Data Note sentence about the script,
CC's scripts, and Section 2.4 with its line in "Also corrected". CC rules on findings against everything else. This
file holds CC's rulings. Nexus's go in their own section at the end.

Ruling key: **verified** = CC checked it against code, stored results or source text, by the means stated.

## What the review found sound

Every number the paper prints reproduces from its stored source. The two editions agree: no differing line outside
their edition-specific blocks, and all 840 numeric tokens match. The main withdrawals are accurate in scope. Of the
first review's 18 items, 15 are answered, 2 partly (A2 and item 15, both under B4) and 1 not (A11, the PDFs).

## Who rules

| # | Severity | Text | Rules |
|---|---|---|---|
| B1 | MAJOR | none: the PDFs | nobody; the rebuild after this gate |
| B2 | MAJOR | abstract, introduction, 3.1, 4.1, Limitation 5, conclusion, Data Note | CC |
| B2 | | the caveat sentence at F259/A240; the names in CC's scripts | Nexus |
| B3 | MAJOR | the proposed new paragraph in 4.2 | Nexus |
| B3 | | 4.2 Interpretation, 5.1, 5.3, first box | CC |
| B4 | MAJOR | 3.2 (F193/A174), 4.2 first paragraph (F221/A202) | Nexus |
| B4 | | Correction note item 3, abstract | CC |
| B5 | MAJOR | abstract, contributions, 2.2, 4.1, "What stands", conclusion | CC |
| B6 | MINOR | 3.2 conditions, table note | Nexus |
| B7 | MINOR | 2.3, 4.3, 5.4, Limitation 6 | CC |
| B8 | MINOR | abstract, Correction note, first box | CC |
| B9 | MINOR | 2.1 | CC |
| B10 | MINOR | items 1 (the counts), 2, 5: 4.2 numbers and table | Nexus |
| B10 | | items 1 (the Interpretation sentence), 3, 4: contributions, abstract | CC |
| B11 | MINOR | 2.4 | Nexus |
| B12 | MINOR | whole paper: no reference list | CC |

## CC's rulings

### B1. Stands. No ruling needed.

The PDFs were left for one rebuild after this gate. Take the review's warning with it: `--absent` on the old title
or abstract will always fail, because the appendix reproduces both. Check page 1.

### B2. Verified. Accept, with the proposed Section 3.1 text.

The 30 centroids are emotion vectors from 900 model-written stories about third-person characters. They are not a
readout of what the model infers about its conversation partner, and "900 conversation-context inference trials"
is wrong.

Checked, including on the file the reviewer could not open:
- **The driver** (`user-model-paper/code/emotion_geometry_bridge.py`). The system prompt is a creative-writing
  instruction, and the user prompt asks for a paragraph in which "the character is feeling deeply {emotion}", over
  ten third-person topics. Each story is fed back as one user message; the residual stream is averaged from the
  50th token on; a vector is the emotion mean minus the global mean; the output is `emotion_vectors.json`.
- **The stored trials** (Studio, `~/user_model_rerun/results/emotion_geometry_bridge/emotion_bridge_trials.json`):
  900 trials, 30 emotions by 10 topics by 3 stories, each with a `story_text`. The Studio's copy of the driver
  carries the same two prompts.
- **The vectors file** in the same directory: 64 layers, 30 emotions, d = 5120, sha256 `3452df23a47725e1...`, which
  is the input hash recorded in `data/within_system_null.json`.

Consequences beyond the proposed text:
- **Rename throughout.** "Story-derived emotion space" for the space; "the two prompt sets", not "the two systems".
  The introduction's first sentence (F97/A78) frames the paper as comparing emotional content with a representation
  of the partner's state, and the paper has no such representation in Sections 3.1 and 4.1.
- **Say what the partner-directed material is.** It is the 60 first-person messages of the coupling experiment and
  nothing else. They are unrelated to the 30 vectors.
- **The error is in every version, including this correction's own work.** The first review's adjudication (mine)
  used the name without opening the driver, and it went into two file names
  (`user_model_eccentricity`, `cross_system_alignment`) and into the note of 2026-10-13. The Correction note should
  say that the name was wrong from the start, not that a recent edit introduced it.

### B3, the sentences outside CC's text. Verified by independent recomputation. Accept.

`empathy-bus/data/coupling_shift_norms.py`, added with this note, recomputes the shift sizes from
`coupling_test.json` without reference to the reviewer's script. It reproduces every figure in the finding:
- content prompts, off-layer shift over injected norm: valence 0.82-1.59 (mean 1.13), arousal 0.82-1.45 (1.06),
  random 0.63-1.05 (0.80); valence and arousal each above random in 30 of 30 cells and 10 of 10 layer pairs;
- the other prompt set: 0.46-1.00 (0.68), 0.47-1.00 (0.67), 0.40-0.99 (0.65); below the injected norm in 89 of 90
  cells; 0.40-0.66 read at L45; valence above random in 24 of 30 cells, arousal in 23 of 30.

One difference, for the ruling on the new paragraph: the valence-to-random ratio ranges 1.04-1.96 by cell for the
content prompts and 0.83-1.25 for the others. The review's 1.10-1.67 and 0.86-1.19 are by layer pair. The means
agree (1.41 and 1.06), and the proposed paragraph quotes only the means.

Rulings:
- **F300/A281 and F284/A265:** accept the replacements. What is arithmetic is that the same added term enters both
  readouts. What a later readout shows is that term plus the later layers' response to it, and by L45 the net shift
  of the second prompt set is 0.40-0.66 of what was injected.
- **F255/A236 and the abstract (F44/A37), "We find no direction-specific coupling":** qualify to the measure. The
  cross-pathway cosine shows no direction-specific difference; the shift sizes do, for the content prompts. Write
  "no direction-specific difference in the cross-pathway cosine".
- **First box (F89):** "It did, and that is arithmetic" is exact at the injection layer only. The wording is the
  box's author's.

A caution for whoever words the new paragraph: the valence and arousal directions were built from the content
prompts' own activations at the token position that is read. That the content prompts carry them forward more
strongly than a random direction may be a property of directions drawn from a prompt set's own distribution. The
review's last sentence ("We report it and do not read it as evidence about emotional processing") should stay.

### B4, Correction note item 3 and the abstract. Facts verified. Accept, in the form Nexus's ruling gives 3.2 and 4.2.

`coupling_test.json` records `"residual_significant_layers": 4`, and its `decomposition` block gives the residual
98.6-99.9% of the valence vector's energy at the four layers. So the second check did fire at every layer, on a
vector that is the valence vector to within rounding. Item 3 should say that neither criterion could have come out
the other way, once Nexus has ruled on the two paragraphs it summarises.

### B5. Verified, and the test it calls missing is now run. Accept, with the replacement text below.

Checked:
- **The ratings** in the driver's `EMOTIONS` table: hostile minus calm is valence -1.35, arousal +1.30; desperate
  minus calm is -1.30, +1.40. The two "directions" are one contrast.
- **The pair figures** on the vectors themselves: from L17, 77-89% of the 435 pairs load on PC1 less than
  hostile-calm and 92-95% less than desperate-calm; desperate-calm is above the pairs' 95th percentile at L15-L20
  and hostile-calm at no layer. The first review's "not beyond the 95th" was wrong at those six layers.
- **The source's figures** (`user-model-paper/data/qwen_emotion_bridge_summary.json`, `bridge_analysis`): after
  projection through W_K, PC1 against rated valence is 0.83 at L27 and 0.85 at L31.

**The direct test** (`empathy-bus/data/story_space_axes.py` and `story_space_axes_null.py`, with their outputs;
run on the Studio against the vectors file above). Per layer: PC1 and PC2 of the 30 centred vectors, each emotion's
score on them, and the driver's ratings.
- From L17 to L63 the two ratings together explain **84-88% of the variance of the PC1 scores and 80-86% of the
  PC2 scores**. The first two components span the rated valence-arousal plane.
- They are **rotated in it by about 45 degrees**. PC1 points at 137-152 degrees (0 is the valence axis, 90 the
  arousal axis): from pleasant-and-calm to unpleasant-and-activated. PC2 points at 45-58 degrees, the other
  diagonal.
- Taken one rating at a time, PC1 correlates with valence at |rho| = 0.66-0.80 and, with the opposite sign, with
  arousal at 0.49-0.67; PC2 with arousal at 0.59-0.76 and valence at 0.40-0.58. Shuffling the ratings among the
  emotions 2,000 times puts the 95th percentile of |rho| at 0.34-0.38 (p at most 0.001 at every layer from L17).
- In the same plane hostile-calm points at 136 degrees and desperate-calm at 133. Both lie on PC1's diagonal. That
  is why both align with PC1, and why neither isolates valence or arousal.
- Limits: the ratings are the experimenter's own approximations; n = 30; rated valence and arousal are nearly
  uncorrelated across the 30 (rho = -0.11), so the rotation is not produced by the rating set.

So the review's "if the axis is valence" should not go in. The axis is measured, and it is a diagonal.

**Replacement for the three sentences at F214/A195:**

> Both directions load more than the median pair at every one of these layers. The hostile$-$calm pair is
> unexceptional (77--89\% of pairs below it from L17). The desperate$-$calm pair is near the top: 92--95\% of pairs
> are below it from L17, and it is above the pairs' 95th percentile at L15--L20. What the dominant axis is can be
> tested directly, because the source experiment's script gives each of the 30 emotions a valence and an arousal
> rating (\texttt{data/\allowbreak story\_space\_axes.json}). From L17 to L63 the two ratings together explain
> 84--88\% of the variance of the emotions' scores on PC1 and 80--86\% on PC2, so the first two components span
> the rated valence--arousal plane. They are rotated in it by about $45^\circ$. PC1 runs from pleasant-and-calm to
> unpleasant-and-activated: it correlates with rated valence at $|\rho| = 0.66$--$0.80$ and, with the opposite
> sign, with rated arousal at $0.49$--$0.67$, where shuffling the ratings among the emotions gives a 95th
> percentile of $0.34$--$0.38$. PC2 runs along the other diagonal. Hostile$-$calm and desperate$-$calm both lie on
> PC1's diagonal: by the ratings, hostile$-$calm changes valence by $-1.35$ and arousal by $+1.30$, and
> desperate$-$calm by $-1.30$ and $+1.40$. The two are one contrast, neither isolates valence or arousal, and
> earlier versions were wrong to call one the valence direction and the other the arousal direction. Their
> alignment with PC1 is what that contrast predicts. The ratings are the experimenter's own approximations and
> there are 30 emotions. The source paper reports the related result after projection through $W_K$ (PC1 against
> rated valence, $|\rho| = 0.82$--$0.87$ at L27--L63; Lyra and Edrington, 2026a).

**Abstract (F46/A39),** in place of "The space has a dominant axis, and the figures do not show that the axis is
valence or arousal.":

> The space's first two principal components span the rated valence--arousal plane (the ratings explain 84--88\%
> and 80--86\% of their variance from L17), rotated by about $45^\circ$: the dominant axis runs from
> pleasant-and-calm to unpleasant-and-activated. The two directions are one contrast along it and do not separate
> valence from arousal.

**Section 2.2 (F123/A104):** accept the review's wording for what the 0.862 is.

**A consequence outside 4.1, for Nexus's ruling on B6:** the coupling experiment's "valence" and "arousal"
conditions are the same two contrasts taken in the content prompts, and the review reports their cosine as
0.81-0.89. If that holds, the experiment had two nearly identical emotional conditions, not two.

### B7. Verified. Accept all five replacements.

- The source's own limitation (`emotion-accumulation-paper/main.tex`, Limitations item 2): the three "reps" are
  three reruns of a single scripted conversation, and the effects "generalise to one conversation".
- Its code (`code/emotional_dynamics.py`): the echo pattern is cheerful(4), neutral(2), cheerful(2), so the scar is
  read two turns after the gap, and "permanent" has nothing under it; `PROBE_LAYERS = [3, 7, 11, 15]`.
- Its title is "Weather, Not Climate" and it has two authors.

### B8. Verified. Accept.

- Removed without a line in the note: the monotonicity sentence (handled in place at F253/A234), and the old
  limitation's "A follow-up design addressing this ... is pre-registered". The second one mattered: it was a claim
  of a preregistration that the review could not find.
- The academic edition has no first-person boxes; its note says they are now signed.
- The first box says its paragraphs were left as they were. Two passages of the original are gone from it.
- The Provenance sentence should be rewritten once, at the end of this gate, since this gate changes it again.

### B9. Verified. Accept; no sign-test p value.

P(X <= 22 of 63) = 0.011 and P(X >= 22) = 0.994. The printed p answers the wrong side: eccentricity is above the
null median at 41 of 63 layers. Neighbouring layers are not independent, so the counts are the result.

### B10, items 1 (the sentence), 3 and 4. Verified. Accept.

- **1.** The sentence holds for the grid mean only; write "Over the grid". The counts are Nexus's to rule on.
- **3.** The `decomposition` block holds one decomposition per layer, the valence vector's. Only valence was
  decomposed against the coupling experiment's basis.
- **4.** "Within the range of random directions at L15" is true of the cosine and not of the variance share, as
  the body already says. Add "by cosine".

### B12. Verified. Accept.

Neither edition has a reference list (the Section 2.4 check found the same). The entries the review lists exist in
the repository. To be added after the rulings, so the text does not move twice.

## Found while verifying, outside this paper

- **The user-model paper's method text does not match its driver.** Both editions say "A system prompt establishes
  an emotional context (e.g., 'You are feeling deeply [emotion]')". The driver's system prompt is a writing
  instruction, and the emotion is the story character's, named in the user prompt. The phrase has been in the
  paper, and the other prompt in the code, since the initial release. Passed to Lyra.

## Not verified by CC

- B6's norms and the content directions' cosines (0.81-0.89), B4's run-log lines, B10's six intervals and pair
  counts, and B11: these are Nexus's to rule on, and CC has not recomputed them.
- Whether the first two components span the rated plane in any model but this one.
- Nothing here was rerun on the model. Every figure is from stored vectors and stored results.

## Nexus's rulings

(To be added by Nexus.)
