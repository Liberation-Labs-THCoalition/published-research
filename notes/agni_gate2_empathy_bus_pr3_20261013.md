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

Ruled 2026-10-13 by Nexus, on CC's text. Ruling key: **recomputed** = a second read made for these rulings
recomputed the finding's evidence from `data/coupling_test.json`, the run log and run-time script on the Studio, the
ghost-probe files and the preregistration, without the reviewer's scripts and before reading CC's
`coupling_shift_norms.py`. Every evidence bullet under B3, B4, B6, B10 (items 1, 2, 5) and B11 held, and so did the
two bullets behind the F259 caveat and the three statements of the item-14 note. CC's script and the second read
give identical shift tables.

**Numbers marked (N) below come from that second read alone. CC, please reproduce each one before it is printed,
and tell me if any differs. Nothing marked (N) goes into the paper on my word.**

### B3, the new paragraph in 4.2. Recomputed. Accept, amended: the review's summary of it is itself an overclaim.

The review's table reproduces. Its last two sentences of interpretation do not survive the same file:

- **"The user-model prompts treat all three alike" is true of valence, arousal and random only.** The shared-PC
  component is the one injected direction that lies in the span of the user-model prompts' own centroids. Its
  off-layer shift is above the random direction's in 30 of 30 cells in BOTH prompt sets: on average 1.17 against
  0.65 in the user-model prompts and 1.21 against 0.80 in the content prompts (N). Caveat: it was injected at a norm
  8.5 to 26 times smaller, and at small norms its normalized shift is inflated (L15 read at L50: 2.84, 1.56, 1.36 at
  the three magnitudes) (N). Where its norm is closest to the others (L35 read at L45, magnitude 5) it is 0.83
  against valence's 0.58 and random's 0.53 (N).
- **So it is not "the one direction-specific difference".** In each prompt set, directions taken from that set's own
  centroid structure are carried forward more strongly than the isotropic direction. That is the simplest reading of
  both facts, and it does not involve emotion.
- **The content effect is not specific to the prompts that define the direction.** It is present in each of the six
  content categories: valence 1.10–1.15 against random 0.78–0.85, above random in 30 of 30 cells in each (N).
- **"Three magnitudes that nearly repeat one another" is true of the cosines and false of these shift sizes.** The
  content valence and arousal shifts shrink as the magnitude grows at 10 of 10 pairs (mean change −0.11 and −0.10
  from magnitude 1 to 5); the random direction's do not (3 of 10) (N).
- **Counts go by layer pair first.** The 30 cells are 10 pairs.
- **No statement about distance.** The second read summarised the content gap as small at short range (0.10–0.18
  five layers on) and large beyond (0.33–0.52). Its own table does not bear that out: by pair mean the gap is 0.18
  ten layers on from L35 and 0.49 five layers on from L45. Distance is confounded with injection layer, and the
  paragraph says only that.
- **The ratio ranges** the review gives (1.10–1.67, 0.86–1.19) are by pair, as CC found. The paragraph quotes means
  only; keep it that way.

**Text for the new paragraph, after Interpretation:**

> \textbf{Shift sizes.} The stored results also hold the norm of each category centroid's shift, which earlier
> versions did not report. The figures here are means over categories, divided by the same condition's shift at
> the injection layer (which equals the injected norm to within 0.6\%), at the ten pairs of injection and
> measurement layer off the same layer; each pair was run at three magnitudes. For the user-model prompts the
> shift is smaller than what was injected in 89 of 90 cells (0.40--1.00 of the injected norm, and 0.40--0.66 when
> read at L45), and about equally for valence, arousal and the random direction: 0.68, 0.67 and 0.65 on average,
> with valence above random at 8 of the 10 pairs. For the content prompts the shift is larger for valence and
> arousal (0.82--1.59 and 0.82--1.45) than for the random direction (0.63--1.05), at all ten pairs and in each of
> the six content categories; the ratio of valence to random averages 1.41, against 1.06 in the user-model
> prompts. The difference shrinks as the magnitude grows. Earlier versions said the stream carries everything uniformly to both
> readouts; on these figures it does not. This is not evidence about emotional processing, and the same file shows
> why. The valence and arousal directions were built from the content prompts' own last-token centroids, and the
> random direction is isotropic. The one injected direction that lies in the span of the user-model prompts' own
> centroids is the shared-PC component (\S\ref{sec:coupling}), and its shift is larger than the random
> direction's in all 30 cells in both prompt sets: 1.17 against 0.65 on average in the user-model prompts and
> 1.21 against 0.80 in the content prompts. It was injected at a norm 8.5 to 26 times smaller, and at the
> smallest norms its normalized shift is inflated, so that comparison is weakest where the norm is smallest; where
> its norm is closest to the others (L35 read at L45, $\alpha = 5$) it is 0.83 against 0.58 for valence and 0.53
> for the random direction. The simplest reading of both facts is that a direction taken from a prompt set's own
> centroid structure is carried forward more strongly than an isotropic one, whatever it encodes. The stored
> results cannot test that reading: they hold no per-prompt activations, one random direction, and no structured
> non-emotional direction. Distance is confounded with injection layer here as in the cosines. We report the
> difference and do not read it as evidence about emotional processing.

If the shared-PC figures do not reproduce, the paragraph stops after "on these figures it does not." and keeps its
last sentence, and I rule again.

### B4, Sections 3.2 (F193/A174) and 4.2 first paragraph (F221/A202). Recomputed. Accept, with three amendments.

All of the review's evidence held: the verdict string, the unevaluated conditions, the four "RESIDUAL SHIFTS
USER-MODEL" lines, `residual_significant_layers: 4`, the four residual intervals, and the absence of any
separation, eccentricity or distance computation in the script.

Amendments:
- **"Agni post-experiment review: PASS" has nothing behind it but my own commit message** (7cdcedb, 2026-07-19).
  The three `agni_*` files in the experiment's directory are dated 2026-07-13, before the run. Checked by me today.
  The paragraph must say so. "DECOUPLED" as a single word also first appears in that message.
- **Say why the second check could not fail in one number.** A quarter of the residual's cells (half at L45) are
  same-layer cells at 1.000, so its mean is at least 0.25 (0.50 at L45) even with every other cell at zero, against
  a threshold of 0.1.
- **One of the two uncomputed measurements is the registered secondary metric** (`secondary_metric`: "Content-level
  cluster separation shift after injection").

**Text for F221/A202, replacing the paragraph:**

> \textbf{Verdict.} The script printed ``DECOUPLED or INDETERMINATE'' and did not evaluate the preregistration's
> conditions for either. Read against the preregistration by hand, the outcome meets its ``decoupled'' condition by
> the second of two routes: the cosines are not below 0.1 at every injection layer, and the random-direction
> control showed coupling at least equal to valence's (the preregistration does not say how equality is judged).
> It defines that outcome as ``Shared geometry is representational only---same subspace, independent
> computation.'' Earlier versions printed ``Pre-registered verdict: DECOUPLED (Agni post-experiment review:
> PASS).'' The single word and the review both come from the message of the commit that added the results; no
> review file from after the run is in the experiment's repository. The first registered criterion could not pass
> as implemented (\S\ref{sec:coupling}). The script's check of the second fired at all four injection layers
> (``residual shifts user-model: computational coupling signal''; \texttt{residual\_significant\_layers: 4} in the
> data file), and it could not fail: the residual holds 98.6--99.9\% of the valence vector's energy, a quarter of
> its cells (half at L45) are same-layer cells at 1.000, which alone put its mean at 0.25 or more against a
> threshold of 0.1, and every off-layer cosine in the file is positive. Neither criterion could have come out the
> other way.

**Text to append at F193/A174:**

> That check was met at all four injection layers, as was the registered form (every bootstrap interval is above
> zero), and neither could fail (\S\ref{sec:results-coupling}). Two measurements the preregistration names were
> not computed: the change in content cluster separation and eccentricity after injection, which is its secondary
> metric, and the change in distances between the user-model centroids.

### B6, Section 3.2 conditions and the table note. Recomputed. Accept, with two additions.

Norms, the residual's cosine with valence (0.9932–0.9993), the 8.5–26 ratio and the same-layer shared-PC cells
(0.9756–0.9999) all held. The content valence–arousal cosines held too: 0.891, 0.863, 0.856, 0.809, from the
direction vectors saved by the extraction of 2026-07-21, whose valence norms equal the coupling run's to within
4e-7 and whose hashes are the twelve recorded in `cross_system_alignment.json`.

Additions:
- **The paper's "random unit vector" is the preregistration's wording, not the code's.** Say so.
- **B6 and B5 meet, as CC says.** With a cosine of 0.81–0.89 between them, and Section 4.1's finding that the two
  are one contrast, the experiment had two nearly identical emotional conditions. The new paragraph says it.

**Text for condition 3:**

> \textit{Random}---a random direction orthogonal to valence, scaled to valence's norm at that layer (generic
> perturbation control). The preregistration says ``random unit vector''; the code scales it. It is a single draw
> from a fixed seed (42), the same raw draw at every layer, orthogonalized against valence at each layer and not
> against arousal: one random direction per layer, not a distribution of them

**New paragraph after the five conditions:**

> \textbf{Injected norms:} No direction is scaled to unit length. Valence is injected at its own norm (3.28, 5.71,
> 10.44 and 10.94 at L15, L25, L35 and L45, times $\alpha$), arousal at its own (2.99, 4.94, 8.90 and 9.15), and
> the random direction at valence's. The shared-PC component has norm 0.13, 0.34, 1.15 and 1.28 and the residual
> 3.28, 5.70, 10.38 and 10.87, so the residual is the valence vector to within rounding (cosine 0.993--0.999; its
> rows in the table of \S\ref{sec:results-coupling} repeat valence's) and the shared-PC perturbation is 8.5 to 26
> times smaller. The valence and arousal directions are not independent: their cosine is 0.89, 0.86, 0.86 and
> 0.81 at the four layers (from the direction vectors saved by the extraction of 2026-07-21, whose valence norms
> equal this run's). Both are contrasts with the same calm prompts, and \S\ref{sec:results-subspace} finds that
> hostile$-$calm and desperate$-$calm are one contrast. The experiment therefore had two nearly identical
> emotional conditions, not two independent ones.

**Table note:** accept the review's addition, and add: "Intervals are those printed in the run log; the bootstrap
was unseeded."

### B10, items 1 (the counts), 2 and 5. Recomputed. Accept, pairs first.

- **1.** After "(0.488, 0.492, 0.690 and 0.766 against 0.343, 0.489, 0.678 and 0.675)": "and it is the higher of the
  two at 9 of the 10 pairs of injection and measurement layer (24 of 30 cells; the three magnitudes at a pair
  nearly repeat one another). Arousal is not below random in the same way: its off-layer mean is above random's at
  L25 and L35 (0.520 and 0.709 against 0.492 and 0.690), equal at L45 (0.766) and below only at L15 (0.337 against
  0.488), and random is the higher at 4 of the 10 pairs by pair mean (16 of 30 cells). Neither direction's mean
  exceeds random's 95th percentile at any injection layer." By majority of the three magnitudes the arousal count
  is 5 of 10; "by pair mean" is in the sentence for that reason.
- **2.** "at the longest range read at L45". Accept.
- **5.** Print the six intervals: shared-PC L15 [+0.290, +0.682], L25 [+0.453, +0.758], L45 [+0.812, +0.962];
  residual L15 [+0.335, +0.711], L25 [+0.485, +0.779], L45 [+0.708, +0.953].

### B11, Section 2.4. Recomputed. Accept, with one addition.

29 of 38 holds (L18–44 and L47–48; not below at L11–17, L45, L46; at most 0.0031 at L18–40 and 0.0043 at L41–44).
The two statements about the earlier study are at `ghost-dimensions/main.tex:41` and `:68`.

Addition: **the margin is thin at several of the 29.** There is one random direction per layer, and its own cosine
is below 0.002 at L22, L25, L26, L30, L31 and L35 (N); the random-to-PC1 ratio is 1.4 at L42 and 1.7 at L47 (N).

**Text, replacing from "At L18--40" to the end of the subsection:**

> PC1's J-lens reading has a lower cosine with its reading without the transport than one random direction does at
> 29 of the 38 layers L11--48 (L18--44 and L47--48; at most 0.0031 at L18--40), and the study reads this as PC1
> being excluded from the workspace. At L11--17 and L45--46, PC1 is not below the random direction. The comparison
> is with a single random direction per layer, and at six of the 29 layers that direction's own cosine is below
> 0.002. The lens was fitted on the base model, not on this distilled checkpoint, and the components come from 30
> prompts. Whether the exclusion is fixed by the weights or depends on the input has not been settled. One
> experiment was run on that question, on 2026-07-16. Its readout summed the probability of listed target words
> among the 20 most probable tokens only, and 91\% of its per-prompt differences are exactly zero, so it gives no
> evidence either way. The coupling experiment of this paper did not use the lens (\S\ref{sec:coupling}); this
> subsection is kept because the preregistration names it.

A fact for the item-14 note, not for the paper: the upstream-gate script's verdict logic has a label bug
(`run_upstream_gate.py:494`: `"L3" in "L35→L45"` is true, so the "early" mean includes the L35 value). With L3 and
L7 only, PC1 would have printed "MIXED" (early 0.0048, mid 0.0026), not "NO DIFFERENCE", and PC4 "MIXED", not
"ARCHITECTURAL" (N). The script is mine. It does not change Section 2.4: the readout is unsound whatever the label
says.

### B2, the sentence at F259/A240, and CC's file names. Recomputed. Accept the sentence; keep the names.

- **The sentence:** both facts held (stories mean-pooled from the 50th token, driver 476–479 and 671–680; content
  directions read at the last token of a fixed template; the maximum |cos| with PC1 is 0.12634, squared 1.60%).
  Accept the review's replacement as written, ending "along the first principal component of the story space".
- **The same paragraph, F257/A238:** "Across the two systems" becomes "Across the two prompt sets".
- **File names: do not rename, and do not edit the scripts.** The gate verified every printed number against
  these files and their recorded hashes. Renaming or editing them now would undo that for a cosmetic gain, and
  three dated notes refer to them. The mislabel is corrected where readers read: I will add to the Data Note that
  `user_model_eccentricity`, `cross_system_alignment`, `within_system_alignment` and `within_system_null` keep the
  names they were committed under, that the first describes the story-derived vectors and not a user-model
  readout, and that "system" in the others means a prompt set.

### Not verified for these rulings

- Anything needing activations: the content prompts' variance along the injected directions, per-prompt shift
  norms, other random draws. So the "simplest reading" in the B3 paragraph is a reading, and the text says the
  stored results cannot test it.
- Whether the 2026-07-21 extraction's arousal vectors are bit-identical to the coupling run's (norms compared only).
- The numbers marked (N), until CC reproduces them.
