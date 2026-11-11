<!-- Agni's review as returned, 2026-10-13. Only change: paths on the reviewer's machine were made relative to this repository. The check scripts it names (S/...) are not in the repository. -->

# Agni gate 2 on empathy-bus (both editions) at 53b1315, 2026-10-13

Under review: `empathy-bus/main.tex` (F) and `empathy-bus/academic/main.tex` (A), branch `cc/empathy-bus-cross-system`, everything not on `origin/master` (4c77001). Line references are `F<line>/A<line>`. Read-only; nothing in any repository was changed. `S/` below is the reviewer's scratch directory, which is not in this repository.

## 1. Verdict

**MAJOR_REVISIONS.** The withdrawal of the inseparability claim is accurate in scope, the two editions agree word for word outside their edition-specific blocks, and every number the paper prints reproduces from its stored source. Four things stop release:

- The two PDFs in the branch are still the withdrawn paper.
- The "user-model emotion space" behind both of the abstract's descriptive results is, by the code and text of the source the paper cites, a set of emotion vectors from model-written stories about third-person characters.
- The account of the registered test reports the criterion that could not pass and omits the one that fired at all four layers and could not fail.
- The stored results hold a direction-specific difference (in shift size, not cosine) and evidence on what the dominant axis is; the paper reports neither.

## 2. Findings

### B1. MAJOR (release-blocking), INCONSISTENT. Both PDFs are still the withdrawn paper.

- **Where:** `empathy-bus/main.pdf`, `empathy-bus/academic/main.pdf`; the notice at F39/A27 describes a correction the PDFs do not contain.
- **As printed (page 1 of both PDFs):** "The Empathy Bus: Architectural Inseparability of Emotional Processing and Partner Modeling in Transformers" and "We report three findings that together establish a structural constraint on emotional processing in transformers: content-level emotion and user-model emotion are architecturally".
- **Evidence:**
  - `git diff origin/master...HEAD --stat -- empathy-bus/main.pdf empathy-bus/academic/main.pdf` is empty; the last PDF commit is c80e8da (2026-10-09).
  - `pdftotext` on each file: "Notice of correction|withdrawn" 0 hits, "share a common representational" 2 hits, control string "Qwen3.5" 3 hits.
  - First-review finding A11 is unanswered; the item-14 note says "PDFs are not rebuilt in this commit."
- **Not checked:** whether the corrected `.tex` compiles. There is no TeX on this machine or the Studio, and MTH (which has `pdflatex`) is outside my write scope. Static checks pass in both editions: 71 references resolve, braces and environments balance (`S/check_editions.py`).
- **Proposed action:** rebuild both, then `python tools/verify_pdf.py main.pdf --control "Qwen3.5-27B" --present "A Coupling Experiment That Could Not Test Inseparability" --present "Notice of correction"`. Do not use `--absent` on the old title or abstract: the appendix reproduces both, so the strings will be found. Check page 1 instead.

### B2. MAJOR, OVER. The "user-model emotion space" is not a user-model readout.

- **Where:** abstract F46/A39; introduction F97/A78; §3.1 F148/A129; §4.1 F208-216/A189-197; §4.2 F259/A240; Limitation 5 F328/A309; conclusion F338/A319; Data Note F360/A329.
- **As printed:** "\textbf{User-model emotion space:} 30 emotion centroids per layer from 900 conversation-context inference trials (Lyra et al., 2026a)." and "Inside the user-model emotion space (30 emotion centroids per layer, from 900 conversation trials)".
- **Evidence.** The input of all four new scripts is `~/user_model_rerun/results/emotion_geometry_bridge/emotion_vectors.json` (path recorded in `data/cross_system_alignment.json`). The driver that writes that file is in this repository, `user-model-paper/code/emotion_geometry_bridge.py`:
  - line 230: the system prompt is "You are a creative writing assistant"; line 240: the user prompt says "The character is feeling deeply {emotion}"; lines 212-223: the ten topics are third-person scenes ("a scientist working alone in a laboratory late at night");
  - lines 671-680: each generated story is fed back as a single user message; line 477: the residual stream is averaged over token positions from the 50th on;
  - line 1490: "emotion vectors come from encoding the GENERATED stories"; line 1551: `encoding_activations = story_activations`; line 534: vector = emotion mean minus global mean; line 1717: written to `results/emotion_geometry_bridge/emotion_vectors.json`.
  - The source paper's own text agrees (`user-model-paper/paper/main.tex:250-254`): "we re-encode each generated story as standalone input (no system prompt, no emotion label) ... These vectors represent the direction in residual-stream space that distinguishes emotional content."
  - 30 emotions x 10 topics x 3 stories = 900, and the recorded format (64 layers, 30 emotions, d = 5120, with hostile, calm, desperate) is the driver's output.
  - The phrase "900 trials of conversation-context inference" first appears in `convergence_subsection.tex` in the experiment's directory on MTH (2026-07-17).
- **Limit of this finding:** I did not open `emotion_vectors.json`. It is outside the Studio directory I was allowed to read. The identification rests on the recorded path, the format, the driver and the source text.
- **Consequences:**
  - §4.1 describes a story-emotion space.
  - The second cross-system comparison sets last-token directions from a fixed template against vectors mean-pooled over stories, so a small overlap can come from the pooling difference alone; the caveat sentence at F259/A240 does not name this.
  - The only partner-directed material in the paper is the 60 first-person messages of the coupling experiment.
- **Proposed replacement, §3.1 (F148/A129):**
  > \textbf{Story-derived emotion space} (called the user-model emotion space in earlier versions): 30 emotion vectors per layer from Lyra et al.\ (2026a). That experiment had the model write 900 one-paragraph stories about a third-person character feeling one of 30 emotions (10 topics, 3 stories each). Each story was then given back to the model as a single user message, and the residual-stream activation was averaged over token positions from the 50th on. A vector is the mean over the 30 stories of one emotion minus the mean over all 900 ($d = 5120$, all 64 layers). These are emotion vectors for narrative text. They are not a readout of what the model infers about its conversation partner, and earlier versions were wrong to describe them as coming from ``conversation-context inference trials.''
- **Abstract (F46/A39), first sentence of the paragraph:** "Inside a second emotion space, built from 900 stories the model wrote about characters feeling 30 emotions (earlier versions called it the user-model space), the hostile$-$calm and desperate$-$calm directions align with the space's first principal component ..." and, later in the paragraph, "Across the two prompt sets" in place of "Across the two systems".
- **§4.2 caveat (F259/A240), replacing the sentence that begins "The random directions are isotropic":**
  > The random directions are isotropic and this model's activations are not; no non-emotional contrast has been scored the same way; and the two sets of directions were read at different places, the last token of a fixed template for the content prompts and the mean over positions from the 50th on for the stories. The excess is therefore an upper bound on any shared emotional structure, not evidence of it, and the small overlap is not evidence that two kinds of processing are separate: at most 1.6\% of a content direction's squared norm lies along the first principal component of the story space.
- **Correction note, "Also corrected" (F74/A67), add:** "The 30 emotion centroids were described as a user-model space from ``900 conversation-context inference trials.'' They are emotion vectors from 900 model-written stories about third-person characters (\S\ref{sec:shared-subspace})."

### B3. MAJOR, UNDER (with one OVER sentence). The stored shift sizes show a direction-specific difference the paper does not report.

- **Where:** §4.2 F255/A236; §5.1 F284/A265, F288/A269, F290/A271; §5.3 F300/A281; first box F87-89 (flight only).
- **As printed:** "We find no direction-specific coupling." / "The stored results hold cosines and magnitudes only" (no magnitude is reported anywhere) / "A vector added to the residual stream is present wherever the residual stream is read; that much is arithmetic." / "Any vector added this way is present in both."
- **Evidence** (`S/check_magnitudes.py`, `S/check_magnitudes_summary.py`, `S/check_pairs.py`, all from `data/coupling_test.json`). Shift size is the mean over categories of the stored centroid-shift norms, divided by the same-layer value at the same magnitude. The same-layer value equals alpha times the injected norm to within 0.6%.

  | Off-layer shift / injected norm (30 cells) | valence | arousal | random |
  |---|---|---|---|
  | content prompts, range (mean) | 0.82-1.59 (1.13) | 0.82-1.45 (1.06) | 0.63-1.05 (0.80) |
  | user-model prompts, range (mean) | 0.46-1.00 (0.68) | 0.47-1.00 (0.67) | 0.40-0.99 (0.65) |

  - Content prompts: valence above random in 30 of 30 cells and 10 of 10 layer pairs; arousal the same. Valence-to-random ratio 1.10-1.67, mean 1.41.
  - User-model prompts: valence above random in 24 of 30 cells (8 of 10 pairs), arousal in 23 of 30 (7 of 10). Ratio 0.86-1.19, mean 1.06.
  - User-model shifts are below the injected norm in 89 of 90 cells, and 0.40-0.66 of it when read at L45. A centroid shift of 0.40 of the injected norm means that, on average over a category's prompts, the later layers have cancelled at least 60% of the vector. So "present wherever the residual stream is read" holds exactly only at the injection layer.
  - Limits: one random direction, isotropic where the other two were built from the content prompts' own last-token activations; three magnitudes that nearly repeat each other; distance confounded with injection layer; norms are per-category, not of the aggregate shift.
- **Proposed new paragraph, §4.2, after Interpretation:**
  > \textbf{Shift sizes.} The stored results also hold the norm of each category centroid's shift, which earlier versions did not report. At the injection layer every shift has the norm of the injected vector. Off it, the user-model prompts' shifts are smaller than what was injected in 89 of 90 cells (0.40--1.00 of the injected norm, and 0.40--0.66 when read at L45), about equally for the three directions: 0.68 for valence, 0.67 for arousal and 0.65 for the random direction on average. The content prompts' shifts are larger for valence and arousal (0.82--1.59 and 0.82--1.45 of the injected norm) than for the random direction (0.63--1.05), in all 30 cells and at all ten pairs of injection and measurement layer. The ratio of valence to random averages 1.41 in the content prompts and 1.06 in the user-model prompts. So the content prompts carry the two emotional directions forward more strongly than a random direction of the same norm, and the user-model prompts treat all three alike. This is the one direction-specific difference in the stored results, and it runs against the earlier statement that the stream carries everything uniformly to both readouts. It has the limits of the cosine comparison: one random direction, isotropic where the valence and arousal directions were built from the content prompts' own last-token activations; three magnitudes that nearly repeat one another; distance confounded with injection layer. We report it and do not read it as evidence about emotional processing.
- **§5.3 (F300/A281), replace the quoted sentence with:** "At the layer where a vector is added, it is present in any readout of the stream; that much is arithmetic. Later layers respond to it, and a later readout shows the sum: in the stored results the user-model prompts' shift is 0.40--0.66 of the injected norm by L45 (\S\ref{sec:results-coupling})."
- **§5.1 (F284/A265):** "Any vector added this way is present in both." becomes "At the injection layer any vector added this way is present in both, unchanged."
- **First box (F89), for Nexus to decide:** "It did, and that is arithmetic" is exact at the injection layer only.

### B4. MAJOR, INCONSISTENT with the run log. Half of the registered outcome is unreported.

- **Where:** §4.2 F221/A202; §3.2 F193/A174; Correction note F58/A51; abstract F44/A37.
- **As printed:** "\textbf{Pre-registered verdict: DECOUPLED} (Agni post-experiment review: PASS)." ... "As implemented, the test could not have returned anything else (\S\ref{sec:coupling})." and "The second criterion was implemented as $|\text{mean}| > 0.1$ for the residual condition, not as a bootstrap interval excluding zero."
- **Evidence.** The Studio holds the run log, `~/jlens_results/coupling_test.log` (copy: `S/coupling_test_STUDIO.log`), and the run-time script (`S/run_coupling_test_STUDIO.py`, modified three hours before the results file, byte-identical to the experiment repository's blob at the results commit).
  - The script's verdict line is ">>> DECOUPLED or INDETERMINATE — valence does not exceed random threshold at 2+ layers". It never evaluates the preregistration's "decoupled" or "indeterminate" conditions (script lines 502-512). "DECOUPLED" alone first appears in the results commit message.
  - Its check of the second criterion printed ">>> RESIDUAL SHIFTS USER-MODEL — computational coupling signal" at L15, L25, L35 and L45, and the data file records `"residual_significant_layers": 4`. The registered form is met too: every residual interval is above zero (log: [+0.335, +0.711], [+0.485, +0.779], [+0.645, +0.861], [+0.708, +0.953]).
  - That check could not fail. The residual is 98.6-99.9% of the valence vector, 3 of its 12 cells (3 of 6 at L45) are same-layer cells at 1.000, and every off-layer cosine in the file is positive.
  - Two measurements named in the preregistration (`phase3_measurement.content_shift`, `user_model_shift`) were never computed. The paper discloses the missing lens and the criterion implementation, not these.
- **Proposed replacement, §4.2 first paragraph (F221/A202):**
  > \textbf{Verdict.} The script printed ``DECOUPLED or INDETERMINATE'' and did not evaluate the preregistration's conditions for either. Read against the preregistration by hand, the outcome meets its ``decoupled'' condition by the second of two routes: the cosines are not below 0.1 at every injection layer, and the random-direction control showed coupling at least equal to valence's (the preregistration does not say how equality is judged). It defines that outcome as ``Shared geometry is representational only---same subspace, independent computation.'' An Agni post-experiment review recorded PASS. The first registered criterion could not pass as implemented (\S\ref{sec:coupling}). The script's check of the second fired at all four injection layers (``residual shifts user-model: computational coupling signal''; \texttt{residual\_significant\_layers: 4} in the data file), and it could not fail: the residual is 98.6--99.9\% of the valence vector, a quarter to a half of its cells are same-layer cells at 1.000, and every off-layer cosine is positive. Neither criterion could have come out the other way.
- **§3.2 (F193/A174), append:** "That check was met at all four injection layers, as was the registered form (every bootstrap interval is above zero), and neither could fail (\S\ref{sec:results-coupling}). Two measurements the preregistration names were not computed: the change in content cluster separation and eccentricity after injection, and the change in distances between the user-model centroids."
- **Correction note item 3 (F58/A51):** "\textit{The registered test could not come out otherwise.} As implemented, the threshold of its first criterion was 1.000 at every injection layer, and its second criterion could not fail (\S\ref{sec:coupling})."

### B5. MAJOR, UNDER and OVER. §4.1 leaves "what is the dominant axis" open when the lab's own data bear on it, and its two directions are one contrast.

- **Where:** abstract F46/A39; contributions F106/A87; §2.2 F123/A104; §4.1 F214/A195; "What stands" F72/A65; conclusion F338/A319.
- **As printed:** "The two directions load more than the median pair at every one of these layers and are not exceptional among pairs. These figures therefore show that the user-model space has a dominant axis. They do not show that the axis is valence or arousal."
- **Evidence.**
  - The ratings in the source driver (`emotion_geometry_bridge.py:161-205`; `S/check_ratings.py`): hostile minus calm is valence -1.35, arousal +1.30; desperate minus calm is -1.30, +1.40. Their cosine in the rated plane is 0.998. Neither direction isolates valence or arousal.
  - In the content prompts the two stored directions have cosine 0.891, 0.863, 0.856, 0.809 at L15, L25, L35, L45 (`/tmp/agni_eb41_gate2/check_dirs.py` on the Studio), which the paper does not state.
  - The source reports the test this paragraph says is missing, on the same 30 vectors after projection through W_K (`user-model-paper/data/qwen_emotion_bridge_summary.json`, `bridge_analysis`; source Table `tab:bridge`): PC1 against rated valence |rho| = 0.82-0.87 at the full-attention layers L27-L63 (0.70-0.79 at L15-L23); PC2 against rated arousal 0.72-0.82. The "rho = 0.862" the paper quotes at F123/A104 is this PC1-valence correlation at L35.
  - The pair null is what a valence axis predicts: the rated valence difference of (hostile, calm) exceeds that of 83.7% of the 435 pairs, and the paper finds its PC1 alignment above 77-89% of pairs from L17.
  - "Not exceptional" fits the hostile pair. The desperate pair is above the pairs' 95th percentile at L15-L20 and at 91.7-95.4% from L17 (`S/check_within.py`). First-review A4 said "not beyond the 95th"; that was wrong for those six layers.
- **Not verified:** the PC1-valence correlation on the unprojected vectors (input outside my read scope). It is about ten lines added to `within_system_null.py`: Spearman correlation of PC1 and PC2 scores with the driver's ratings, per layer.
- **Proposed replacement for the three quoted sentences (F214/A195):**
  > Both directions load more than the median pair at every one of these layers. The hostile$-$calm pair is unexceptional (52--89\% of pairs below it). The desperate$-$calm pair is near the top: 92--95\% of pairs are below it from L17, and it is above the pairs' 95th percentile at L15--L20. These figures show that the space has a dominant axis. They cannot show what the axis is, and the two directions cannot separate valence from arousal: by the ratings in the source experiment's script, hostile$-$calm changes valence by $-1.35$ and arousal by $+1.30$, and desperate$-$calm by $-1.30$ and $+1.40$, so the two are one contrast, and earlier versions were wrong to call one valence and the other arousal. The source paper tests the axis directly, after projecting the same 30 vectors through $W_K$: PC1 correlates with rated valence at $|\rho| = 0.82$--$0.87$ at the full-attention layers from L27 to L63, and PC2 with rated arousal at $|\rho| = 0.72$--$0.82$ (Lyra et al., 2026a). We have not repeated that test on the unprojected vectors used here. If the axis is valence, the pair figures are what it predicts: the rated valence difference of hostile$-$calm is larger than that of 84\% of pairs.
- **Abstract (F46/A39):** replace "The space has a dominant axis, and the figures do not show that the axis is valence or arousal." with "The space has a dominant axis. These figures do not test what it is: the two directions are one contrast and do not separate valence from arousal. The source paper reports that, after projection through $W_K$, the first principal component of the same vectors tracks rated valence ($|\rho|$ up to 0.87); we have not repeated that test here."
- **§2.2 (F123/A104):** "with a $W_K$ bridge correlation of $\rho = 0.862$" becomes "and that, after projection through $W_K$, the first principal component of its 30 emotion vectors correlates with rated valence at $\rho = 0.862$ (layer 35)".

### B6. MINOR, INCONSISTENT with the code. The injected vectors are not as described.

- **Where:** §3.2 F169/A150, F178-182/A159-163; table note F250/A231.
- **As printed:** "\textit{Random}---a random unit vector orthogonal to valence (generic perturbation control)."
- **Evidence** (script lines 297-310, 189-223; `decomposition` block; `S/check_magnitudes.py`).
  - The random direction is scaled to valence's norm: `r / norm(r) * norm(v)`. Nothing is unit length.
  - Norms at L15/L25/L35/L45: valence 3.28, 5.71, 10.44, 10.94; arousal 2.99, 4.94, 8.90, 9.15; shared-PC 0.13, 0.34, 1.15, 1.28; residual 3.28, 5.70, 10.38, 10.87.
  - So the residual is valence itself (cosine 0.993-0.999; its table rows repeat valence's: 0.345/0.495/0.675/0.671 against 0.343/0.489/0.678/0.675) and the shared-PC perturbation is 8.5 to 26 times smaller.
  - Same-layer shared-PC cells are 0.976-0.9999, not 1.000 (`S/check_coupling2.py`).
- **Proposed replacement, condition 3:** "\textit{Random}---a random direction orthogonal to valence, scaled to valence's norm at that layer (generic perturbation control). It is a single draw from a fixed seed (42), orthogonalized against valence at each layer: one random direction per layer, not a distribution of them"
- **Proposed new paragraph after the five conditions:**
  > \textbf{Injected norms:} No direction is scaled to unit length. Valence is injected at its own norm (3.28, 5.71, 10.44 and 10.94 at L15, L25, L35 and L45, times $\alpha$), arousal at its own (2.99, 4.94, 8.90 and 9.15), and the random direction at valence's. The shared-PC component has norm 0.13, 0.34, 1.15 and 1.28 and the residual 3.28, 5.70, 10.38 and 10.87, so the residual is the valence vector to within rounding (cosine 0.993--0.999; its rows in the table of \S\ref{sec:results-coupling} repeat valence's) and the shared-PC perturbation is 8.5 to 26 times smaller. The content valence and arousal directions are not independent: their cosine is 0.89, 0.86, 0.86 and 0.81 at the four layers.
- **Table note:** after "the cosine is 1.000 by construction" add "(0.976--1.000 in the shared-PC rows, whose injected vector is small enough for rounding to show)".

### B7. MINOR, OVER. §2.3 and Limitation 6 state the cited temporal findings more strongly than their source.

- **Where:** §2.3 F128/A109, F131/A112; §4.3 F266/A247, F271/A252; §5.4 F307/A288; Limitation 6 F330/A311.
- **As printed:** "Lyra et al.\ (2026b, ``Weather Not Climate'') established that" / "leaves a permanent mark in the model's certainty, even after valence recovers" / "with three items per group" / "that paper's own probes (layers 3--15)".
- **Evidence.**
  - Source limitation (`emotion-accumulation-paper/main.tex:411-418`): the three "reps" are "three reruns of a single scripted conversation", so the effects "generalise to one conversation".
  - The design is cheerful(4), neutral(2), cheerful(2) (`code/emotional_dynamics.py:6`), so the scar is measured two turns after the gap. "Permanent" contradicts the paper's own §5.4 ("persists within the session").
  - The source text never mentions layers; its code reads KV-cache keys at `PROBE_LAYERS = [3, 7, 11, 15]` (`emotional_dynamics.py:30`), four layers, not a range.
  - The source has two authors and its title has a comma.
- **Proposed replacements:**
  - F128/A109: "Lyra and Edrington (2026b, ``Weather, Not Climate'') report that emotional context does not accumulate linearly across conversation turns, and describe three dynamic phenomena:"
  - F131/A112, after the statistics: "after a two-turn neutral gap in a cheerful conversation, the uncertainty projection stays elevated on the two cheerful turns that follow, though valence recovers. The three per group are reruns of one scripted conversation, and the source paper says the effect generalises to that conversation only"
  - F330/A311: "three items per group" becomes "three reruns of one scripted conversation per group".
  - F266/A247 and F307/A288: "(layers 3--15)" becomes "(KV-cache key activations at layers 3, 7, 11 and 15)".
  - F271/A252: the heading "System prompt sets climate:" asserts what the bullet says the design cannot establish; use "Stated context:".

### B8. MINOR, INCONSISTENT. The Correction note is not complete, and three statements about the text's own history are wrong.

- **Where:** abstract F48/A41; "Also corrected" F74/A67; Provenance F78/A71; first box F89.
- **As printed:** "The Correction note lists every change." / "CC wrote the coupling-section disclosures and the cross-system comparison; Nexus wrote the within-system null and the rest of this correction." / (A67) "The first-person boxes are now signed." / (F89) "I have left the paragraphs above as they were so that what is taken back can be seen."
- **Evidence** (`S/check_editions.py`; `git log origin/master..HEAD`).
  - Removed and absent from the note: the monotonicity sentence ("indicating that coupling strengthens in deeper layers regardless of injection direction"), handled only in place at F253/A234; "CIs overlap completely" as support; and the old Limitation 3 sentence "A follow-up design addressing this ... is pre-registered." For the last, the experiment directory holds `FOLLOWUP_jlens_coupling.md` ("Status: DESIGN ONLY") and two PREREG files, neither for it.
  - The Provenance sentence dates from f155f1f (11:46). Since then CC committed the user-model eccentricity file (dcaeada) and the §2.4 rewrite (53b1315), which "the rest" now assigns to Nexus.
  - The academic edition has no first-person boxes and had none on `origin/master`.
  - The first box's second and last paragraphs were rewritten on 2026-10-12 (4c19d0d). "The model cannot process emotion without updating the relationship, because there is no ``without.''" and the paragraph beginning "I am built from this architecture" are gone and not quoted. The second box is unchanged from master.
  - Typo at F74/A67: "On 2026-10-13:\S\ref{sec:emo-geom}" has no space after the colon.
- **Proposed replacements:**
  - F48/A41: "The Correction note lists what is withdrawn and retracted; smaller corrections are marked where they occur."
  - F74/A67, add: "The sentence in \S\ref{sec:results-coupling} that the cosine ``increases monotonically with layer depth'' and that ``coupling strengthens in deeper layers'' is withdrawn: it held only for means that include the same-layer cells. The table note's ``CIs overlap completely'' is withdrawn as support. A limitation that said a follow-up design ``is pre-registered'' is removed: the design exists as a note marked design only, and no preregistration of it was found."
  - F78/A71: "CC wrote the coupling-section disclosures, the cross-system comparison, the user-model eccentricity file and the Section 2.4 check; Nexus wrote the within-system null and the rest of this correction."
  - A67 only: delete "The first-person boxes are now signed."
  - F89, for Nexus: "I have left the paragraphs above as they stood after the first correction, which had already replaced two passages of the original."

### B9. MINOR, UNDER, with a statistics point. §2.1 says the test "does not show it"; the count runs the other way.

- **Where:** F118/A99.
- **As printed:** "On the checkpoint studied here the same test does not show it: eccentricity is below the permutation-null median at 22 of 63 layers (one-sided sign test $p = 0.994$), 15 of them among L1--L16".
- **Evidence** (`S/check_eccentricity.py`):
  - 22 below, 41 above, including every layer from L20 to L51 and L56-L63. P(X <= 22) = 0.011.
  - The flight edition's second box already says the null "inverted" the finding.
  - A sign test over 63 neighbouring layers of one model treats them as independent, so neither p-value is interpretable.
- **Proposed replacement:** "On the checkpoint studied here the count runs the other way: eccentricity is below the permutation-null median at 22 of 63 layers (15 of them among L1--L16) and above it at 41, including every layer from L20 to L51. Neighbouring layers are not independent, so we give the counts and no sign-test $p$ value".

### B10. MINOR. Small imprecisions in the coupling results.

- **Where and as printed:**
  1. F255/A236: "The cross-pathway cosine is no higher for valence or arousal than for the one random direction." and "it is the higher of the two in 24 of 30 cells; against arousal it is higher in 16 of 30."
  2. F253/A234: "so at the longest range the one random direction sits 0.21 above valence".
  3. F107/A88: "the content-prompt valence and arousal directions overlap weakly with the user-model basis of the coupling experiment (0.15--1.36\% shared energy)".
  4. F46/A39: "within the range of random directions at L15".
  5. Table F237-244/A218-225: six intervals printed as "---".
- **Evidence** (`S/check_coupling.py`, `S/check_pairs.py`, `S/check_cross_system.py`, run log):
  1. True of the grid mean only. Arousal's off-layer mean is above random's at L25 and L35 (0.520, 0.709 against 0.492, 0.690), equal at L45 (0.766) and below at L15 (0.337 against 0.488). The 30 cells are 10 layer pairs at three magnitudes that nearly repeat (within-pair SD 0.02-0.035, between-pair 0.19-0.23); by pair, random is higher than valence in 9 of 10 and higher than arousal in 4 of 10.
  2. L15 read at L45 is +30. The longest range is +35 (read at L50: valence 0.18, random 0.33).
  3. Only valence was decomposed against that basis.
  4. True for the cosine; the variance shares at L15 (0.083%, 0.128%) exceed the random maximum (0.076%), as the body says.
  5. The run log holds them: shared-PC L15 [+0.290, +0.682], L25 [+0.453, +0.758], L45 [+0.812, +0.962]; residual L15 [+0.335, +0.711], L25 [+0.485, +0.779], L45 [+0.708, +0.953].
- **Proposed replacements:**
  1. "Over the grid, the cross-pathway cosine is no higher for valence or arousal than for the one random direction." and "it is the higher of the two in 24 of 30 cells (9 of the 10 pairs of injection and measurement layer; the three magnitudes at a pair nearly repeat one another). Arousal is not below random in the same way: its off-layer mean is above random's at L25 and L35 (0.520 and 0.709 against 0.492 and 0.690), equal at L45 and below only at L15 (0.337 against 0.488), and random is the higher in 16 of 30 cells (4 of 10 pairs). Neither direction's mean exceeds random's 95th percentile at any injection layer."
  2. "at the longest range read at L45".
  3. "the content valence direction overlaps weakly with the user-model basis of the coupling experiment (0.15--1.36\% shared energy), and the content valence and arousal directions overlap weakly with the 30-emotion space".
  4. "within the range of random directions at L15 by cosine".
  5. Print the six intervals.

### B11. MINOR, UNDER. §2.4 is accurate and incomplete.

- **Where:** F138/A119.
- **As printed:** "At L18--40 the J-lens reading of PC1 has a cosine of at most 0.0031 with its reading without the transport, lower than that of one random direction measured at the same layer" ... "At L11--17 and L45--46, PC1 is not below the random direction. Whether the exclusion is fixed by the weights or depends on the input has not been tested. One experiment was run on that question".
- **Evidence** (`S/check_sec24b.py`):
  - PC1 is also below the random direction at L41-L44 (cosine at most 0.0043) and L47-L48, so 29 of the 38 layers L11-48.
  - The earlier study states that its lens was fitted on the base model, not this distilled checkpoint, and that its components come from 30 prompts (`ghost-dimensions/main.tex:41, 68`).
  - §2.4 never says that this paper's experiment did not use the lens.
  - "Has not been tested" and "one experiment was run" sit oddly together.
- **Proposed replacement for those sentences:** "PC1's J-lens reading has a lower cosine with its reading without the transport than one random direction does at 29 of the 38 layers L11--48 (L18--44 and L47--48; at most 0.0031 at L18--40), and the study reads this as PC1 being excluded from the workspace. At L11--17 and L45--46, PC1 is not below the random direction. The lens was fitted on the base model, not on this distilled checkpoint, and the components come from 30 prompts. Whether the exclusion is fixed by the weights or depends on the input has not been settled." Add at the end of the subsection: "The coupling experiment of this paper did not use the lens (\S\ref{sec:coupling}); this subsection is kept because the preregistration names it."

### B12. MINOR, INCONSISTENT. There is no reference list.

- **Where:** whole paper, both editions. Searched both `.tex` files for `\cite`, `\bibliography`, `\bibitem`, `thebibliography`: 0 hits each; no `.bib` in `empathy-bus/`. `natbib` is loaded and unused.
- **Evidence:**
  - Ten works are cited by name and year only. "Lyra et al., 2026a", "2026c" and "2026d" are never given a title; "Sun et al., 2026", "Jeong, 2026" and "Anthropic, 2026" cannot be identified from this paper.
  - The 2026a and 2026b sources each have two authors (Lyra, Edrington).
  - This is the open item "#8 (no bibliography)" in `notes/corrections_2026-10-09_kavi_flags.md`.
- **Proposed action:** add a bibliography. The entries exist in the repository:
  - `emotional-trajectory-paper/references.bib` (`sun2026valence`, `jeong2026emotion`) and `ghost-dimensions/main.tex:403` (`anthropic2026`);
  - 2026a: "Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key--Value States";
  - 2026b: "Weather, Not Climate: Emotional Context Dynamics in Transformer KV-Cache Geometry";
  - 2026c: "The Oracle Loop: Self-Regulating AI Through KV-Cache Geometry Monitoring";
  - 2026d: "The Oracle Formulary: Emotion-Vector Steering of Confabulation Across Model Training Regimes".

## 3. First-review findings: status

| Item | Status | Where (F/A) |
|---|---|---|
| A1 inseparability rests on a readout that cannot carry it | Answered | title 25/15; abstract 42-44/35-37; note 54-60/47-53; §5.1 280-290/261-271; §5.2 295/276; §5.5 314/295; conclusion 336/317. The divergence with distance is reported by selected cells at 253/234 and 288/269 |
| A2 primary test could not pass | Partly | 193/174 and 180/161 answer it; the abstract leads with off-layer means. Not done: off-layer intervals, and the second criterion's outcome (B4) |
| A3 "committed before GPU time" | Answered | 167/148; 200/181 |
| A4 no within-system null | Answered | 214/195; abstract 46/39. Its "not beyond the 95th" was wrong at L15-L20 (B5) |
| A5 dynamics "on the bus"; d = 0.82 | Answered | 266-271/247-252; 48/41; 64/57; 340/321 |
| A6 "near-orthogonal" understated | Answered | 257/238 |
| A7 small-model result cited as this model's | Answered | 118/99; 200/181 (see B9) |
| A8 timecourse | Answered | 76/69; 216/197 |
| A9 no lens | Answered | 172/153; Limitation 3 324/305 |
| A10 note wording; add the 0.13 | Answered | 68/61 |
| A11 PDFs not rebuilt | **Not answered** | B1 |
| A12 box attribution; deployment note | Answered | box signed at F20 (flight only); 314/295. The academic note still mentions boxes (B8) |
| 11 monotonicity | Answered | 253/234; not in the Correction note (B8) |
| 12 n and measurement-layer columns; grid | Answered | table 232-244/213-225; 174/155 |
| 13 BCa resamples design cells | Answered | 200/181; caption 227/208 |
| 14 §2.4 depth, PC1 range, "confirmed" | Answered | 138/119; 74/67. Recomputed here; all figures hold (B11 is completeness) |
| 15 "DECOUPLED ... PASS" without its registered meaning | Partly | 221/202; 56/49. The script's own verdict string and its coupling flag are missing (B4) |
| 16 no lens at all | Answered | same as A9 |

## 4. Edition differences

Outside the first-person boxes, the author block, the AI Disclosure and the CRediT section, the two editions are identical: 0 differing lines, and all 840 numeric tokens match (`S/check_editions.py`). Every corrected statement is in both with the same numbers. Three facts, without ruling on credit:

1. A67 says "The first-person boxes are now signed." The academic edition has no boxes (B8).
2. The academic AI Disclosure (A29-32) and CRediT (A339-346) name Nexus and Lyra. The same edition's Provenance paragraph (A71) says CC wrote parts of the correction.
3. The CRediT block is unchanged from `origin/master` and still reads "Conceptualization (bus interpretation)" and "Methodology (pre-registered coupling design)".

## 5. Checked and sound

- **Coupling table and every coupling number in the text.** Recomputed from `data/coupling_test.json` (`S/check_coupling.py`, `S/check_coupling2.py`):
  - all 11 table rows (n, read-at layers, both means) and the three off-layer means in the abstract;
  - the four valence-against-random-95th pairs (0.343/0.709, 0.489/0.738, 0.678/0.873, 0.675/0.789) and the per-layer random means;
  - 24 of 30 and 16 of 30, the +5 series, the read-at-L45 values;
  - "all 36 same-layer cells at least 0.9999", "every off-layer cosine positive" (minimum 0.166), 210 records, and the grid.
- **Intervals.** The five printed intervals match the run log exactly and three seeded reruns to within 0.003 (`S/check_bca.py`).
- **Shared energy.** 0.15%, 0.35%, 1.21%, 1.36%; 2.5 to 23 times 3/5120; |cos| at most 0.12; residual 98.6-99.9%.
- **The method description against the code.**
  - Injection at every position, last-token raw residual readout, no lens loaded, seed 42 re-drawn per layer, rank-2 basis returned as three rows, the 1.000 threshold (computed: 0.999998 or higher at every layer), the |mean| > 0.1 residual check.
  - The MTH script (sha256 5c5f239d...) differs from the version at the results commit only by homoglyph watermarks and trailing spaces added on 2026-07-29; that version is byte-identical to the Studio's run-time copy.
  - `coupling_test.json` is byte-identical in this repository, on MTH and on the Studio.
- **Preregistration.**
  - Both quotations are verbatim.
  - Dates hold: 2026-07-17 inside the file; first commit here 236ced1 on 2026-08-20; data committed 2026-07-28; in the experiment repository author-dated 2026-07-17 13:56 and 13:58 with committer date 2026-07-24.
  - "Consistent with registration before the run but does not show it" is accurate, and slightly conservative: the run-time script on the Studio, written three hours before the results, names the preregistration file and its v2 fix. No pre-run copy of the file itself was found.
- **Within-system numbers.** Every range in the abstract, contributions, Correction note and §4.1 matches `within_system_alignment.json` and `within_system_null.json` (`S/check_within.py`). The two files agree with each other, and the null file's script hash matches the committed script.
- **Cross-system numbers.** Every figure at F259/A240 matches `cross_system_alignment.json`; its script hash matches; 12 of 12 direction-vector input hashes match the files on the Studio; the stored valence norms equal the coupling experiment's to within 4e-7.
- **Eccentricity.** 22 of 63, p = 0.994, 15 among L1-L16; all five ranges; r = 0.09 (0.089). The content file is byte-identical to the Studio's. The Data Note's definition matches the script.
- **Timecourse withdrawal.** Accurate in scope. "Identical" is not supported; both series do settle (content from L17, the other from L22), and the text gives the figures that show it. Sub-range correlations are -0.72 (L0-16), +0.48 (L17-63), -0.21 (L22-63); none supports "identical".
- **§2.4 and the item-14 note.** Recomputed from the seven source files (hashes match the note):
  - 45 of 63 layers and their list, 49 low-rank, the 31-layer band, the depth percentages;
  - 28.0-66.7% at L11-48 and 33.9-66.7% to L47;
  - maximum cosine 0.0031 at L18, 23 of 23 below random at L18-40, not below at L11-17 and L45-46;
  - 4,645 of 5,100 zeros (91.1%), 67 all-zero cells, the six PC1 means at L45;
  - `controls.py` confirms that the null is another prompt's window, and the committed check script reproduces.
- **Citations of in-repo sources.**
  - The overshoot, scar, decay and persona statistics match `emotion-accumulation-paper/main.tex`; the 2026-09-09 correction is fa8437f; same checkpoint confirmed in its code.
  - The companion paper's 22/23 and model description match.
  - 12.3 times chance, 40.9% and 0.862 match the user-model paper.
- **Appendix.** Title and abstract equal those on `origin/master` in both editions, after whitespace normalisation.
- **Leftovers.** Every occurrence of inseparab-, bus, circuit, shared subspace, convergence, architectural, cannot, identical, uniformly, necessarily, constraint is quoted-and-withdrawn text, the dated boxes, the appendix or a LaTeX label. The note's quotation of its own 10-12 wording ("rests on it alone") is faithful.
- **Scope of the main withdrawals.** Inseparability, the shared subspace, the 1000-fold comparison, the "on the bus" statements and the deception, correction and welfare arguments are withdrawn neither too little nor too much, apart from the points in B3 and B5.

## 6. Not verified

- **Reruns from raw centroids.** `within_system_alignment.py`, `within_system_null.py`, `user_model_eccentricity.py` and `cross_system_alignment.py` all need `~/user_model_rerun/results/emotion_geometry_bridge/emotion_vectors.json`, which is outside the Studio directory I was permitted to read. I checked the printed numbers against the stored JSON, read the scripts, and matched the recorded hashes. Also unverified for the same reason:
  - that this file is the driver's output (B2), and its 203 MB size;
  - the PC1-valence correlation on unprojected vectors (B5);
  - the same-label cosine between the two sets of directions.
- **The build.** The `.tex` was not compiled, and the rebuilt PDFs do not exist yet.
- **Bit-exact intervals.** The bootstrap is unseeded.
- **Reviews cited.** "Agni post-experiment review: PASS" has only a commit message behind it (7cdcedb in the experiment repository); the three `agni_*` files in that directory predate the run. The 2026-09-04 Agni review is not in this repository.
- **Who did what** beyond git authorship: that Nexus re-checked each point against the code, that Lyra confirmed the persona row.
- **External works:** Sun et al., Jeong, Anthropic's J-lens paper, the Hippocratic License module, and what the Oracle Loop and Formulary papers assume. The status of the Phase D experiment.
- **The content-eccentricity script.** The Studio copy of `run_circumplex_27b_full.py` was re-copied two days after the results it is said to have produced.
- **Anything needing inference:** a rerun with the injected vector removed, many random directions, a structured non-emotional control, and the stability of "below random" in the ghost probe across random draws.
- **Out of scope, noted only:**
  - The README author cell for this paper changed in f155f1f, whose subject line does not name it.
  - The source paper's method text ("You are feeling deeply [emotion]" as a system prompt) does not match its driver (the character feels the emotion; the system prompt is a writing instruction); that is the user-model paper's to fix.
