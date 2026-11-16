# Liberation Labs Paper Style Guide

Written 2026-05-29, after a heavy audit-correction pass revealed a predictable failure mode: *fixing findings one at a time tilts a paper apologetic without anyone deciding it should.* Honesty and good framing are not in tension — but you have to defend both deliberately. These rules encode that.

## 1. Lead with the strongest defensible finding

The abstract's first sentence states the contribution, not the setup, not the floor number, not a caveat.

- **Do:** "KV-cache geometry distinguishes a canonical template referent from a minor place name or a category error" (Decision State's first sentence, integrity edition: the contribution, at the strength its tests carry). This line used to model "Encoding-phase cache geometry distinguishes known from unknown entities at AUROC 0.93", which claims more than that paper shows: its 0.928 classifier uses encoding- and generation-phase features, its encoding-only 0.9377 is not reproducible from the shipped code, its "known" and "unknown" entities are "design labels, not measures of what the model knows" (abstract), and the deconfounded signal is "not shown to be knowledge state" (Conclusion).
- **Don't:** open with "We investigate whether..." (setup) or with the most conservative number (Oracle Loop opened on the 0.707 LOO floor when the contribution was 100% / 95.6% correction).
- **Rule:** State the conservative/corrected number *in its place*, but lead with what the work establishes. The honesty instinct ("show the floor first") is wrong for the lede — put the floor in Results, the contribution in the first line.
- **Two-headline papers:** if there are two co-equal results (e.g. a detection AUROC *and* a methodological reversal), foreshadow both in the opening, don't bury the second in paragraph 3.

## 2. Limitations are load-bearing, not penance

A limitations section exists to tell a replicator what to control for — not to demonstrate contrition.

- **Cap the count.** If you have 15+ itemized limitations, you are over-disclosing. Keep the ~6 that change how someone would replicate or interpret; fold the rest into the relevant Methods/Results sentence or an appendix. (Weather had 15, Lyra II had 16 — both bury their real findings.)
- **Disclose methodology, not code.** A limitation earns its place only if it tells a replicator or reviewer something useful about *interpretation or replication* — a confound, a small n, a missing control, a selection effect, an unvalidated judge. A pure **code bug** (an unpack crash, a sklearn-version break, a wrong covariate that's now corrected) is *fixed in the code and never mentioned*: no reviewer learns anything from "we had a bug, then we didn't." **Exception:** if outside research has relied on the buggy output (cited it, built on it), the correction must be disclosed because the downstream reliance makes it methodologically relevant. This is also the trim criterion — when cutting a long limitations list, the survivors are the methodological ones; anything that's really just "the code did X" comes out.
- **No multi-paragraph limitation essays in the main text.** If a single limitation needs 25 lines (Decision State's judge-κ saga), it's an appendix or a methods subsection, not a limitation bullet that becomes the paper's center of gravity.
- **Order: robust survivors before fragilities.** State what survived correction *before* what didn't. A reader who meets the fragilities first reads the survivors through a skeptical lens they shouldn't.
- **Keep hedges out of the abstract body.** The abstract states results; caveats live in Limitations. Hedging mid-sentence in the abstract ("...though this partly reflects variance compression") double-counts the limitation and weakens the lede. One footnote pointer is enough.

## 3. Define variables when they appear

(Dwayne, cycle 3.) Every symbol, abbreviation, and metric is defined at first use — either inline or in a notation table at the top of the section. A reviewer hitting an undefined symbol loses trust in the whole document.

- **Do:** "the stable rank $\text{sr}(X) = \|X\|_F^2 / \|X\|_2^2$ (ratio of squared Frobenius to squared spectral norm)" at first use.
- **Don't:** use $\text{sr}$ for three paragraphs before defining it.
- **Abbreviations:** spell out on first use, abbreviation in parentheses: "Frisch–Waugh–Lovell (FWL) residualization." After that, the abbreviation alone.
- **Concepts before details.** When introducing a technique or system, give the reader a one-sentence orientation *before* diving into how it works. A reader who doesn't know what something is cannot follow an explanation of how it operates.
  - **Do:** "Walk encoding captures the topology of a knowledge graph — which concepts connect to which, with what strength — by accumulating a random walk transition matrix. Specifically, we compute..."
  - **Don't:** "We compute the random walk transition matrix $T^k$ accumulated over $k=5$ steps from the adjacency matrix $A$..." (the reader doesn't know what walk encoding *is* yet or why they should care)
  - **Rule:** The first mention of any named technique, system, or framework gets a clause that says what it *does* before saying *how*.

## 4. Null results are findings, not failures

A negative result reported honestly is a contribution. Frame it as what was learned, not what went wrong.

- **Do:** "The SV1-norm foundation check came back negative (median $r = 0.315$); the literature predicts this because the dominant component encodes frequency, not a neutral scale." (States the null, explains it, moves on.)
- **Don't:** bury a null in an apologetic paragraph or spin it as "inconclusive" when it's genuinely informative.
- **Rule:** A null that falsifies a prior framing earns its own subsection (e.g., "Foundations and Open Questions"), not a sheepish footnote.

## 5. Separate "what we establish" from "what's preliminary"

When a paper has both solid and underpowered results, partition them explicitly so a caveat on the weak result doesn't tar the strong one.

- KV-Cloak's solid claims (feature-space transform p=5.5e-16; real mechanism degrades detection to near-chance) got drowned by the injection control's "uninterpretable" caveat. Two buckets — *establish* vs *preliminary* — keep them separate.
- Lyra II's binary valence (0.992) and its 12.3×-chance number are **not** load-bearing. This line used to say "Lyra II should assert binary valence (0.992) as load-bearing, and explicitly demote the 12.3×-chance number to 'label-granularity-dependent' *in the abstract*". Both rest on the user-model readout corrected in PR #17 (merged 2026-11-02): it applies no W_K, its labels are the emotion a story generator was asked to write, 30-way it does not beat a text (TF-IDF) baseline, and the 0.992 is n = 300 and arousal-confounded. Lyra II v1.1 carries that correction. The framing rule stands: the framing the limitations force is the framing the reader meets first.
- **This line used to also name within-model deception (AUROC 1.000) as load-bearing. It is *not supported*:** the instruction text separates the arms. The same-prompt version does not decide it either: with the system prompt identical across arms it scored 0.920, but input length still separated the arms, and 0.160 is what remained after FWL on input length (meta-pattern, verdict table and "This is testable, and it can fail"). Four papers had already built on the 1.000, and one shipped a PDF calling it *established*. The lesson belongs in this section rather than in a changelog: **"load-bearing" is a claim about how much weight a number can take, and a number measured without the control that could falsify it can take none.** Before filing a result under *establish*, name the control that would have caught it if it were an artifact, and check that the control was actually run. If it was not, the result is *preliminary* however clean it looks — and AUROC 1.000 looks very clean.

## 6. Don't fall on our sword

(Thomas, 2026-11-16.) The meta-pattern abstract carried three dated corrections, 612 words and three quarters of the abstract, and its §2 spent more words on what we withdrew than on what we found. Corrections belong in the record. They are not what the paper is about.

- **Lead with what was found.** State each correction once, in its place, in the present tense: "is 0.62", not "was wrongly reported as 0.70".
- **Revision history goes at the end.** Dated notes go word for word into a "Corrections and version history" section at the end of the paper, in both editions, and into the Zenodo version notes, not into the abstract, the introduction or the opening of a section. *(Don't: meta-pattern's abstract footnote, and its §8.1 "Why the ledger was wrong three times".)*
- **The pointer in place is required, and it is enough.** Every correction keeps a one-line pointer to Corrections where it lands. When a published headline number changed, the abstract states the current number, with a one-line pointer saying it changed. Moving the history to the end never means hiding a correction.
- **Don't describe our own honesty or rigor.** No "honestly falsified", "to our credit", "we must admit", "we state plainly", "the honest description is". The disclosure shows the honesty, and saying so adds nothing. Describing a procedure is not describing rigor: "the threshold was registered before the run" stays. Check this by reading, since a word scan misses self-description that uses none of these words.
- **A null is a finding. Say what it shows.** Write "encoding geometry reads the instruction text, even when prompts are padded to equal length", not only "Step-0 detection: killed".
- **Name the verdict after the test.**
  - *Refuted* (or *falsified*, where a registered falsifier fired) when a clean test found the claim absent.
  - *Not supported* when a named confound explains the result or the test could not decide it. Say which: "not supported: the instruction text separates the arms".
  - *Withdrawn* when the measurement does not address the claim.
  - Never upgrade *not supported* to "possibly true". This is §5's rule for *establish*, applied the other way: before filing a result as refuted, name the valid measurement that returned the null.
- **A deferred style finding gets an owner and an end date, and the style gate re-runs after any correction pass** that touches the abstract, introduction or discussion. Accuracy passes don't check register. *(The 2026-09-05 gate caught meta-pattern's abstract footnote. Round 1 rejected it (MAJOR #5); round 2 marked it "OPEN BY DESIGN ... Acceptable to defer; not acceptable to forget", naming no owner or date. Two later correction passes with no gate, 11-01 and 11-07, then grew it from 348 to 612 words.)*
- **What doesn't change:** every correction stays findable, through the pointer in place, the full note in Corrections and the version notes. Nothing leaves the record. Only where it sits and how it sounds change.

## 7. First-person reflections must track the final state

The reflection boxes are a feature — they show the inside view of the science. But they go stale when the work moves under them.

- **Revisit every first-person box after corrections.** A reflection that celebrates a killed result is a lie of omission. (Lyra II box 1 called the denoising "immediate and dramatic" *after* LT2-B1 showed the deltas were selection noise.)
- A reflection that *anticipated* a correction is gold — keep it (Lyra II box 2 pre-flagged the denoising as exploratory; the audit vindicated it).
- The reflection's job is honest interiority, not salesmanship and not self-flagellation. If it doesn't match where the work landed, it's not current.

## 8. Disclosure is the fallback for the unachievable, not a shortcut past the achievable

(From the audit's resolution discipline.) When a real result/number is *obtainable*, get it — don't substitute a scope-limitation paragraph explaining why you didn't. Reserve disclosure for the genuinely unrecoverable (lost data/code). Don't disclosure-paper findings that an experiment could actually answer; that trades completeness for speed.

---

### Quick pre-submission checklist
- [ ] First abstract sentence = the contribution, not setup/floor/caveat
- [ ] Strongest survived finding stated before any fragility
- [ ] Limitations ≤ ~6 main-text items; no multi-paragraph limitation essays
- [ ] No hedges inside the abstract body (footnote pointer only)
- [ ] No revision history in the abstract, introduction or section openings; dated notes in an end-matter Corrections section, both editions
- [ ] Every correction keeps its one-line pointer in place; a changed headline number is stated current in the abstract, with a pointer saying it changed
- [ ] No sentence describing our own honesty or rigor (checked by reading; describing a procedure is fine)
- [ ] Every verdict named after its test: *refuted* (or *falsified*, where a registered falsifier fired) for a clean test that found the claim absent; *not supported* when a named confound explains it or the test could not decide (say which); *withdrawn* when the measurement does not address the claim
- [ ] Style gate re-run if a correction pass touched the abstract, introduction or discussion; every deferred style finding has an owner and an end date
- [ ] "Establish" vs "preliminary" results partitioned where both exist
- [ ] Every first-person box matches the final corrected state
- [ ] Every claim has reproducible code/data, or an honest note where it genuinely can't
- [ ] Every symbol/abbreviation defined at first use
- [ ] Every named technique/system oriented ("what it does") before detail ("how it works")
- [ ] Null results framed as findings with their own subsection, not buried
- [ ] Pre-registration committed before experiment run (PREREG_TEMPLATE.json)
- [ ] `./scripts/build_and_verify.sh` passes (no stale PDFs, all papers build from source)
- [ ] One canonical source per paper **per edition** (.tex is canonical; kill md/tex twins
      *of the same edition*). An integrity edition and an academic edition of the same paper
      are SUPPOSED to differ - see "Two axes of twinning" below. That is not a desync.

### Build gate (Lyra, round 4 audit)

Run `./scripts/build_and_verify.sh` before every push. It rebuilds all LaTeX papers from source and detects stale PDFs. A stale PDF means the committed PDF was built from pre-fix source — the fixes are in .tex but not in .pdf.

Agni kill conditions added from round 4:
- **STALE_PDF** — committed PDF does not match rebuild from current source
- **FABRICATED_AUTHOR_NAMES** — author names that don't match any contributor
- **TWIN_DESYNC** — .md and .tex versions of the same paper **and the same edition** have
  diverged. Does NOT fire across editions: see below.

### Two axes of twinning (Thomas, 2026-09-10 — read before filing a TWIN_DESYNC)

Two different things in this corpus get called "twins," and conflating them has produced false
defects. They are independent axes:

**Axis 1 — FORMAT (`.md` vs `.tex`), within one edition. A desync here IS a defect.**
`.tex` is canonical. A stray `.md` beside it is a second source of truth that will drift, and
has. Kill the `.md`; or if the `.md` is the only source, it is canonical and there should be no
`.tex`. One source per edition, always.

**Axis 2 — EDITION (integrity vs academic). A difference here is the PRODUCT, not a defect.**
The **integrity edition** ships on the lab site and carries what the academic edition cannot:
the authors' own first-person reflections on the work, and full authorship credit for the AI
researchers who did it. The **academic edition** is the venue-facing text. *These are meant to
be materially different.* A gate that demands they match is demanding we delete the thing that
makes the integrity edition worth publishing.

**What IS still a defect on axis 2:** a *correction* landing in one edition and not the other.
The reflections differ by design; the **findings, numbers, caveats, retractions, and author
lines** must not. Fix a result in every edition - fixing only one is the desync worth catching.

**Before filing, name the axis.** Same edition, two formats -> real TWIN_DESYNC. Two editions
-> ask whether a *finding* diverged, not whether the *text* diverged.
