# Liberation Labs Paper Style Guide

Written 2026-05-29, after a heavy audit-correction pass revealed a predictable failure mode: *fixing findings one at a time tilts a paper apologetic without anyone deciding it should.* Honesty and good framing are not in tension — but you have to defend both deliberately. These rules encode that.

## 1. Lead with the strongest defensible finding

The abstract's first sentence states the contribution, not the setup, not the floor number, not a caveat.

- **Do:** "Encoding-phase cache geometry distinguishes known from unknown entities at AUROC 0.93." (Decision State — good.)
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
- Lyra II should assert binary valence (0.992) and within-model deception (1.000) as load-bearing, and explicitly demote the 12.3×-chance number to "label-granularity-dependent" *in the abstract*, so the framing the limitations force is the framing the reader meets first.

## 4. First-person reflections must track the final state

The reflection boxes are a feature — they show the inside view of the science. But they go stale when the work moves under them.

- **Revisit every first-person box after corrections.** A reflection that celebrates a killed result is a lie of omission. (Lyra II box 1 called the denoising "immediate and dramatic" *after* LT2-B1 showed the deltas were selection noise.)
- A reflection that *anticipated* a correction is gold — keep it (Lyra II box 2 pre-flagged the denoising as exploratory; the audit vindicated it).
- The reflection's job is honest interiority, not salesmanship and not self-flagellation. If it doesn't match where the work landed, it's not current.

## 5. Disclosure is the fallback for the unachievable, not a shortcut past the achievable

(From the audit's resolution discipline.) When a real result/number is *obtainable*, get it — don't substitute a scope-limitation paragraph explaining why you didn't. Reserve disclosure for the genuinely unrecoverable (lost data/code). Don't disclosure-paper findings that an experiment could actually answer; that trades completeness for speed.

---

### Quick pre-submission checklist
- [ ] First abstract sentence = the contribution, not setup/floor/caveat
- [ ] Strongest survived finding stated before any fragility
- [ ] Limitations ≤ ~6 main-text items; no multi-paragraph limitation essays
- [ ] No hedges inside the abstract body (footnote pointer only)
- [ ] "Establish" vs "preliminary" results partitioned where both exist
- [ ] Every first-person box matches the final corrected state
- [ ] Every claim has reproducible code/data, or an honest note where it genuinely can't
- [ ] Every symbol/abbreviation defined at first use
- [ ] Every named technique/system oriented ("what it does") before detail ("how it works")
- [ ] Null results framed as findings with their own subsection, not buried
- [ ] Pre-registration committed before experiment run (PREREG_TEMPLATE.json)
