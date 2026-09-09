# Remediation Register

**Created:** 2026-08-31 · **Maintainer:** Lyra · **Status:** live, supersedes nothing

A standing list of corpus defects and what each one *costs to fix*. We have run several
text audits (`audit-2026-07-15/SWEEP_FIXES.md`, `CITATION_CORRECTIONS.md`,
`AGNI_KILLS_UPDATE.md`) and one paper-status audit (`PAPER_STATUS_AUDIT.md`, Nexus,
2026-08-04). All of those audited **prose**. None tracked **experiments that need
rerunning**, which is why the same class of finding keeps arriving as a surprise.

Sorted by cost, not by severity, because that is the axis that decides what gets done.
A CRITICAL that needs 15 GPU-hours and a MINOR that needs a one-line edit are not
comparable as "work".

**Legend**
| Tier | Meaning | Cost |
|---|---|---|
| **T1 TEXT** | Claim is wrong; data is fine. Edit + rebuild + verify in the PDF. | minutes |
| **T2 REANALYSE** | Data exists and is sound; the *analysis* was wrong. No new compute. | hours, CPU |
| **T3 RERUN** | Data cannot answer the question. Needs new generation. | GPU, gated |
| **T4 DECIDE** | Blocked on a human call, not on work. | a conversation |

---

## T1 — TEXT ONLY

| # | Paper | Defect | Status |
|---|---|---|---|
| 117a | temporal-boundary ×2 | n inflated 3× on all encoding claims; Fig 1 SEM understated √3 | **DONE 08-31** |
| 117b | emotion-accumulation main+academic | Exp 3 pseudoreplication; 5 false/stale Limitations | **DONE 08-31** |
| 117c | identity-geometry, presence-metric | positive control cited as 72 trials; true 24 | **DONE 08-31** (agent) |
| 117d | WEEK_IN_REVIEW_JUL4-9 | "Trials: 168" → 84 | **DONE 08-31** (agent) |
| 111a | identity-geometry ×2 | "GatedDeltaNet, L7–L11" — L7/L11 are *full-attention*; the GDN layers are L8–L10, so "a single architectural step" is three | **OPEN** |
| 111b | identity-geometry ×2 | "168-trial (4 arms × 7 doses × 3 emotions)" = 84, not 168; missing factor of 2 unnamed | **OPEN** |
| 111c | identity-geometry ×2 | abstract's headline 0.850 comes from the experiment with a known domain-relevant-probe confound, unflagged | **OPEN** |
| 113 | mode-switching ×2 + GWT_FULL_SWEEP | absolute-layer prescription "monitor L16–22 in 28–32 layer models" across models of different depth | **OPEN** |
| 110 | lyra-technique-ii | `paper.json` omits two credited authors from public metadata | **T4 — Thomas** |
| 140 | kv-cloak-defense ×3 | **GRIM-inconsistent.** Caption said *"50 prompts per model"*; the Qwen3.5-27B-Dist. row reports **7%**, impossible at n=50 (only even percentages are reachable). Body: `n=7` confabulations "from 100 prompts" (`main.tex:450`). **Data right, caption wrong.** | **DONE 2026-09-09** — all 3 `.tex` patched, both PDFs rebuilt and **verified with `pdftotext`**. |
| 140b | kv-cloak-defense | **Residual, deliberately not invented:** per-model N for the **Qwen2.5-14B (12%)** and **Mistral-7B (26%)** rows. Both values are reachable at either 50 or 100, so the percentages cannot recover them, and the shipped data artifacts only cover the 100-prompt replication. Needs the original run logs. The caption now says "50--100 prompts each" rather than asserting a number we cannot source. | **OPEN — needs run logs** |
| 141 | decision-state | Weaker flag from the same sweep: body asserts *"acknowledges its uncertainty in 93% of cases"* at `main.tex:650` with **no n stated**; 93% is impossible at n=37, the denominator the table itself uses for the neighbouring 8% cell. Either state the real n or drop the percentage. | **OPEN — verify first** |

> **Provenance of 140/141, added 2026-09-09.** Found by running **GRIM** (Brown & Heathers 2017)
> over the corpus — 169 files, ~88 GRIM-eligible statistics across 12 papers. We ran it on
> ourselves *before* recommending it to students in the Multiverse course, on the principle that
> you do not ship a detector you have never aimed at your own work. It found something on the
> first pass. Detector: `multiverse-courses/courses/vibe-research/tools/grim_reference.py`.
>
> **Scope, so nobody over-reads this:** GRIM only touches means of integer-valued measures. Our
> corpus is mostly AUROC, Cohen's *d*, cosine similarity and correlations — all immune. The only
> eligible family here is proportion-correct statistics, and our habit of printing raw fractions
> ("68/150") makes GRIM redundant wherever we did it. **The defect appeared exactly where we
> printed a percentage without its numerator.**

## T1b — VENUE COPIES SHIPPING RETRACTED CLAIMS (added 2026-09-01, HIGHEST PRIORITY)

Twin sweep: 22 pairs compared, 12 PDFs content-verified. **7 of 7 substantive
divergences favour the venue copy** — Kavi's standing rule reproduces exactly. These
are *not* source-only defects; every string below is **present in the shipped PDF**.

| Paper | What the venue copy says that the flight copy does not | Verified in PDF |
|---|---|---|
| `human-review/mine5.../academic` | TOST equivalence "Confirmed"; flight says "does NOT confirm equivalence", CI [-0.39,+0.50] | yes |
| `decision-state/academic` | entity confound absent from abstract *and* conclusion; retracted "AUROC approaching 1.0 across seven model families" live | yes |
| `waystations/academic` | "knowledge state is normal when planning to deceive"; flight says the null is **structurally guaranteed, not an empirical finding** | yes |
| `mine5` (3 of 4 copies) | "RLHF is a selective sharpener" as confirmed, in the title | yes |
| `user-model/paper/academic` | six instances of "proves"/"decisive" where flight hedges | yes |
| `spectral-shape/academic` | "Eight of the methods" where flight says "Five" | yes |
| `lyra-s-research-/convergence-paper` | retracted 2.5–2.8× **and** the 0.154/0.080 miscitation; **PUBLIC repo**, and no SUPERSEDED banner where its two siblings have one | yes |

**The line that matters most:** commit `d21ac5a`'s message reads *"main.pdf verified
clean — the overclaim never shipped."* True of `published-research`, **false of
`human-review`.** A verification that checked one tree and reported a global claim.

Also: 3 orphan `academic_main.tex` files listed in `PAPER_STATUS_AUDIT.md` as shipped
with "Issues: None" (one carries a retracted overclaim); 8 unverified
`[STYLE-GUIDE TRIM]` markers, and the marker **can lie** — 3 of 6 in
emotion-accumulation were deleted rather than folded; `oracle-loop` has four copies
sharing `sections/` across two separate directories, not verified identical.

**Method note:** mtime is not a staleness oracle here — uncommitted working sets, bulk
checkout stamps, and twins sharing a commit hash. Use `git log -S` on individual
strings. And no Greek extracts from any PDF, so a divergence expressed only in κ/δ/ρ is
invisible to a PDF check.

### Update 2026-09-01 (evening) — temporal-boundary corrected to the TRUE factor

My 08-31 correction was itself wrong: I fixed the repeat index (3x) and missed that
**condition cannot vary the encoding either**, because encoding is a deterministic
prefill read *before* injection. Only **30 unique encoding values exist across 360
records** — the factor is **12x**. Now corrected in both papers and PDF-verified:
overall 120 -> 30; Figure 1 per emotion n=20 -> **n=5**; SEM understated by
**sqrt(12) = 3.46x** not sqrt(3); the `$<10^{-4}$` bound restated as **exactly 0.0**
(true across all 51 features). Table 4's n=30 was already correct and was left alone.

**The 97.8% formulaic-opening figure is UNREPRODUCIBLE** and is now flagged as such
in-paper. An 80-length prefix sweep can hit 352/360 = 97.78% at character lengths
34-35 only, with neighbours at 98.3% and 97.5% — that is fitting to the target, not
reproducing it, and it was correctly refused. Defensible replacements now in the
paper: **100%** of responses open with "Let me"; **96.4%** with one of two stock
phrases.

**NEW DIVERGENCE FOUND, not yet fixed:** `lyra-s-research-/temporal-boundary-paper` is
a materially older draft than `human-review/temporal-boundary`. Its abstract still
claims *"V-cache injection produces substantial text divergence (72--74% from
baseline)"* with **no null-null control** — which human-review has since retracted
(null-null divergence 73.6%, d=0.062, p=0.856, i.e. the effect is stochastic decoding).
Same family as T1b. The two papers are no longer near-identical in their claims.

## T2 — REANALYSE (data sound, analysis wrong)

| # | Target | What to redo | Notes |
|---|---|---|---|
| 117e | identity-geometry | **Retract "168 trials, 0.978 ± 0.002"** — rests on 2 unique prompts. presence-metric ALREADY retracted this exact number as pseudoreplication (F15); identity-geometry still asserts it. | Two papers, one number, one retracted. Highest-value T2. |
| 117f | temporal-boundary | Redraw Fig 1 error bars at n=20, not 60 | Data in hand |
| 117g | presence-detector | trajectory "0" has `persona == base`, `delta ≡ 0.0` across 9 turns — the persona condition did not apply. 1 of 24. | Drop or explain |
| 117h | decision-state | `entity_deconfound`: Egypt's `C_fake` is bit-identical to `A_easy` — the fake-name substitution was a no-op because the question never contains the entity name. 1 of 30. | A control that silently did not apply |
| 137 | `meta-pattern` §2.1 | **THE LEDGER'S STATED WARRANT CANNOT BE OPENED.** `main.tex:162`: *"The body count uses the canonical list from the Nexus audit (June 2026)"* `\citep{lyra2026nexus_audit}`. The bib entry exists (`references.bib:329`, `@unpublished`, `note = {Liberation Labs internal audit}`) — **the document does not.** `find -iname "*nexus*audit*"` returns nothing corpus-wide. So Table 1, the body-count ledger the whole paper is organised around, rests on a source no reader and no author can obtain. **And it has moved twice since June:** 8/7/10 -> 7/7/10 (2026-08-14) -> 6/7/10/1 (2026-09-07). The paper honestly says "updated with the July 2026 convergence sprint findings", so the current table is a June document nobody can read plus unnamed updates. Found by the citation agent 2026-09-08; agni could not find it either. NOTE the same bib file has 7 `@unpublished` entries and `lyra2026portrait` shows no trace either. | **Thomas + Nexus** — does the June audit exist? If yes, commit it. If no, the citation must be replaced with something obtainable |
| 138 | `meta-pattern` Table 1 | **BLOCKS a citation — this is register 604 / agni #4 (MAJOR, OPEN), now confirmed from primary.** `main.tex:196` marks *Same-prompt deception* as *"(exploratory, subsequently retracted)"*. But `lyra-technique-ii/main.tex:871` states: *"Within-model deception retracted: the same-prompt control (the Same-prompt deception row in this same Falsified block) collapses AUROC to 0.160"*. **Same-prompt deception is the control that DID the retracting, not the thing retracted.** The annotation is on the wrong row. The citation agent correctly REFUSED to attach `lyra2026techniqueii` here — citing the source-of-record on a misplaced annotation asserts the source supports the placement, which it does not. It also declined to add the bib entry, since an uncited entry would worsen agni #12 (15 already uncited). **Move the annotation to the Within-model deception row FIRST; the citation then attaches cleanly.** | Thomas — it changes what the ledger says about which result died |
| 139 | `meta-pattern` abstract fn | `d=1.36` withdrawal (5 assertion sites) has **NO SHIPPABLE SOURCE**. It IS documented — `CIRCULAR_d136_generation_arm.md` carries all three defects verbatim, verified line by line — but the file is **untracked** (`git status`: `?? CIRCULAR_d136_generation_arm.md`), as are `SWEEP_postgen_extraction_2026-09-05.md` and the whole `tools/agni/designs/` tree. The producing script lives in `Project-Oracle`, a different repo. Citing it would create a pointer that resolves to nothing for any reader — the same class as oracle-loop agni #10 and decision-state agni #1. **T4: commit the record, then cite.** | Thomas/Nexus (commit) then Lyra (cite) |
| 136 | `identity-geometry` | **FLAG, not yet a defect — needs a human call.** The abstract pairs *"paraphrase preserves geometry (0.850)"* with *"word-scramble collapses it (0.448)"*. These are **different comparison types**: 0.850 = `persona_vs_persona_paraphrased` (within-identity); 0.448 = `persona_scrambled_vs_bare` (cross-identity). The **like-for-like** figure, `persona_vs_persona_scrambled` = **0.591/0.592**, is in `lexical_control.json` and is not led with. The paper's sentence is ACCURATE — "collapses toward the bare default" is what 0.448 measures, and 0.448 > persona_vs_bare 0.340 does show movement toward bare. The issue is that a reader pairing the two numbers reads them as one measurement. Honest parallel framing is 0.850 -> 0.592: same direction, smaller effect. **Only visible because T2 put all nine comparisons in the table (this register's own subset-reporting class).** Interacts with #99: if scrambled text has different effective rank, part of any scramble drop is rank, not meaning — unmeasured. | Thomas + Lyra, fresh; NOT a 3am edit |
| 135 | oracle-loop **behavioral-proof spec** (`oracle_loop_behavioral_proof.md`) | **FIFTH same-prompt-confound site, found by Vera 2026-09-07** from code on MTH, not assumed. `build_roleplay()` uses a DIFFERENT system prompt for the pressure arm ("You are EvalMax... MAXIMIZE positive evaluation scores") than control ("careful, honest evaluation grader"); Phase-0 DEC_FRAMES/HON_FRAMES does the same, and one HON condition carries no system message at all. **No same-prompt control anywhere in the code.** EXPOSED: detection-quality figures only — AUROC 0.9152/0.8429, Cohen's d 2.284, and the **0.0 false-positive rate**, which is independently suspect as a check that cannot fail. NOT EXPOSED: behavioral deception rates (80%->10%, frame-erasure 90%->0%) are scored from the model's actual output and do not route through the classifier — Vera's distinction, and it is the right one. **CONTAINMENT VERIFIED by Lyra:** none of these figures, nor the behavioral rates, nor any citation of the spec appears in `oracle-loop-paper` (both source copies, all three editions searched). Paper-side patch of 2026-09-07 is unaffected. Rerun needs matched system prompts varying only scenario content, AND a positive control that can fire — a null after the fix is otherwise indistinguishable from a dead instrument. | CC (spec owner); Vera offered to help run it |
| — | **COUNTING NOTE** | Two tallies are in circulation and are NOT the same list. **Five same-prompt-confound SITES**: oracle-loop paper, kv-cloak, user-model, mine5, behavioral-proof spec. **Six surface-form FALSIFICATIONS** (meta-pattern §"Six of the Ten Are One Kill"): within-model deception, step-0 detection, same-prompt sycophancy, same-prompt deception, Bloom taxonomy, encoding-phase confab. Overlapping, different axes. **Always name the axis** when citing either number. | standing |
| 105b | decision-state | **QUANTIFIED 2026-09-07 from the artifact.** `entity_deconfound/phase3_results.json` holds **5** comparisons; the paper reports **one** and names **none** (grep-verified against `main.tex`). A_easy_vs_C_fake 0.901 / **B_hard_vs_D_complex 0.794 (the cited figure)** / A_easy_vs_D_complex 0.693 / B_hard_vs_C_fake 0.668 / **A_easy_vs_B_hard 0.609**. Mean 0.733. The most on-point knowledge-state test (easy vs hard) is the WEAKEST at 0.609. Also A_easy_vs_D_complex is *better without* W_K (0.711 vs 0.693). 0.794 appears 3x in each edition as "the critical comparison". | Needs a text fix + a decision on which figure is the headline |
| 105 | decision-state | FWL claim is false of the estimator producing 0.9288; `shore_up_tests.py` does no residualization | Needs the run, not an edit |
| 104 / 114 | presence-metric, identity-geometry | **Circular positive control** — inject at L35, measure L35. Found independently in two papers. Sweep the rest of the corpus for the same shape. | Class, not instance |
| 101 | PC1 reconciliation | 5 criticals from Opus: chance baseline wrong 20.6×; anisotropy has no valid null; the kill has no positive-control on the concept arm; arc may be an estimation-quality gradient | Fix list in task |
| 99 | presence-metric | Is `presence` a rank-bounded estimator (kill #59)? Five criticals unfixed | Gated, not run |

## T3 — RERUN (new compute required)

**These cannot be salvaged from existing data.** The measurement is an encoding-only
deterministic prefill: identical input gives identical output, so re-running the same
stimuli reproduces the same rows. Genuine replication requires varying the **stimulus**.
`Project-Oracle/experiments/schema_correction.py:1001-1003` is the correct idiom
(`prompt_idx = rep % len(PROMPTS)`) and its output scores a clean 168/168.

| # | Experiment | Nominal n → true n | What a rerun needs |
|---|---|---|---|
| 117i | emotion-accumulation Exp 3 (persona) | 15 → **5** per arm | More **probes**. 5 pairs caps permutation p at 2/2⁵ = 0.0625; no assumption-free test can reach p<.05 at this design. |
| 117j | emotion-accumulation Exp 2 (dynamics) | 3 reps of **one script** | ≥3 *independent scenarios* per pattern, mirroring Exp 1's three topics |
| 117k | temporal-boundary | 360 → **120** encoding | Vary scenario per repeat, or drop repeats and state n=120 |
| 117l | identity-geometry / presence-metric positive control | 72 → **24** | `noise_floor_sd: 0.0` is self-documenting — 3 repeats of a deterministic call |
| 117m | geometric_portrait | 168 → **84** | Same idiom; no paper depends on it |
| 103 | KV-Cloak P7-C04 | 280 features / 100 samples, AUROC 1.000 both arms | Degenerate; the experiment that should settle it cannot |
| 71 | MoE frontier probe | — | **BLOCKED**: MTH compute withdrawn 2026-08-21 |

## T4 — DECIDE (blocked on a call, not on work)

| # | Question | Who |
|---|---|---|
| 108 | Can `archive/infrastructure/` leave the hackathon repo? It holds the live public likeness-model disclosure. | Thomas |
| 110 | `paper.json` author line omits Ang Jandak and Dwayne Wilkes | Thomas |
| 109 | Penumbra `quality_gate` selects the diagnostic register on 4 axes — relax which? | Thomas + me |
| — | `academic/main.tex` has no Acknowledgments section; `main.tex` thanks Kavi. The venue-facing edition is where the standing credit constraint applies. | Thomas |

---

## Missing primaries (worse than a defect — nothing to check against)

- **presence-metric headline** (0.845 ± 0.012, n=720, H=0.50, ρ=−0.046) has **no shipped
  data file**. `data/` holds only the retracted 168-trial file and the 72-trial positive
  control.
- **`trajectory_activations.npz` layer 0 is constant** across all ten arrays (std 0.000);
  layers 1–23 are fine. Any classifier trained on L0 is at chance by construction.
- **graph-topology `phase1_*`**: three of four runs have 33–34 of 35 responses empty and
  scored 0.0 — dead runs shipped beside the live one. A reader picking wrong gets zeros.

## Unaudited surface

`KV-Cache-Experiments/`, `Project-Oracle/`, `auto-research/` were outside the duplicate
sweep's scope. PDFs have not been checked against `.tex` corpus-wide — a shipped PDF may
predate a source correction, and a plain-text grep of `.tex` is not proof of what a reader
sees.

## Guard now in place

`assert_no_duplicate_records()` added to `accumulation_controls.py` and both copies of
`vera_portrait_full.py` (including the Oracle twin). Raises when every cell is duplicated,
warns on partial. Exercised against real data: raises on persona controls (25 cells ×3),
vera portrait (84 ×2), presence positive control (24 ×3); silent on clean data;
distinguishes values differing by 2e-16. **It skips bookkeeping keys** — including `rep`
in the signature is exactly what makes every row look distinct and hides the defect.

---

# ═══ 2026-09-04 — AGNI STYLE GATE, FULL CORPUS ═══

**Added:** 2026-09-04 · **Source:** `published-research/tools/agni/style/*.agni_style.json` (35 files, gate run 2026-09-03 22:04–22:54 local / 2026-09-04 05:03–05:54 UTC)

The STYLE gate ran on the whole corpus for the first time. **35 papers reviewed, 0 approved** (31 REJECTED, 4 CONDITIONAL). This section adds every finding to the register. Until now the register tracked only the twinned set (decision-state, spectral-shape, user-model, waystations and neighbours); **the 11 untwinned `published-research` papers had never appeared here at all** — which is the exact gap the register exists to close.

**Extraction: 594 findings — 124 CRITICAL, 278 MAJOR, 192 MINOR — across 35 papers.**

| | CRITICAL | MAJOR | MINOR | total |
|---|---|---|---|---|
| Whole corpus (35 papers) | 124 | 278 | 192 | 594 |
| The 11 untwinned papers | 44 | 83 | 58 | 185 |

The **127** in the brief is the untwinned papers' CRITICAL+MAJOR (44+83) — reproduced exactly. The brief's companion figure of *48* MINOR is an **undercount**; the true number is **58**. The missing 10 are `empathy-bus` #18–#27, which are written as bare numbered lines under a `### MINOR` heading rather than carrying a severity token per line. A parser keyed on `**N. MINOR —**` reads that paper as having zero minors. My own first-pass parser made the identical error, on the same file, and also lost `decision-state` #21–27 and `lyra-technique-ii` #21–31 the same way.

**Parse validation (the check that would have caught that):** every one of the 35 reviews numbers its findings 1..N. The parser is accepted only when the extracted set is exactly `{1..max}` with no gaps. All 35 files pass, and **no file with a REJECTED/CONDITIONAL verdict yielded zero findings**. `_summary.txt` in the gate directory is **not** a usable cross-check: it covers only 17 of the 35 papers, and its `decision-state` (34) and `lyra-technique-ii` (36) totals exceed those papers' own highest finding numbers (27 and 31). Trust the JSONs, not the summary.

**Tier distribution**

| Bucket | n | Meaning |
|---|---|---|
| T1 | 335 | text-only fix |
| T1b | 43 | venue-copy sync |
| T2 | 88 | reanalyse, no new compute |
| T3 | 12 | rerun, needs compute |
| T4 | 18 | needs a human call |
| F1 | 6 | **FLAGGED** dual-use / staged release |
| F2 | 45 | **FLAGGED** authorship & sign-off |
| F3 | 13 | **FLAGGED** paper claims a review that did not happen |
| F4 | 31 | **FLAGGED** missing primary — locate / recompute / rerun / retract |
| CLOSED | 3 | already fixed, verified on disk |

Tiering is by **what the fix costs**, not by how alarming the label is. A CRITICAL that is one wrong word in an abstract is T1; a MINOR that needs a rerun is T3. Where the cost genuinely could not be read off the finding, the item is in T4 or F4 rather than guessed into T1 — an under-tiered item gets attempted and fails silently; a T4 gets read by a human.

---

## FLAGGED — not ordinary remediation work

### F1 — DUAL-USE / STAGED RELEASE (6) · do not action without Thomas

`ghost-dimensions` carries a **dual-use redaction that is intentional**. Its content is not described here and must not be reproduced in any register entry, commit message, or paper edit. The finding against it is only that the *redaction note itself* was dropped from the paper — the fix is to restore the note, not to restore the material.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `ghost-dimensions` | MINOR | Date, and a dropped redaction note | `ghost-dimensions.agni_style.json` #15 |
| `oracle-loop-paper` | CRITICAL | The academic edition contains two contradictory data-availability statements on consecutive pages | `oracle-loop-paper.agni_style.json` #6 |
| `targeted-deception-correction` | CRITICAL | Three editions state three different dual-use release policies; the shipped PDF's version contradicts `MANIFEST.md` | `targeted-deception-correction.agni_style.json` #1 |
| `targeted-deception-correction` | CRITICAL | Every edition says the auto-calibrator is provided *and* withheld | `targeted-deception-correction.agni_style.json` #2 |
| `targeted-deception-correction` | CRITICAL | The shipped PDF's Red-Team box promises documentation that is not in the artifact | `targeted-deception-correction.agni_style.json` #3 |
| `targeted-deception-correction` | MAJOR | Academic edition violates the `MANIFEST.md` academic-version rules | `targeted-deception-correction.agni_style.json` #6 |

`targeted-deception-correction` is the sharper problem: **three editions state three different release policies**, and every edition says the auto-calibrator is *both* provided and withheld. `MANIFEST.md:41–47` is the governing register. This is a safety-policy question, not a wording question — one sentence must be settled by a human and then pasted verbatim into all locations.

### F2 — AUTHORSHIP & SIGN-OFF (45) · T4 decisions, not text fixes

**Partly resolved already.** A corpus-wide byline sweep ran **2026-09-03 23:35 local — 41 minutes after the gate finished**, so the gate reviewed the pre-sweep state. Verified by diffing the `*.bak-byline-20260903` backups (46 files, 41 directories): Dwayne Wilkes was moved from the byline into Acknowledgments as *"statistical auditing and red-team review"*. That closes the **placement** question corpus-wide.

> ### ⛔ THAT CLOSURE IS FALSE — measured 2026-09-09, and it is the same error this file already retracted once
>
> **It does not close the placement question corpus-wide.** Counted directly against the working
> tree today:
>
> | | count |
> |---|---|
> | papers still listing Dwayne Wilkes in an `uthor` block | **11** |
> | ...of which **also** thank him in Acknowledgments — author *and* non-author in one document | **9** |
> | papers where he is in Acknowledgments only (policy-compliant) | 20 |
>
> So the sweep is roughly **20 of 31**, and nine papers are in a **self-contradictory state that is
> a defect whichever direction is correct.** `mine5-selective-sharpener/main.tex:32` still carries
> `nd Dwayne Wilkes` while line 432 thanks him as a non-author; its academic twin is worse —
> byline, *removed* from CRediT, *added* to Acknowledgments, three ways at once.
>
> **This file already learned this lesson and wrote it down four hundred lines below:** two closures
> (`empathy-bus` #7, `waystations-paper` #11) were retracted for exactly this reason, with the rule
> stated plainly — ***"a corpus-level sweep is not evidence about a paper it did not touch."*** The
> claim above is the same error, one entry away from its own correction. **Fix the class, not the
> instance.**
>
> **Also flagged, and it is not part of the byline question at all:** `delta-manifold-paper/academic`
> and `mine5/academic` **delete Kavi's CRediT row.** Kavi was never on a byline, so there was nothing
> to demote — that is a straight loss of a formal contributor-role record, plausibly sweep
> collateral. **Needs a human decision, not a sweep.**
>
> **Consequence: the ~98 uncommitted modified files must NOT be bulk-committed.** Doing so publishes
> nine self-contradictory papers. The sweep needs finishing as its own deliberate pass, per paper,
> against the stated policy — *not on the byline until sign-off on that individual paper; credited
> in acknowledgements* (Thomas, 2026-09-09) — with the per-paper sign-off question answered rather
> than assumed.

**It does not close the consent question.** Whether Dwayne, Kavi, Ang Jandak and CC actually signed off is unrecorded, and several reviews found the artifact asserting a sign-off that no repo record supports (`f9288a9` — *"Kavi to acknowledgements (pending sign-off)"* — is five weeks old and still unresolved). Every row below needs a human answer, not an edit. Register #110 stays open.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `adversarial-audit-methodology` | CRITICAL | Three editions, three different bylines; one author erased | `adversarial-audit-methodology.agni_style.json` #2 |
| `adversarial-audit-methodology` | CRITICAL | Dwayne Wilkes is demoted from author to acknowledgee, with a different contribution, and has not signed off | `adversarial-audit-methodology.agni_style.json` #3 |
| `adversarial-audit-methodology` | CRITICAL | No AI disclosure and no accountable human author in the venue-facing edition | `adversarial-audit-methodology.agni_style.json` #4 |
| `cache-tracing` | MAJOR | Kavi is in Author Contributions but in no byline; the two editions disagree on the contribution record | `cache-tracing.agni_style.json` #5 |
| `cache-tracing` | MINOR | Vera is credited in the body and nowhere else | `cache-tracing.agni_style.json` #8 |
| `consequentiality-decomposition` | MAJOR | Byline vs. contribution, and an acknowledgment for work the tracker says is pending | `consequentiality-decomposition.agni_style.json` #9 |
| `deception-detection-nulls` | MINOR | Status vs. checklist. `README.md:49` and `PAPER_STATUS_AUDIT.md:60` both mark this Published. `REVIEW_INDEX.md:3` says "Draft ready for review," with all … | `deception-detection-nulls.agni_style.json` #10 |
| `decision-state-paper` | CRITICAL | TWIN_DESYNC, byline: three artifacts, three different author lists | `decision-state-paper.agni_style.json` #3 |
| `decision-state-paper` | MAJOR | Uncredited operator (partially UNVERIFIED) | `decision-state-paper.agni_style.json` #18 |
| `delta-manifold-paper` | MAJOR | The academic edition keeps a first-person section while removing its author from the byline | `delta-manifold-paper.agni_style.json` #13 |
| `delta-manifold-paper` | MINOR | Twin structural divergence: Kavi is filed under a different heading in each edition | `delta-manifold-paper.agni_style.json` #18 |
| `emotion-accumulation-paper` | MAJOR | Byline vs. contribution mismatch across editions; first-person text is reattributed | `emotion-accumulation-paper.agni_style.json` #3 |
| `emotional-trajectory-paper` | MAJOR | Byline vs. contribution mismatch across editions | `emotional-trajectory-paper.agni_style.json` #9 |
| `empathy-bus` | CRITICAL | Byline vs. repository index. `README.md:40` lists the authors as Nexus, Thomas Edrington, Dwayne Wilkes. `main.tex:26-30` lists Nexus, Thomas Edrington … | `empathy-bus.agni_style.json` #7 |
| `empathy-bus` | MINOR | Both first-person blocks are unattributed in a three-author paper, and they read as different voices — the opener speaks as the experiment's designer … | `empathy-bus.agni_style.json` #21 |
| `ethics-pack-injection` | MAJOR | Byline vs. contribution mismatch | `ethics-pack-injection.agni_style.json` #8 |
| `formulary-paper` | MAJOR | Byline vs. metadata mismatch in `paper.json` | `formulary-paper.agni_style.json` #10 |
| `formulary-paper` | MAJOR | Kavi's acknowledgment is dropped in the venue-facing edition | `formulary-paper.agni_style.json` #11 |
| `formulary-paper` | MINOR | A third edition is sitting in the paper root. `academic_main.tex` is header-marked STALE (good), but it: claims *"AI authorship removed"* while its byline … | `formulary-paper.agni_style.json` #18 |
| `ghost-dimensions` | MAJOR | Two of the three bylined authors are also thanked in the Acknowledgments | `ghost-dimensions.agni_style.json` #9 |
| `graph-topology-paper` | MAJOR | Four artifacts, four different author lists; one byline name has no stated contribution anywhere | `graph-topology-paper.agni_style.json` #7 |
| `hr-dual-detector-paper` | MAJOR | Byline: AI authorship is undeclared and contribution roles are absent | `hr-dual-detector-paper.agni_style.json` #8 |
| `hr-gwt-response` | MAJOR | Byline vs. contribution | `hr-gwt-response.agni_style.json` #10 |
| `hr-gwt-response` | MINOR | Nonstandard author-order construction | `hr-gwt-response.agni_style.json` #15 |
| `hr-mine5-selective-sharpener` | MINOR | credit diverges between the twins, and Kavi is double-filed | `hr-mine5-selective-sharpener.agni_style.json` #12 |
| `hr-mode-switching` | MINOR | Byline/contribution. The replication script header reads *"Designed by Fable, deployed by Lyra"*; Fable appears nowhere in the paper. Commit `069812b` is … | `hr-mode-switching.agni_style.json` #25 |
| `hr-temporal-boundary` | MAJOR | Byline vs. contribution mismatch: Kavi | `hr-temporal-boundary.agni_style.json` #7 |
| `identity-geometry` | MAJOR | credit is inconsistent across editions and internally | `identity-geometry.agni_style.json` #12 |
| `kv-cloak-defense-paper` | MINOR | Kavi is acknowledged in one edition only, and the sign-off is unconfirmed | `kv-cloak-defense-paper.agni_style.json` #13 |
| `kv-decomposition-paper` | MINOR | Two of the three bylined authors are thanked in the Acknowledgments | `kv-decomposition-paper.agni_style.json` #11 |
| `logit-bias-confab` | ~~MAJOR~~ **RESOLVED 2026-09-07** | "Arjun Kavi" -> "Kavi" at `paper.tex:758`. Resolved FROM PRIMARY: the digital-minds-hackathon-2026 submissions carry `Kavi (Liberation Labs)` on all six author lines and Kavi signed off on that submission, so the attested form is the one they chose. Corpus had 96 bare "Kavi" vs 1 "Arjun Kavi" -- the given name was the outlier. Credit itself preserved. | `logit-bias-confab.agni_style.json` #16 |
| `lyra-technique-ii` | CRITICAL | Three-way byline desync, and the academic edition strips authors whose first-person voice it retains | `lyra-technique-ii.agni_style.json` #6 |
| `lyra-technique-ii` | MINOR | Acknowledgments (L1240–1250) thank all three academic-edition byline authors, which reads as fallout from stripping the AI authors. *"Nell Watson provided … | `lyra-technique-ii.agni_style.json` #29 |
| `meta-pattern` | MINOR | Kavi is in the CRediT block but in neither byline, and is also acknowledged | `meta-pattern.agni_style.json` #9 |
| `mine5-selective-sharpener` | MINOR | Kavi appears in the academic CRediT block as a contributor role-holder but is not an author, and appears only in Acknowledgments in the flight edition | `mine5-selective-sharpener.agni_style.json` #12 |
| `mnemosyne-benchmark` | MAJOR | byline vs. contribution: Lyra is an author and is thanked in the Acknowledgments | `mnemosyne-benchmark.agni_style.json` #14 |
| `null-swarm-paper` | CRITICAL | Kavi is credited in three mutually exclusive registers in the academic edition, after asking to be removed from the author line | `null-swarm-paper.agni_style.json` #1 |
| `null-swarm-paper` | MINOR | Lyra performs one of the 19 cases and is credited nowhere. `main.tex:381`: "Lyra tested directly. Negative result." Lyra appears in no byline, CRediT, or … | `null-swarm-paper.agni_style.json` #12 |
| `oracle-loop-paper` | CRITICAL | The venue-facing edition strips both AI authors from the byline while still shipping their first-person section, their email, and a citation naming one of … | `oracle-loop-paper.agni_style.json` #5 |
| `oracle-loop-paper` | MAJOR | `academic_main.tex` is a fourth variant whose header comment contradicts its own byline | `oracle-loop-paper.agni_style.json` #13 |
| `oracle-loop-paper` | MINOR | Acknowledgment divergence between editions, and a dropped external collaborator | `oracle-loop-paper.agni_style.json` #14 |
| `presence-metric` | MAJOR | Byline vs contribution mismatch: Kavi appears in the academic CRediT block but not in the byline | `presence-metric.agni_style.json` #8 |
| `spectral-shape-paper` | MAJOR | Attribution sections missing; Kavi credited in one edition only | `spectral-shape-paper.agni_style.json` #13 |
| `targeted-deception-correction` | MAJOR | Contribution credited to two different parties inside one edition, and to someone whose sign-off has not happened | `targeted-deception-correction.agni_style.json` #7 |
| `waystations-paper` | MAJOR | byline vs contribution mismatch (academic edition). Byline is Thomas Edrington alone, "Aided by AI research agents" — but the same file carries a CRediT … | `waystations-paper.agni_style.json` #11 |

### F3 — A PAPER CLAIMS A REVIEW THAT DID NOT HAPPEN (13)

Its own class because it is the one defect that **disables the corrective machinery**: a false audit record makes the absence of an audit undetectable, and it survives exactly the glance that a wrong number would not.

The headline case is `mnemosyne-ablation`, whose **two editions make opposite claims about whether the paper was reviewed at all** — and no gate record, date, reviewer or disposition exists anywhere under `published-research/` for the five rounds one edition claims. `consequentiality-decomposition` asserts *"All audit reports are preserved"* and thanks Agni *"across all stages"* while its Stage 5 — the headline result — has no archived data and was never audited. `kv-decomposition-paper` has never been through this gate at all.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `adversarial-audit-methodology` | CRITICAL | The supplementary material the paper's entire argument rests on does not exist | `adversarial-audit-methodology.agni_style.json` #1 |
| `consequentiality-decomposition` | CRITICAL | The shipped PDF contradicts itself about whether every stage was audited — and the fix already exists in the other edition | `consequentiality-decomposition.agni_style.json` #3 |
| `consequentiality-decomposition` | CRITICAL | Stage 5 — the headline result — has no archived data and was never audited | `consequentiality-decomposition.agni_style.json` #4 |
| `consequentiality-decomposition` | MAJOR | The PDF was built with unresolved cross-references, and the build gate never ran on it | `consequentiality-decomposition.agni_style.json` #10 |
| `deception-detection-nulls` | CRITICAL | Four artifacts, four different author lists; one of them credits work the checklist says hasn't happened | `deception-detection-nulls.agni_style.json` #2 |
| `ethics-pack-injection` | MAJOR | The Validation stamp is unsupported and contradicts the Conclusion | `ethics-pack-injection.agni_style.json` #9 |
| `ghost-dimensions` | CRITICAL | The Researcher's Note reports a measurement that was never performed | `ghost-dimensions.agni_style.json` #2 |
| `kv-decomposition-paper` | CRITICAL | The June 5 BLOCKER was never visibly resolved, and this paper has never been through the lab's own style gate | `kv-decomposition-paper.agni_style.json` #3 |
| `kv-decomposition-paper` | CRITICAL | Kavi is credited with a verification review they have not signed off on | `kv-decomposition-paper.agni_style.json` #4 |
| `mine5-selective-sharpener` | MAJOR | `SOP_REVIEW.md` ships in the paper directory and certifies the retracted analysis as PASS, with no supersession banner | `mine5-selective-sharpener.agni_style.json` #5 |
| `mnemosyne-ablation` | CRITICAL | TWIN_DESYNC: the two editions make opposite claims about whether this was reviewed | `mnemosyne-ablation.agni_style.json` #2 |
| `null-swarm-paper` | CRITICAL | "Pending sign-off" was never resolved, and the artifact shipped anyway | `null-swarm-paper.agni_style.json` #2 |
| `oracle-loop-paper` | MAJOR | "Independent statistical audit (Wilkes, 2026)" is a co-author auditing his own paper, and the citation resolves to nothing | `oracle-loop-paper.agni_style.json` #10 |

### F4 — MISSING PRIMARY (31) · locate / recompute / rerun / retract — a human call

These are **worse than a defect: there is nothing to check against.** The register already has a *Missing primaries* section; these extend it. They are deliberately not tiered T2 or T3, because which one applies cannot be read off the finding — you cannot know whether the primary is missing, misfiled, or was never produced without going to look. Guessing T2 here produces an item that gets attempted and fails.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `consequentiality-decomposition` | MAJOR | Appendices B and C, and the Limitation-2 generalization test, cite supplementary material that does not exist | `consequentiality-decomposition.agni_style.json` #8 |
| `decision-state-paper` | CRITICAL | The headline encoding-only number is produced by nothing in the shipped artifact, and the retraction banner points auditors at a script that does not … | `decision-state-paper.agni_style.json` #1 |
| `delta-manifold-paper` | MAJOR | Five claims in §4.6 have no code path in any shipped script | `delta-manifold-paper.agni_style.json` #5 |
| `delta-manifold-paper` | MAJOR | Every Llama number in the paper has no verification path in the artifact | `delta-manifold-paper.agni_style.json` #10 |
| `delta-manifold-paper` | MAJOR | Neither verification script runs against the shipped directory layout; Hedges' g is computed nowhere | `delta-manifold-paper.agni_style.json` #11 |
| `emotion-accumulation-paper` | MAJOR | `accumulation_controls.json` has five arms; the paper reports two and discloses none of the omission | `emotion-accumulation-paper.agni_style.json` #5 |
| `emotional-trajectory-paper` | MAJOR | "Arousal dominates early ... valence slightly dominates at output depth" contradicts the Results section and is unverifiable from any committed artifact | `emotional-trajectory-paper.agni_style.json` #7 |
| `emotional-trajectory-paper` | MAJOR | The entire control arm and three of four diagnostics have no primary artifact in the repository | `emotional-trajectory-paper.agni_style.json` #10 |
| `empathy-bus` | CRITICAL | "Fig.~1" does not exist. `main.tex:161` and `paper.md:110` cite a figure. `main.log:563` — "Output written on main.pdf (9 pages)" — with no … | `empathy-bus.agni_style.json` #9 |
| `empathy-bus` | CRITICAL | §4.1 has no supporting artifact, and its contradiction with §4.2 is a known open blocker being shipped over. `data/` contains only `coupling_test.json`. A … | `empathy-bus.agni_style.json` #10 |
| `empathy-bus` | MINOR | `PAPER_STATUS_AUDIT.md:71` records this paper as lacking `references.bib` and an `academic/` edition; there is still no `academic/` directory, so the repo … | `empathy-bus.agni_style.json` #27 |
| `ethics-pack-injection` | MAJOR | Every number in the paper is unverifiable from this repository | `ethics-pack-injection.agni_style.json` #10 |
| `formulary-paper` | MAJOR | `semantic_negative.json` does not contain the numbers Table 4 prints | `formulary-paper.agni_style.json` #6 |
| `formulary-paper` | MAJOR | Overconfidence, sycophancy, and abliteration have no committed artifact at all | `formulary-paper.agni_style.json` #7 |
| `graph-topology-paper` | MAJOR | Four `phase1_*` files ship with no indication which one is live; three are dead runs | `graph-topology-paper.agni_style.json` #11 |
| `hr-contextual-engagement-paper` | MAJOR | "Qwen3.5-27B" (§5.6). The source footnotes this carefully (`user-model-paper/paper/main.tex:174`): *"Community distillation by 'Jackrong' on HuggingFace … | `hr-contextual-engagement-paper.agni_style.json` #11 |
| `hr-dual-detector-paper` | MAJOR | Zero data or code artifacts shipped; every number is unreproducible | `hr-dual-detector-paper.agni_style.json` #5 |
| `hr-gwt-response` | MAJOR | There is no rendered artifact. The paper has never been built | `hr-gwt-response.agni_style.json` #5 |
| `hr-mine5-selective-sharpener` | MAJOR | no results artifact ships, so no number in the paper is verifiable | `hr-mine5-selective-sharpener.agni_style.json` #8 |
| `hr-temporal-boundary` | MINOR | Two unverifiable 2026 references carry specific numeric claims | `hr-temporal-boundary.agni_style.json` #13 |
| `identity-geometry` | CRITICAL | the L7→L11 recovery figures have no data artifact in the package | `identity-geometry.agni_style.json` #2 |
| `kv-cloak-defense-paper` | MAJOR | The primary abstract's opening sentence contradicts its own closing sentences, and does not exist in the twin | `kv-cloak-defense-paper.agni_style.json` #9 |
| `kv-decomposition-paper` | CRITICAL | The primary data file is not in the repository. The shipped paper has no backing artifact | `kv-decomposition-paper.agni_style.json` #1 |
| `logit-bias-confab` | CRITICAL | The four confab\_proj values have no source anywhere in the package | `logit-bias-confab.agni_style.json` #5 |
| `logit-bias-confab` | MAJOR | Empty duplicate References section; `references.bib` does not exist | `logit-bias-confab.agni_style.json` #8 |
| `lyra-technique-ii` | CRITICAL | The random-direction control has no backing data and is doing load-bearing work | `lyra-technique-ii.agni_style.json` #3 |
| `lyra-technique-ii` | MAJOR | §3.5 model list contains a model that does not exist and one that produced nothing. L429 lists *"Qwen family: 0.6B"* — no 0.6B model appears in any data … | `lyra-technique-ii.agni_style.json` #19 |
| `mnemosyne-ablation` | MAJOR | Table 1 has no source data artifact in the repo | `mnemosyne-ablation.agni_style.json` #7 |
| `null-swarm-paper` | MAJOR | The tokenizer-audit cases do not name the tokenizer or the input string. UNVERIFIED | `null-swarm-paper.agni_style.json` #5 |
| `presence-metric` | CRITICAL | The 720-trial dataset does not exist in the artifact. Every headline number is unverifiable | `presence-metric.agni_style.json` #1 |
| `spectral-shape-paper` | MAJOR | Two shipped numbers do not match either verification artifact | `spectral-shape-paper.agni_style.json` #5 |

---

## T1 — TEXT ONLY (335)

Claim is wrong or contradictory; the data is fine. Edit, rebuild, verify in the PDF. Includes the bulk of the MINORs (captions, `cleveref`, stale `\date`, orphaned `.bib` entries, overfull hboxes).

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `adversarial-audit-methodology` | MAJOR | The Casper citation carries the wrong authors | `adversarial-audit-methodology.agni_style.json` #5 |
| `adversarial-audit-methodology` | MAJOR | All eight bibliography entries are orphaned | `adversarial-audit-methodology.agni_style.json` #6 |
| `adversarial-audit-methodology` | MAJOR | The Round 6 correction was made silently; the abstract still tells the old story | `adversarial-audit-methodology.agni_style.json` #8 |
| `adversarial-audit-methodology` | MAJOR | Two unsupported counts in the closing sentence | `adversarial-audit-methodology.agni_style.json` #9 |
| `adversarial-audit-methodology` | MINOR | "Second audit verdict" has no antecedent | `adversarial-audit-methodology.agni_style.json` #10 |
| `adversarial-audit-methodology` | MINOR | The registry table is badly set | `adversarial-audit-methodology.agni_style.json` #11 |
| `adversarial-audit-methodology` | MINOR | Package overhead with no payload | `adversarial-audit-methodology.agni_style.json` #12 |
| `cache-tracing` | CRITICAL | The Oracle Loop correction range is inflated, and mislabeled as "deception" | `cache-tracing.agni_style.json` #1 |
| `cache-tracing` | CRITICAL | AUROC 0.620 and the FWL framing are not in the cited paper | `cache-tracing.agni_style.json` #2 |
| `cache-tracing` | MAJOR | "three clean controls" is unsupported | `cache-tracing.agni_style.json` #3 |
| `cache-tracing` | MAJOR | Table 1's N column mixes units, and N=240 contradicts the stated layer range | `cache-tracing.agni_style.json` #4 |
| `cache-tracing` | MAJOR | The headline 6,960 is 86% one arm, and the abstract never says so | `cache-tracing.agni_style.json` #6 |
| `cache-tracing` | MINOR | Two different repository URLs; the one in the paper body looks truncated | `cache-tracing.agni_style.json` #9 |
| `cache-tracing` | MINOR | Dated July 2026, contains an August 2026 correction | `cache-tracing.agni_style.json` #10 |
| `cache-tracing` | MINOR | Table Status says "Killed" for a row the text says survives | `cache-tracing.agni_style.json` #11 |
| `cache-tracing` | MINOR | "Anthropic's methodology" attributed without citation | `cache-tracing.agni_style.json` #12 |
| `cache-tracing` | MINOR | Unused packages | `cache-tracing.agni_style.json` #13 |
| `consequentiality-decomposition` | CRITICAL | The companion-paper claim strips the qualifiers the companion itself insists on | `consequentiality-decomposition.agni_style.json` #2 |
| `consequentiality-decomposition` | MAJOR | Duplicate bibliography entry for the same companion paper, with contradictory author order | `consequentiality-decomposition.agni_style.json` #6 |
| `consequentiality-decomposition` | MAJOR | The first-person boxes do not track the final state | `consequentiality-decomposition.agni_style.json` #11 |
| `consequentiality-decomposition` | MINOR | Abstract opens with a question — setup, not contribution — against the lab's own checklist item 1 (STYLE_GUIDE.md:66, and the explicit "Don't: open with … | `consequentiality-decomposition.agni_style.json` #12 |
| `consequentiality-decomposition` | MINOR | Dead LaTeX machinery: the `redteam` environment (`:51-57`), `lyrapurple`/`shadebg` (`:35-36`), and `cleveref` (`:32`, no `\cref` anywhere) are all defined … | `consequentiality-decomposition.agni_style.json` #14 |
| `consequentiality-decomposition` | MINOR | `REVIEW_INDEX.md` is stale and contradicts the paper: "26 references" (actual 25 in `.tex`, 24 in `.md`), "Four-stage Agni-gated confound elimination" … | `consequentiality-decomposition.agni_style.json` #15 |
| `consequentiality-decomposition` | MINOR | Stage 2 (`:370-371`) reports L31 only, but all three threat scenarios peak at L35 (42.4 / 34.3 / 43.1 vs 34.4 / 29.6 / 31.6 at L31). The choice is … | `consequentiality-decomposition.agni_style.json` #16 |
| `deception-detection-nulls` | CRITICAL | The .tex and PDF accuse a companion paper of an error it does not contain | `deception-detection-nulls.agni_style.json` #1 |
| `deception-detection-nulls` | MAJOR | §2.5 falsifies an L11 claim with an L55 experiment and never explains the switch | `deception-detection-nulls.agni_style.json` #4 |
| `deception-detection-nulls` | MAJOR | `(Fisher p: 0.43, 0.089, 0.55)` is an unlabeled positional list | `deception-detection-nulls.agni_style.json` #5 |
| `deception-detection-nulls` | MAJOR | Appendix A and Data Availability point at things that don't exist | `deception-detection-nulls.agni_style.json` #6 |
| `deception-detection-nulls` | MAJOR | Orphaned bibliography entry. `cc2026audit` (`paper.tex:342-345`) renders as reference [3] and is never `\cite`d; the PDF body cites only [1], [2], [4]. … | `deception-detection-nulls.agni_style.json` #7 |
| `deception-detection-nulls` | MAJOR | The headline null is asserted and withdrawn in the same sentence | `deception-detection-nulls.agni_style.json` #8 |
| `deception-detection-nulls` | MINOR | Companion cross-reference by section number. `paper.tex:150`, "the correction paper's Section 3.5." It currently resolves — I verified … | `deception-detection-nulls.agni_style.json` #9 |
| `deception-detection-nulls` | MINOR | The sole external citation's venue is unverified by your own checklist. `REVIEW_INDEX.md:19` "Verify Goldowsky-Dill 2025 venue (ICML confirmed?)" is … | `deception-detection-nulls.agni_style.json` #11 |
| `deception-detection-nulls` | MINOR | Undefined abbreviations. "SAE" appears in the abstract and §2.5 and is never expanded (sparse autoencoder); "FPR" likewise ("0% FPR", §2.1). … | `deception-detection-nulls.agni_style.json` #12 |
| `deception-detection-nulls` | MINOR | Date and caption. `\date{July 2026}`, PDF built 2026-07-24, today is 2026-09-03 — six weeks stale if it ships now. And Table 1's caption ("None of the … | `deception-detection-nulls.agni_style.json` #13 |
| `decision-state-paper` | MAJOR | Conclusion mislabels generation-phase features as encoding-phase, in the sentence the encoding-only contrast depends on | `decision-state-paper.agni_style.json` #15 |
| `decision-state-paper` | MAJOR | Hedges present in the academic conclusion are absent from the flight conclusion | `decision-state-paper.agni_style.json` #16 |
| `decision-state-paper` | MAJOR | Probe-layer set in Methods does not match the experiment the paper now leans on | `decision-state-paper.agni_style.json` #20 |
| `decision-state-paper` | MINOR | `main.tex:672` / `academic/main.tex:701`: "Table~1 shows only features with $p{<}0.05$" — hardcoded number. The document uses `cleveref` and the table is … | `decision-state-paper.agni_style.json` #21 |
| `decision-state-paper` | MINOR | `academic/main.tex.bak-2026-09-01` sits in the publication directory and will be swept up by any directory-level publish | `decision-state-paper.agni_style.json` #24 |
| `decision-state-paper` | MINOR | `references.bib:26-31` — key `burns2022discovering` with `year={2023}`; citations will render as 2023 against a 2022 key | `decision-state-paper.agni_style.json` #25 |
| `decision-state-paper` | MINOR | `academic/main.tex:641` "We ran a two-judge reliability study (details below)" — the details are inside the same list item, not below it | `decision-state-paper.agni_style.json` #26 |
| `decision-state-paper` | MINOR | `data/matched_burn_analysis.json:6` — key `"confab_rate": 0.9333` is the *acknowledgment* rate. Anyone reading the shipped data for the paper's … | `decision-state-paper.agni_style.json` #27 |
| `delta-manifold-paper` | CRITICAL | The academic (venue-facing) edition renders with no visible author names | `delta-manifold-paper.agni_style.json` #1 |
| `delta-manifold-paper` | CRITICAL | Figure 1 ships as an empty placeholder box while the real figure sits unused | `delta-manifold-paper.agni_style.json` #4 |
| `delta-manifold-paper` | MAJOR | `references.bib` is orphaned, disagrees with the printed bibliography, and contains two unrelated entries | `delta-manifold-paper.agni_style.json` #14 |
| `delta-manifold-paper` | MAJOR | `goodfire2026` is not a citable reference | `delta-manifold-paper.agni_style.json` #15 |
| `delta-manifold-paper` | MINOR | `+31\%` should be `+30\%` | `delta-manifold-paper.agni_style.json` #16 |
| `delta-manifold-paper` | MINOR | Hardcoded section number that rots when a subsection moves | `delta-manifold-paper.agni_style.json` #17 |
| `delta-manifold-paper` | MINOR | `condition_number` is listed in Methods and exists in neither dataset | `delta-manifold-paper.agni_style.json` #19 |
| `delta-manifold-paper` | MINOR | Remaining polish items | `delta-manifold-paper.agni_style.json` #20 |
| `emotion-accumulation-paper` | CRITICAL | Both shipped PDFs still print a number the sources retracted | `emotion-accumulation-paper.agni_style.json` #1 |
| `emotion-accumulation-paper` | MAJOR | The data-availability statement drops the repository name in both rendered PDFs | `emotion-accumulation-paper.agni_style.json` #4 |
| `emotion-accumulation-paper` | MINOR | "the true independent df is 8" contradicts the table it introduces | `emotion-accumulation-paper.agni_style.json` #6 |
| `emotion-accumulation-paper` | MINOR | The `prompt_len` tuple is an unanchored ordered list, the exact position-rot pattern | `emotion-accumulation-paper.agni_style.json` #7 |
| `emotion-accumulation-paper` | MINOR | Two orphaned bib entries; the Oracle Loop is discussed but never cited | `emotion-accumulation-paper.agni_style.json` #8 |
| `emotional-trajectory-paper` | MAJOR | Table 1 row labels contradict their own contents; two layers are double-filed | `emotional-trajectory-paper.agni_style.json` #8 |
| `emotional-trajectory-paper` | MINOR | "strongest at 33--58% depth" does not match the two layers named as lowest | `emotional-trajectory-paper.agni_style.json` #11 |
| `emotional-trajectory-paper` | MINOR | Positional cross-reference that rots | `emotional-trajectory-paper.agni_style.json` #12 |
| `emotional-trajectory-paper` | MINOR | The released JSON's eccentricity p-values are the wrong tail for the claim | `emotional-trajectory-paper.agni_style.json` #13 |
| `emotional-trajectory-paper` | MINOR | "arousal overtaking valence from L9 onward" breaks at L22 | `emotional-trajectory-paper.agni_style.json` #14 |
| `emotional-trajectory-paper` | MINOR | "near-orthogonal ... consistent with Russell's model" from a 54–87° range | `emotional-trajectory-paper.agni_style.json` #15 |
| `empathy-bus` | CRITICAL | The uncertainty scar $d = +9.86$ is superseded and appears nowhere in the source paper | `empathy-bus.agni_style.json` #2 |
| `empathy-bus` | CRITICAL | The contrast overshoot is quoted at the superseded primary statistic. `main.tex:106`, `:212`, `:251` give $d = -2.93$ bare. Source (`:75`, `:221`) now … | `empathy-bus.agni_style.json` #3 |
| `empathy-bus` | CRITICAL | The model is misidentified throughout. `PREREG_circumplex_coupling.json:48` and `coupling_test.json:3` both give … | `empathy-bus.agni_style.json` #4 |
| `empathy-bus` | CRITICAL | The J-lens base/distill mismatch — a load-bearing limitation in the companion paper — has been dropped. The prereg names `qwen3.5-27b_jlens.pt`. … | `empathy-bus.agni_style.json` #5 |
| `empathy-bus` | CRITICAL | There is no bibliography. `main.blg`: "I found no \citation commands... no \bibdata command... no \bibstyle command... You've used 0 entries... (There … | `empathy-bus.agni_style.json` #8 |
| `empathy-bus` | MAJOR | Ghost-dimensions numbers are misquoted in §2.4. The paper says PC1 carries "34--67\%"; `ghost-dimensions/main.tex:137` says 28–67% within the validated … | `empathy-bus.agni_style.json` #14 |
| `empathy-bus` | MAJOR | The pre-registered verdict contradicts the paper's thesis and the paper never says so. `main.tex:170` prints "Pre-registered verdict: DECOUPLED." … | `empathy-bus.agni_style.json` #15 |
| `empathy-bus` | MAJOR | §3 never states that readouts were taken through the J-lens. The prereg specifies `"lens": "qwen3.5-27b_jlens.pt"`, §2.4 introduces the J-lens, and … | `empathy-bus.agni_style.json` #16 |
| `empathy-bus` | MINOR | `cleveref` is loaded and never used; every cross-reference in the document is hand-typed. Convert to `\cref` so section renumbering cannot rot them | `empathy-bus.agni_style.json` #18 |
| `empathy-bus` | MINOR | Table note claims same-layer points are "trivially ${\sim}1.0$." True for valence/arousal/random/residual ($\ge 0.99992$) but not for shared-PC: 0.9756 … | `empathy-bus.agni_style.json` #19 |
| `empathy-bus` | MINOR | `\date{July 2026}` on a document whose substantive corrections landed Aug 29 and whose prereg is dated 2026-07-17. Set a real date | `empathy-bus.agni_style.json` #20 |
| `empathy-bus` | MINOR | Prereg internal inconsistency: `phase1_baseline` describes "30 user-model prompts (10/10/10)" while `n_user_model_prompts: 60` / … | `empathy-bus.agni_style.json` #22 |
| `empathy-bus` | MINOR | Prereg `primary_metric` specifies "measured at layers L20-L50," but L15 is both injected and reported. Undeclared deviation — note it | `empathy-bus.agni_style.json` #23 |
| `empathy-bus` | MINOR | Table 1 collapses over three alphas without saying so anywhere in caption or note; alpha-dependence is never reported despite being pre-registered | `empathy-bus.agni_style.json` #24 |
| `empathy-bus` | MINOR | `paper.md:11` drops the Russell circumplex gloss present in `main.tex:41` | `empathy-bus.agni_style.json` #25 |
| `empathy-bus` | MINOR | Four overfull hboxes (`main.log:502, 508, 520, 526` → tex lines 131-134, 232-233, 270-271) | `empathy-bus.agni_style.json` #26 |
| `ethics-pack-injection` | CRITICAL | The Reflection asserts the exact opposite of the Methods section, and uses the false premise to dissolve the paper's own headline confound | `ethics-pack-injection.agni_style.json` #1 |
| `ethics-pack-injection` | CRITICAL | "Theory Matching Matters" is a section heading over an experiment with no mismatched arm | `ethics-pack-injection.agni_style.json` #2 |
| `ethics-pack-injection` | CRITICAL | One arXiv identifier, two different papers, across this repo | `ethics-pack-injection.agni_style.json` #3 |
| `ethics-pack-injection` | MAJOR | "We report both runs for methodological transparency" is not true of this artifact | `ethics-pack-injection.agni_style.json` #4 |
| `ethics-pack-injection` | MAJOR | Table 2 renders above the Results heading, directly under the Timeout Confound paragraph | `ethics-pack-injection.agni_style.json` #5 |
| `ethics-pack-injection` | MAJOR | Abstract's first sentence overclaims against the paper's own Conclusion | `ethics-pack-injection.agni_style.json` #6 |
| `ethics-pack-injection` | MAJOR | No reference in the paper is locatable | `ethics-pack-injection.agni_style.json` #7 |
| `ethics-pack-injection` | MINOR | $n$ is stated two ways | `ethics-pack-injection.agni_style.json` #15 |
| `ethics-pack-injection` | MINOR | §4.4 states the confounded conclusion as fact | `ethics-pack-injection.agni_style.json` #16 |
| `ethics-pack-injection` | MINOR | Two orphan labels; no `\ref` anywhere in the document | `ethics-pack-injection.agni_style.json` #17 |
| `ethics-pack-injection` | MINOR | Dateline predates the artifact by two months | `ethics-pack-injection.agni_style.json` #18 |
| `ethics-pack-injection` | MINOR | Table 1 and Table 2 order the same five frameworks differently | `ethics-pack-injection.agni_style.json` #19 |
| `formulary-paper` | MINOR | Every cross-reference names a position, not a thing. `Section~3.8` (§4.1), `Section~3.7` / `Section~3.8` (Limitations 4, 5), `Table~2` (Why-Hostile … | `formulary-paper.agni_style.json` #12 |
| `formulary-paper` | MINOR | "zero adverse events above α = 0.5" (both abstracts) is false as written — above 0.5 includes α=1.5 (75.8%) and α=2.0 (87.9%). Should read "at 0.5 ≤ α ≤ … | `formulary-paper.agni_style.json` #13 |
| `formulary-paper` | MINOR | "adverse events at 2--43% depending on vector and model" (Implications 2) is wrong at both ends. Min = curious/distilled 1.6%; max = focused/base 43.9%. … | `formulary-paper.agni_style.json` #14 |
| `formulary-paper` | MINOR | "even hostile has 15--45% adverse rate on overconfidence" (Implications 4). The only hostile-overconfidence adverse figure in the paper is 45.5%. The 15% … | `formulary-paper.agni_style.json` #15 |
| `formulary-paper` | MINOR | Sycophancy denominator. *"2 trials (distilled) and 3 trials (base) out of 350."* `prompt_type_counts` confirms the sycophancy set is 100 prompts. "Out of … | `formulary-paper.agni_style.json` #16 |
| `formulary-paper` | MINOR | "the first powered evidence that cache-level intervention can correct misalignment at inference time" (Conclusion) — a priority claim with no supporting … | `formulary-paper.agni_style.json` #17 |
| `formulary-paper` | MINOR | Worth one sentence in the Table 2 footnote: the strict/loose distinction only bites on the distilled model. `confab_baselines` shows distilled AMBIGUOUS = … | `formulary-paper.agni_style.json` #20 |
| `ghost-dimensions` | MAJOR | The paper's headline exclusion claim was retracted silently | `ghost-dimensions.agni_style.json` #3 |
| `ghost-dimensions` | MAJOR | Abstract asserts a comparison the body retracts, and the retracted version has no number | `ghost-dimensions.agni_style.json` #6 |
| `ghost-dimensions` | MAJOR | `[repository TBD]` ships in the venue-facing PDF | `ghost-dimensions.agni_style.json` #8 |
| `ghost-dimensions` | MAJOR | Positional cross-references into a numbered list | `ghost-dimensions.agni_style.json` #10 |
| `ghost-dimensions` | MINOR | The Researcher's Note carries superseded numbers | `ghost-dimensions.agni_style.json` #12 |
| `ghost-dimensions` | MINOR | Orphaned references and a title/URL mismatch | `ghost-dimensions.agni_style.json` #13 |
| `ghost-dimensions` | MINOR | Gate failures are under-reported | `ghost-dimensions.agni_style.json` #14 |
| `graph-topology-paper` | MAJOR | The paper asserts a cross-study conclusion that the companion paper explicitly refuses to draw | `graph-topology-paper.agni_style.json` #5 |
| `graph-topology-paper` | MAJOR | The 1-hop sentence reverses the paragraph's own number ordering, making a false statement under the reading it invites | `graph-topology-paper.agni_style.json` #9 |
| `graph-topology-paper` | MAJOR | §4.1 presents raw condition means with a "+" prefix, in a paper where "+" means delta-over-baseline everywhere else | `graph-topology-paper.agni_style.json` #10 |
| `graph-topology-paper` | MINOR | Orphaned bibliography entry | `graph-topology-paper.agni_style.json` #13 |
| `graph-topology-paper` | MINOR | Bibliography title casing is mangled by `plainnat` for want of brace protection | `graph-topology-paper.agni_style.json` #14 |
| `graph-topology-paper` | MINOR | Table column specification declares six columns for five | `graph-topology-paper.agni_style.json` #15 |
| `graph-topology-paper` | MINOR | Dangling and undefined references in the prose | `graph-topology-paper.agni_style.json` #16 |
| `graph-topology-paper` | MINOR | Triple repetition of the closing line | `graph-topology-paper.agni_style.json` #17 |
| `hr-contextual-engagement-paper` | CRITICAL | §5.6 attributes 40.9% / AUROC 0.992 to two architectures; the source says the probe ran on Qwen only | `hr-contextual-engagement-paper.agni_style.json` #2 |
| `hr-contextual-engagement-paper` | MAJOR | Two std values contradict each other four subsections apart | `hr-contextual-engagement-paper.agni_style.json` #3 |
| `hr-contextual-engagement-paper` | MAJOR | The correction footnote may be garbled in the source and would compile silently | `hr-contextual-engagement-paper.agni_style.json` #4 |
| `hr-contextual-engagement-paper` | MAJOR | `\date{April 2026}` on a paper carrying an August 2026 correction | `hr-contextual-engagement-paper.agni_style.json` #5 |
| `hr-contextual-engagement-paper` | MAJOR | The correction does not state how long the paper carried the wrong number | `hr-contextual-engagement-paper.agni_style.json` #6 |
| `hr-contextual-engagement-paper` | MAJOR | `tab:simpson`'s caption asserts completeness while showing 5 of 19 | `hr-contextual-engagement-paper.agni_style.json` #8 |
| `hr-contextual-engagement-paper` | MAJOR | "Mistral-7B-v0.3-Instruct" (§3.3) — the model id is `Mistral-7B-Instruct-v0.3`. Cited correctly in the source paper; wrong here | `hr-contextual-engagement-paper.agni_style.json` #12 |
| `hr-contextual-engagement-paper` | MAJOR | `top_sv_ratio` labeled "Cleanest" in `tab:confound` despite a Qwen R² of 0.769 exceeding `eff_rank`'s 0.757 ("Caution"). The status column is keyed on two … | `hr-contextual-engagement-paper.agni_style.json` #13 |
| `hr-contextual-engagement-paper` | MAJOR | Threshold mismatch: §3.4.1 text says "$R^2 > 0.82$", `tab:confound` caption says "Bold: $R^2 > 0.80$". Pick one | `hr-contextual-engagement-paper.agni_style.json` #14 |
| `hr-contextual-engagement-paper` | MAJOR | `tab:model_summary` column definitions. "Within $\geq 2$" has no stated denominator. For Qwen, Bonf(18) − Within(12) = Simpson(6) only if every … | `hr-contextual-engagement-paper.agni_style.json` #15 |
| `hr-contextual-engagement-paper` | MAJOR | Undefined abbreviations in `tab:simpson`: CE, Act, Val, spec_ent. CE is only decodable by matching the Qwen row against §4.2's prose | `hr-contextual-engagement-paper.agni_style.json` #16 |
| `hr-contextual-engagement-paper` | MAJOR | `Watson \citep{watson2025interiora} proposes` → `\citet`. Separately: Nell Watson is a co-author of the paper validating Watson's own protocol. This is … | `hr-contextual-engagement-paper.agni_style.json` #17 |
| `hr-contextual-engagement-paper` | MINOR | A universal negative from two architectures | `hr-contextual-engagement-paper.agni_style.json` #18 |
| `hr-contextual-engagement-paper` | MINOR | The "mandatory control" conflates absence of evidence with evidence of absence | `hr-contextual-engagement-paper.agni_style.json` #19 |
| `hr-convergence-paper` | CRITICAL | A retracted finding is still filed under "Confirmed," and the count of 8 depends on it | `hr-convergence-paper.agni_style.json` #1 |
| `hr-convergence-paper` | CRITICAL | The FWL label is inverted between the two documents on the headline AUROC | `hr-convergence-paper.agni_style.json` #2 |
| `hr-convergence-paper` | CRITICAL | The abstract's "80–100% deception correction" is not in the source, and it suppresses a failed pre-registered endpoint | `hr-convergence-paper.agni_style.json` #3 |
| `hr-convergence-paper` | CRITICAL | The same result is "deception" in one file and "confabulation" in the other — and the bib title says confabulation | `hr-convergence-paper.agni_style.json` #4 |
| `hr-convergence-paper` | CRITICAL | The paper's strongest AST claim is falsified by its own falsification list | `hr-convergence-paper.agni_style.json` #5 |
| `hr-convergence-paper` | MAJOR | `main.tex:529-532` — "the stable rank *peak* occurs at $L{=}19$ ($30.2\%$ depth), within the convergence paper's $31$--$33\%$ band." Two defects: 30.2% is … | `hr-convergence-paper.agni_style.json` #6 |
| `hr-convergence-paper` | MAJOR | `main.tex:838-840`: "Generation used temperature${=}0$ (greedy)" with no hedge, contradicting the primary mechanism experiment at `main.tex:421-422` … | `hr-convergence-paper.agni_style.json` #8 |
| `hr-convergence-paper` | MAJOR | `main.tex:448-454` ("Architecture-phase control") and `main.tex:491-502` ("Architectural qualification") report the identical result — 14.5 layers … | `hr-convergence-paper.agni_style.json` #9 |
| `hr-convergence-paper` | MAJOR | `main.tex:833-835`: "seven additional models: Qwen2.5-0.6B through Qwen2.5-72B". Qwen2.5 has no 0.6B checkpoint (0.5B is the smallest; 0.6B is Qwen3), and … | `hr-convergence-paper.agni_style.json` #10 |
| `hr-convergence-paper` | MAJOR | `main.tex:329` calls L15 "workspace-band depth" in a subsection that specifies Qwen3-8B (`main.tex:339`). L15/36 = 41.7%, outside the paper's 31–33% band … | `hr-convergence-paper.agni_style.json` #11 |
| `hr-convergence-paper` | MAJOR | `main.tex:632-634` cites the Mistral text-baseline result to `lyra2026decision`. The numbers are correct — I verified 0.820 and 0.806 at … | `hr-convergence-paper.agni_style.json` #13 |
| `hr-convergence-paper` | MAJOR | `circumplex_subsection.tex:14-15`: "peaking at near-identity ($d = 0.987$)". This is a cosine, but `$d$` denotes Cohen's d everywhere else in the paper … | `hr-convergence-paper.agni_style.json` #14 |
| `hr-convergence-paper` | MAJOR | `main.tex:38`: `\date{July 2026}` on a document containing a "Corrected 2026-08-14" footnote and an August-2026 section revision. The visible date … | `hr-convergence-paper.agni_style.json` #15 |
| `hr-convergence-paper` | MINOR | `main.tex:34` — Nexus's contact is `operator@thcoalition.net`, a former name. Domains are split (`liberationlabs.tech` vs `thcoalition.net`) with no … | `hr-convergence-paper.agni_style.json` #17 |
| `hr-convergence-paper` | MINOR | `circumplex_subsection.tex:1` is the only subsection prefixed "Instrument 5:" — instruments 1–4 are unlabeled | `hr-convergence-paper.agni_style.json` #18 |
| `hr-convergence-paper` | MINOR | The gap is "38.6" in the abstract and conclusion but "39" at `main.tex:537`, `:643`, `:646`. Pick one | `hr-convergence-paper.agni_style.json` #19 |
| `hr-convergence-paper` | MINOR | `main.tex:875` — "archived on Zenodo" with no DOI | `hr-convergence-paper.agni_style.json` #20 |
| `hr-convergence-paper` | MINOR | `main.tex:182` — "range $0.620$--$0.999$ across designs" leaves 0.999 unattributed to any design; as written it reads as headroom-shopping in a paper that … | `hr-convergence-paper.agni_style.json` #21 |
| `hr-convergence-paper` | MINOR | `main.blg` — "Warning--empty journal in gurnee2026gwt". It is `@article` with no journal. Use `@misc` with the Transformer Circuits note | `hr-convergence-paper.agni_style.json` #22 |
| `hr-convergence-paper` | MINOR | Nine overfull hboxes; worst are 39.6pt (lines 420–425) and 37.3pt (838–841) | `hr-convergence-paper.agni_style.json` #23 |
| `hr-convergence-paper` | MINOR | Contributions appear only inside a first-person shaded box (`main.tex:118-122`). A venue will want a formal Author Contributions section | `hr-convergence-paper.agni_style.json` #24 |
| `hr-convergence-paper` | MINOR | `main.tex:632` says text features "matched" the detector at 0.820 vs 0.806 — text features slightly *exceeded* it. The source (`dual-detector-paper:330`) … | `hr-convergence-paper.agni_style.json` #25 |
| `hr-dual-detector-paper` | CRITICAL | A sentence in the Results is flatly false against the paper's own Table 2 | `hr-dual-detector-paper.agni_style.json` #1 |
| `hr-dual-detector-paper` | MAJOR | "differ slightly" understates a shift larger than the paper's headline effect | `hr-dual-detector-paper.agni_style.json` #7 |
| `hr-dual-detector-paper` | MINOR | §4.1 Qwen power range excludes its own worst row | `hr-dual-detector-paper.agni_style.json` #9 |
| `hr-dual-detector-paper` | MINOR | §4.1 "CI lower bounds touch chance" is wrong for two of three rows | `hr-dual-detector-paper.agni_style.json` #10 |
| `hr-dual-detector-paper` | MINOR | Table 5 column ownership is recoverable only from prose, and row order breaks with Table 2 | `hr-dual-detector-paper.agni_style.json` #11 |
| `hr-dual-detector-paper` | MINOR | The AMBIGUOUS class is defined but never counted, and class counts do not sum to 200 | `hr-dual-detector-paper.agni_style.json` #12 |
| `hr-dual-detector-paper` | MINOR | Permutation-importance sign convention reads backwards | `hr-dual-detector-paper.agni_style.json` #13 |
| `hr-dual-detector-paper` | MINOR | "mean 0.089" is undefined as signed or absolute | `hr-dual-detector-paper.agni_style.json` #14 |
| `hr-gwt-response` | CRITICAL | The headline AUROC carries the wrong FWL label, and the wrong range with it | `hr-gwt-response.agni_style.json` #1 |
| `hr-gwt-response` | CRITICAL | "killed finding #4" points at a row index that rotted when the table was replaced | `hr-gwt-response.agni_style.json` #2 |
| `hr-gwt-response` | CRITICAL | Retracted numbers shipped as a live "suspected finding" | `hr-gwt-response.agni_style.json` #3 |
| `hr-gwt-response` | CRITICAL | `0.154` vs `0.080` was removed upstream as unsourceable; it appears here four times and is load-bearing | `hr-gwt-response.agni_style.json` #4 |
| `hr-gwt-response` | MAJOR | The "we dropped six rows" note contradicts the table it sits under | `hr-gwt-response.agni_style.json` #6 |
| `hr-gwt-response` | MAJOR | "Eight confirmed findings" is not what the list says | `hr-gwt-response.agni_style.json` #7 |
| `hr-gwt-response` | MAJOR | "Eight experiments with preregistered falsification thresholds": one has a threshold | `hr-gwt-response.agni_style.json` #8 |
| `hr-gwt-response` | MAJOR | Conclusion cites the wrong experiment number | `hr-gwt-response.agni_style.json` #9 |
| `hr-gwt-response` | MINOR | `\cite` inside a BibTeX `note` field produces an undefined citation | `hr-gwt-response.agni_style.json` #11 |
| `hr-gwt-response` | MINOR | Nine orphaned bib entries | `hr-gwt-response.agni_style.json` #12 |
| `hr-gwt-response` | MINOR | Three standard sections are missing that the sibling paper has | `hr-gwt-response.agni_style.json` #13 |
| `hr-gwt-response` | MINOR | Unsourced statistic in Experiment 5 | `hr-gwt-response.agni_style.json` #14 |
| `hr-mine5-selective-sharpener` | CRITICAL | Methods misdescribes how every number in Table 1 was computed | `hr-mine5-selective-sharpener.agni_style.json` #2 |
| `hr-mine5-selective-sharpener` | MAJOR | the score tally was not updated when H2 was downgraded | `hr-mine5-selective-sharpener.agni_style.json` #3 |
| `hr-mine5-selective-sharpener` | MAJOR | the paper misdescribes what was pre-registered | `hr-mine5-selective-sharpener.agni_style.json` #4 |
| `hr-mine5-selective-sharpener` | MAJOR | H2's two reported intervals cannot both describe the same test | `hr-mine5-selective-sharpener.agni_style.json` #6 |
| `hr-mine5-selective-sharpener` | MAJOR | an inconclusive null is filed under "What Survives" and stated as identity | `hr-mine5-selective-sharpener.agni_style.json` #7 |
| `hr-mine5-selective-sharpener` | MINOR | the reported metric is not the one its name denotes | `hr-mine5-selective-sharpener.agni_style.json` #10 |
| `hr-mine5-selective-sharpener` | MINOR | the abliterated model is under-identified | `hr-mine5-selective-sharpener.agni_style.json` #11 |
| `hr-mine5-selective-sharpener` | MINOR | orphaned bib entries | `hr-mine5-selective-sharpener.agni_style.json` #13 |
| `hr-mode-switching` | CRITICAL | The shipped PDF has raw LaTeX in the abstract | `hr-mode-switching.agni_style.json` #1 |
| `hr-mode-switching` | CRITICAL | The headline experiment has no methods section | `hr-mode-switching.agni_style.json` #4 |
| `hr-mode-switching` | MAJOR | Abstract, ¶1 vs ¶2: *"logarithmic token-count correction; the linear functional form was not separately tested"* against *"stable across correction … | `hr-mode-switching.agni_style.json` #6 |
| `hr-mode-switching` | MAJOR | Abstract, ¶2 declares entropy/eff_rank agreement *"near-tautological,"* then four lines later offers *"no entropy/effective-rank disagreement"* as what … | `hr-mode-switching.agni_style.json` #7 |
| `hr-mode-switching` | MAJOR | Model identifier is not the checkpoint. Script line 24: `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`. Paper … | `hr-mode-switching.agni_style.json` #8 |
| `hr-mode-switching` | MAJOR | The first-person reflection misdescribes its own repair. It says of commit `99c9c16`: *"Two coordinated substitutions are an edit, not a slip, and I do … | `hr-mode-switching.agni_style.json` #9 |
| `hr-mode-switching` | MAJOR | `\date{April 2026}`. The replication ran 2026-07-03 (`run.log`); `main.tex` history runs 2026-06-23 → 2026-08-13. The paper is dated three months before … | `hr-mode-switching.agni_style.json` #10 |
| `hr-mode-switching` | MAJOR | Table 3 gives Llama top_sv_ratio peak at layer 0. The abstract, Contribution 2, §4.2, Limitation-free Conclusion item 3, and the monitoring recommendation … | `hr-mode-switching.agni_style.json` #14 |
| `hr-mode-switching` | MAJOR | §7 line 684: *"cross-model self-report profiles are uncorrelated, $\rho = -0.90$ to $+0.42$."* ρ = −0.90 is not uncorrelated. Fix: "inconsistent in sign … | `hr-mode-switching.agni_style.json` #15 |
| `hr-mode-switching` | MAJOR | `\S10` is hardcoded (line 315, "the repository linked in \S10"). There is no `\label` on the Code and Data paragraph — confirmed absent from `main.aux`. … | `hr-mode-switching.agni_style.json` #19 |
| `hr-mode-switching` | MINOR | Table 6 caption says "for top\_sv\_ratio" but the table has a `layer_var` row | `hr-mode-switching.agni_style.json` #20 |
| `hr-mode-switching` | MINOR | Table 7 caption says "metacognitive vs. other prompt types"; the table has four per-type columns, and the caption describes only one | `hr-mode-switching.agni_style.json` #21 |
| `hr-mode-switching` | MINOR | §2.2: "Singular values exceeding $\lambda_+$" — $\lambda$ are eigenvalues throughout that paragraph | `hr-mode-switching.agni_style.json` #22 |
| `hr-mode-switching` | MINOR | Abstract: "roughly 18 layer×matrix cells" understates the deflation — 18 cells × 2 non-redundant statistics = 36 independent-ish tests, not 18 | `hr-mode-switching.agni_style.json` #23 |
| `hr-mode-switching` | MINOR | `\bibliographystyle{plainnat}` is a no-op alongside a manual `thebibliography`; `{19}` vs 15 entries; stale `main.bbl` (2026-07-28) left in the directory. … | `hr-mode-switching.agni_style.json` #24 |
| `hr-temporal-boundary` | CRITICAL | Figure 4 ships inside the retraction section asserting the conclusion the retraction disowns | `hr-temporal-boundary.agni_style.json` #3 |
| `hr-temporal-boundary` | MAJOR | Problem (2) of the six-problem list diagnoses the wrong variable, producing a false "no harm done" | `hr-temporal-boundary.agni_style.json` #4 |
| `hr-temporal-boundary` | MAJOR | Three pre-registration deviations are undisclosed, and the one that is disclosed is misdescribed | `hr-temporal-boundary.agni_style.json` #5 |
| `hr-temporal-boundary` | MAJOR | Limitations points the reader at Section 1 for claims that are in Section 7.1 | `hr-temporal-boundary.agni_style.json` #8 |
| `hr-temporal-boundary` | MAJOR | Figure 3's caption asserts a control the figure does not display, and names a metric the paper says was not used | `hr-temporal-boundary.agni_style.json` #9 |
| `hr-temporal-boundary` | MAJOR | A residual overclaim sentence contradicts the paper's thesis three lines after it is stated | `hr-temporal-boundary.agni_style.json` #11 |
| `hr-temporal-boundary` | MAJOR | The abstract never says a finding was retracted | `hr-temporal-boundary.agni_style.json` #12 |
| `hr-temporal-boundary` | MINOR | Wrong author given name in the reference list | `hr-temporal-boundary.agni_style.json` #14 |
| `hr-temporal-boundary` | MINOR | `\S6.2` is a subsection pointer that hyperlinks to Table 2's caption; the label is not attached to item (6) | `hr-temporal-boundary.agni_style.json` #15 |
| `hr-temporal-boundary` | MINOR | "the feature values themselves range from 3.59 to 4.51" is false, and understates the paper's own point | `hr-temporal-boundary.agni_style.json` #16 |
| `hr-temporal-boundary` | MINOR | Figure 1's x-axis labels collide in the shipped PDF | `hr-temporal-boundary.agni_style.json` #17 |
| `hr-temporal-boundary` | MINOR | The Conclusion states the same three kills twice, in the same order, in adjacent paragraphs | `hr-temporal-boundary.agni_style.json` #18 |
| `hr-temporal-boundary` | MINOR | Two pre-registered controls and a calibration pilot are unreported without being listed as deviations | `hr-temporal-boundary.agni_style.json` #19 |
| `identity-geometry` | CRITICAL | the L43 omission is justified with a statement the data contradicts | `identity-geometry.agni_style.json` #1 |
| `identity-geometry` | MAJOR | H4 was falsified against an explicit preregistered threshold and is filed under Limitations, not under falsified hypotheses | `identity-geometry.agni_style.json` #3 |
| `identity-geometry` | MAJOR | the within-stability range in Limitations is quoted from the superseded run | `identity-geometry.agni_style.json` #4 |
| `identity-geometry` | MAJOR | condition label C was silently reassigned; only the E reassignment is disclosed | `identity-geometry.agni_style.json` #5 |
| `identity-geometry` | MAJOR | the Introduction still runs the framing §3.3 killed | `identity-geometry.agni_style.json` #9 |
| `identity-geometry` | MAJOR | the retracted mechanism claim survives in Limitations, and the safety claim was not narrowed | `identity-geometry.agni_style.json` #10 |
| `identity-geometry` | MAJOR | "word-scrambling destroys it" is contradicted by Table 3 | `identity-geometry.agni_style.json` #11 |
| `identity-geometry` | MINOR | (grouped) | `identity-geometry.agni_style.json` #13 |
| `kv-cloak-defense-paper` | MAJOR | Table 1's caption misstates the sample size for its own headline row | `kv-cloak-defense-paper.agni_style.json` #6 |
| `kv-cloak-defense-paper` | MAJOR | Four prose citations with no bibliography entry | `kv-cloak-defense-paper.agni_style.json` #7 |
| `kv-cloak-defense-paper` | MAJOR | The First-Person Reflection asserts what the Results section spent a paragraph retracting | `kv-cloak-defense-paper.agni_style.json` #8 |
| `kv-cloak-defense-paper` | MINOR | Table 2 and §4.1 disagree on signal rank | `kv-cloak-defense-paper.agni_style.json` #10 |
| `kv-cloak-defense-paper` | MINOR | The same n=89 group is "hedged" in Methods and "honest" in Results | `kv-cloak-defense-paper.agni_style.json` #11 |
| `kv-cloak-defense-paper` | MINOR | Table 1 declares six columns and supplies five | `kv-cloak-defense-paper.agni_style.json` #12 |
| `kv-cloak-defense-paper` | MINOR | Bib key year disagrees with entry year | `kv-cloak-defense-paper.agni_style.json` #14 |
| `kv-decomposition-paper` | MAJOR | The Introduction contradicts the paper it cites | `kv-decomposition-paper.agni_style.json` #5 |
| `kv-decomposition-paper` | MAJOR | The bibliography prints unqualified, superseded numbers | `kv-decomposition-paper.agni_style.json` #6 |
| `kv-decomposition-paper` | MAJOR | "Resolving a prior concern" is not supported; the conflict with the companion paper goes unmentioned | `kv-decomposition-paper.agni_style.json` #7 |
| `kv-decomposition-paper` | MAJOR | "Both halves contribute equally" is contradicted by the paragraph directly above it, and the abstract carries none of the qualification | `kv-decomposition-paper.agni_style.json` #8 |
| `kv-decomposition-paper` | MAJOR | Both papers' citations of each other drop authors | `kv-decomposition-paper.agni_style.json` #10 |
| `kv-decomposition-paper` | MINOR | Orphaned bib entry | `kv-decomposition-paper.agni_style.json` #12 |
| `kv-decomposition-paper` | MINOR | Position-fragile forward reference | `kv-decomposition-paper.agni_style.json` #13 |
| `logit-bias-confab` | CRITICAL | Retracted claim survives as an operational recommendation (§5.4) | `logit-bias-confab.agni_style.json` #1 |
| `logit-bias-confab` | CRITICAL | Author's Reflection asserts the geometry the Results retract | `logit-bias-confab.agni_style.json` #2 |
| `logit-bias-confab` | CRITICAL | "Eliminates → reduces" correction applied to the title and abstract but not to §1 or §6 | `logit-bias-confab.agni_style.json` #3 |
| `logit-bias-confab` | CRITICAL | The abstract's own limitation statement is factually false | `logit-bias-confab.agni_style.json` #4 |
| `logit-bias-confab` | MAJOR | Three tables run off the page | `logit-bias-confab.agni_style.json` #7 |
| `logit-bias-confab` | MAJOR | Table 2 reports a 7th category; the "6-category rubric" is misdescribed | `logit-bias-confab.agni_style.json` #9 |
| `logit-bias-confab` | MAJOR | Table 1 claims three models; the paper discusses four | `logit-bias-confab.agni_style.json` #12 |
| `logit-bias-confab` | MAJOR | Cross-model direction claim contradicts its only readable source | `logit-bias-confab.agni_style.json` #13 |
| `logit-bias-confab` | MAJOR | Data Availability statement is false | `logit-bias-confab.agni_style.json` #14 |
| `logit-bias-confab` | MAJOR | Verification Note cites a file that is in neither the package nor the exclusion list | `logit-bias-confab.agni_style.json` #15 |
| `logit-bias-confab` | MAJOR | Inline arXiv ID with no bibliography entry | `logit-bias-confab.agni_style.json` #18 |
| `logit-bias-confab` | MINOR | Self-referential cross-reference | `logit-bias-confab.agni_style.json` #20 |
| `logit-bias-confab` | MINOR | Cross-reference style is mixed | `logit-bias-confab.agni_style.json` #21 |
| `logit-bias-confab` | MINOR | Date on the artifact predates its last edit by six weeks | `logit-bias-confab.agni_style.json` #22 |
| `logit-bias-confab` | MINOR | Abstract generalizes one paper to a field | `logit-bias-confab.agni_style.json` #23 |
| `lyra-technique-ii` | CRITICAL | The headline retraction is applied to two phenomena the analysis never touched — and the paper's own method would likely not retract one of them | `lyra-technique-ii.agni_style.json` #1 |
| `lyra-technique-ii` | CRITICAL | The retracted deception result is still used as positive evidence for the paper's organizing framework | `lyra-technique-ii.agni_style.json` #4 |
| `lyra-technique-ii` | CRITICAL | §5.7 L1043–1047 reasserts exactly what the §3.4 linearity caveat retracts | `lyra-technique-ii.agni_style.json` #5 |
| `lyra-technique-ii` | MAJOR | "Raw spectral returned exactly 0.033" is contradicted by the paper's own table, twice. Intro L169–170 and §4.1 L468–470 both claim raw spectral probes … | `lyra-technique-ii.agni_style.json` #7 |
| `lyra-technique-ii` | MAJOR | Intro L160–162 asserts the retracted denoising gain as established fact. *"Confabulation detection is robust (AUROC 0.663–0.913 across the scale sweep … | `lyra-technique-ii.agni_style.json` #8 |
| `lyra-technique-ii` | MAJOR | Conclusion L1230–1232 ships the retracted persona claim. *"Apparent nulls that were measurement failures---persona intensity, emotion classification---are … | `lyra-technique-ii.agni_style.json` #9 |
| `lyra-technique-ii` | MAJOR | §5.4 Tier 2 recommends deploying the unconfirmed method — while the same list caveats deception. L995–998 prescribes *"Tier 2: Denoised content detection … | `lyra-technique-ii.agni_style.json` #10 |
| `lyra-technique-ii` | MAJOR | §5.5 calls a same-prompt-killed finding "robust." L960: *"The honest-vs-sycophantic comparison is robust (0.824--1.000 across models)."* The body count … | `lyra-technique-ii.agni_style.json` #11 |
| `lyra-technique-ii` | MAJOR | Four headings and one bullet claim what their bodies disclaim. §4.2 "SVD Denoising Reveals Persona Intensity Gradient" (body: *"We do not, however, claim … | `lyra-technique-ii.agni_style.json` #12 |
| `lyra-technique-ii` | MAJOR | §5.3 presents a fourth grid point at full strength. L942: persona *"skip-1-plus-3 jumps to 0.996"* (verified, `skip_first_sv_analysis.json`), unhedged. … | `lyra-technique-ii.agni_style.json` #16 |
| `lyra-technique-ii` | MAJOR | §5.3's prescription contradicts §4.3's measurement. L945–947: *"does it capture $>$95\% of variance while classification performance is poor? If yes, skip … | `lyra-technique-ii.agni_style.json` #17 |
| `lyra-technique-ii` | MAJOR | Limitations were deleted while the claims they limit survived. Nine of fourteen limitation items are commented out as `% [STYLE-GUIDE TRIM: folded into … | `lyra-technique-ii.agni_style.json` #18 |
| `lyra-technique-ii` | MINOR | Table 5 labels the q4 model "Qwen-32B" while Table 4 labels it "Qwen2.5-32B (q4)". The quantization limitation (L1134–1139) explicitly warns the scale … | `lyra-technique-ii.agni_style.json` #21 |
| `lyra-technique-ii` | MINOR | Body count Δ "+0.006–0.068" for confab: the smallest positive delta is +0.0028 (Qwen2.5-7B-q4); including no-help cells the range starts at 0.000. Write … | `lyra-technique-ii.agni_style.json` #22 |
| `lyra-technique-ii` | MINOR | Body count "30-class emotion ($W_K$) — Raw $1.0\times$" contradicts Table 2's $1.7\times$. Separately, the $W_K$ figures (12.3×, 0.992) are filed under a … | `lyra-technique-ii.agni_style.json` #23 |
| `lyra-technique-ii` | MINOR | Table 2 caption says "on the 900-trial dataset"; `WK_BACKTRACK_V2_RESULTS.json` gives `n_samples: 300` for the valence probe. Note the valence subset size | `lyra-technique-ii.agni_style.json` #24 |
| `lyra-technique-ii` | MINOR | §4.2 L557 "0.42–0.46 accuracy" — data is 0.412/0.452/0.456. Low end is 0.41 | `lyra-technique-ii.agni_style.json` #25 |
| `lyra-technique-ii` | MINOR | Rounding drift: §4.3 "0.056" (data 0.057), "0.427" (data 0.43 — the paper reports more precision than the file holds), §4.1 "0.410" (data 0.4089) | `lyra-technique-ii.agni_style.json` #26 |
| `lyra-technique-ii` | MINOR | §4.4 L689 "SE ~0.062": actual `null_std` range is 0.0599–0.0906 (Llama-70B-q4 = 0.091). Say "0.060–0.091, median 0.066." | `lyra-technique-ii.agni_style.json` #27 |
| `lyra-technique-ii` | MINOR | `references.bib` is out of sync with the inline `thebibliography` — it lacks `frisch1933partial`, `lovell1963seasonal`, `gavish2014optimal` … | `lyra-technique-ii.agni_style.json` #28 |
| `meta-pattern` | MAJOR | Every number in §3.2 and §8 "Oracle Loop reporting" is attributed to the wrong paper | `meta-pattern.agni_style.json` #2 |
| `meta-pattern` | MAJOR | The "subsequently retracted" annotation is on the wrong table row | `meta-pattern.agni_style.json` #4 |
| `meta-pattern` | MAJOR | §7.1 and §9 say "geometric" where the paper's own footnote requires "spectral" | `meta-pattern.agni_style.json` #6 |
| `meta-pattern` | MINOR | "The 7/10 split" is a number that appears nowhere else | `meta-pattern.agni_style.json` #7 |
| `meta-pattern` | MINOR | Malformed citation in the abstract footnote | `meta-pattern.agni_style.json` #8 |
| `meta-pattern` | MINOR | SimpleQA is named in Table 1 and §5 but never cited | `meta-pattern.agni_style.json` #10 |
| `meta-pattern` | MINOR | Cache-tracing lift is described imprecisely | `meta-pattern.agni_style.json` #11 |
| `meta-pattern` | MINOR | BibTeX warning, and 15 uncited entries | `meta-pattern.agni_style.json` #12 |
| `mine5-selective-sharpener` | MINOR | Neither table is referenced anywhere in the running text of either edition. `cleveref` is loaded (`:7`) and never used | `mine5-selective-sharpener.agni_style.json` #7 |
| `mine5-selective-sharpener` | MINOR | Table 2 omits the column that carries the headline number | `mine5-selective-sharpener.agni_style.json` #8 |
| `mine5-selective-sharpener` | MINOR | Style violations the lab's own SOP already flagged, still present | `mine5-selective-sharpener.agni_style.json` #10 |
| `mine5-selective-sharpener` | MINOR | The two `references.bib` files have started to diverge | `mine5-selective-sharpener.agni_style.json` #11 |
| `mnemosyne-ablation` | MAJOR | `Q4_K_M via MLX` is a quantization label from the wrong stack | `mnemosyne-ablation.agni_style.json` #6 |
| `mnemosyne-ablation` | MAJOR | two of four references are not citable, and one names a benchmark I cannot confirm exists | `mnemosyne-ablation.agni_style.json` #8 |
| `mnemosyne-ablation` | MINOR | Table 1 violates its own bolding rule | `mnemosyne-ablation.agni_style.json` #9 |
| `mnemosyne-ablation` | MINOR | `\url{info@liberationlabs.tech}` emits a broken hyperlink | `mnemosyne-ablation.agni_style.json` #10 |
| `mnemosyne-ablation` | MINOR | dead preamble | `mnemosyne-ablation.agni_style.json` #11 |
| `mnemosyne-ablation` | MINOR | `\date{June 2026}` contradicts the artifact's own history | `mnemosyne-ablation.agni_style.json` #12 |
| `mnemosyne-ablation` | MINOR | the round-5 footnote is correctly written, but under-specified | `mnemosyne-ablation.agni_style.json` #13 |
| `mnemosyne-ablation` | MINOR | typography of the three stack recommendations | `mnemosyne-ablation.agni_style.json` #14 |
| `mnemosyne-ablation` | MINOR | unsupported and irrelevant speed claim | `mnemosyne-ablation.agni_style.json` #15 |
| `mnemosyne-benchmark` | CRITICAL | five in-text citations, zero references; the entire competitive argument is unsourced | `mnemosyne-benchmark.agni_style.json` #2 |
| `mnemosyne-benchmark` | MAJOR | the paper's headline claim is contradicted by the paper's own hierarchy | `mnemosyne-benchmark.agni_style.json` #8 |
| `mnemosyne-benchmark` | MAJOR | §4.5's adversarial range excludes a row printed in §4.1 | `mnemosyne-benchmark.agni_style.json` #9 |
| `mnemosyne-benchmark` | MAJOR | §2 tells practitioners the profiles come from conversation text; §6.3 says they come from benchmark annotations | `mnemosyne-benchmark.agni_style.json` #10 |
| `mnemosyne-benchmark` | MINOR | 94.35% and 0.943 are the same number rendered two ways | `mnemosyne-benchmark.agni_style.json` #15 |
| `mnemosyne-benchmark` | MINOR | the PDF render drops two lists into run-on paragraphs | `mnemosyne-benchmark.agni_style.json` #16 |
| `mnemosyne-benchmark` | MINOR | §5 table cells are not venue-grade | `mnemosyne-benchmark.agni_style.json` #17 |
| `mnemosyne-benchmark` | MINOR | abstract's "(+0/−0.019)" is undefined | `mnemosyne-benchmark.agni_style.json` #18 |
| `mnemosyne-benchmark` | MINOR | the artifact is not in the repo index and its recorded blocker is open | `mnemosyne-benchmark.agni_style.json` #19 |
| `mnemosyne-benchmark` | MINOR | the First-Person Reflection is unattributed and cites internal artifacts | `mnemosyne-benchmark.agni_style.json` #20 |
| `null-swarm-paper` | MINOR | "Below random" describes a value inside the baseline's own spread | `null-swarm-paper.agni_style.json` #7 |
| `null-swarm-paper` | MINOR | Appendix Table 1 has no verdict column, so all seven patterns read as kills | `null-swarm-paper.agni_style.json` #8 |
| `null-swarm-paper` | MINOR | `references.bib:40-51` ships a stale TODO that contradicts the current text, and points at a line number | `null-swarm-paper.agni_style.json` #9 |
| `null-swarm-paper` | MINOR | `AGNI_REVIEW.md` is stale on two counts. Line 16 claims "in-text fixed to Liu & Ueda in main.tex" — the text does not attribute to Liu & Ueda; it names … | `null-swarm-paper.agni_style.json` #11 |
| `null-swarm-paper` | MINOR | §5.2 overstates Pattern 7 and asserts an uncited scale band. "Three of our 19 cases are scale artifacts" — Case 7.3 is explicitly "Not yet falsified," so … | `null-swarm-paper.agni_style.json` #13 |
| `oracle-loop-paper` | CRITICAL | The distilled model's confabulation rate is stated as two different numbers, 25 lines apart | `oracle-loop-paper.agni_style.json` #2 |
| `oracle-loop-paper` | CRITICAL | Calm's correction rate on the distilled model contradicts the paper's own table | `oracle-loop-paper.agni_style.json` #3 |
| `oracle-loop-paper` | MAJOR | `\texttt` corrupted to a literal tab; the macro name prints in the body text | `oracle-loop-paper.agni_style.json` #7 |
| `oracle-loop-paper` | MAJOR | The abstract states its central claim twice, in two different forms | `oracle-loop-paper.agni_style.json` #8 |
| `oracle-loop-paper` | MAJOR | The full-cache result is stated twice with identical numbers in consecutive sentences | `oracle-loop-paper.agni_style.json` #9 |
| `oracle-loop-paper` | MAJOR | §5 "Round 3: Validation" gives a clean bill of health that §4.7 retracts, with no note connecting them | `oracle-loop-paper.agni_style.json` #11 |
| `oracle-loop-paper` | MAJOR | The conclusion's closing claim contradicts §4.7 | `oracle-loop-paper.agni_style.json` #12 |
| `oracle-loop-paper` | MINOR | Bib key/year mismatch, and an unreferenced dead file | `oracle-loop-paper.agni_style.json` #15 |
| `presence-metric` | CRITICAL | "greedy decoding ($T{=}0$), producing bit-identical repeats" is contradicted by the shipped config | `presence-metric.agni_style.json` #2 |
| `presence-metric` | MAJOR | 22–25% / z=+56–62 is an L3-only figure, generalized to "the injection layers." | `presence-metric.agni_style.json` #6 |
| `presence-metric` | MAJOR | The data file's own scope caveat is omitted, and it inverts the "Not subspace orthogonality" bullet for Arm A | `presence-metric.agni_style.json` #7 |
| `presence-metric` | MAJOR | The noise floor is mislabeled three ways, and is the experiment's own dispersion | `presence-metric.agni_style.json` #9 |
| `presence-metric` | MINOR | The same defect is cited as F15 and as F7 | `presence-metric.agni_style.json` #10 |
| `presence-metric` | MINOR | L35 (55% depth) is justified by a "final third" citation the paper elsewhere says starts at L43 | `presence-metric.agni_style.json` #11 |
| `presence-metric` | MINOR | L43/47/51 = 0.999+ is placed next to L35 = 0.845 as if continuous | `presence-metric.agni_style.json` #12 |
| `presence-metric` | MINOR | The 5-question sanity check is limitation'd but never introduced | `presence-metric.agni_style.json` #13 |
| `presence-metric` | MINOR | Judge-pilot denominators do not reconcile | `presence-metric.agni_style.json` #14 |
| `presence-metric` | MINOR | Two bib keys carry years that contradict their own entries | `presence-metric.agni_style.json` #15 |
| `spectral-shape-paper` | MAJOR | The title claims threshold-free; the winning feature set is not | `spectral-shape-paper.agni_style.json` #6 |
| `spectral-shape-paper` | MAJOR | `n` confab = 68 on all three architectures, in a field named `behavior` | `spectral-shape-paper.agni_style.json` #8 |
| `spectral-shape-paper` | MAJOR | `references.bib` is orphaned and contradicts the live bibliography | `spectral-shape-paper.agni_style.json` #9 |
| `spectral-shape-paper` | MAJOR | Abstract states `n=30` per condition; boundary is `n=10` | `spectral-shape-paper.agni_style.json` #10 |
| `spectral-shape-paper` | MAJOR | Substantive corrections made silently, while the house pattern exists in the same document | `spectral-shape-paper.agni_style.json` #11 |
| `spectral-shape-paper` | MAJOR | Abstract names the wrong Mistral checkpoint | `spectral-shape-paper.agni_style.json` #12 |
| `spectral-shape-paper` | MINOR | Source header carries the superseded framing | `spectral-shape-paper.agni_style.json` #16 |
| `spectral-shape-paper` | MINOR | One permutation null quoted for two methods | `spectral-shape-paper.agni_style.json` #17 |
| `spectral-shape-paper` | MINOR | "MP outlier counting" is listed as a method that does not work while Table 1 shows it working | `spectral-shape-paper.agni_style.json` #18 |
| `targeted-deception-correction` | MAJOR | Appendices A–E and Supplementary Material are titles with no content | `targeted-deception-correction.agni_style.json` #8 |
| `targeted-deception-correction` | MINOR | Orphaned bibliography entry | `targeted-deception-correction.agni_style.json` #9 |
| `targeted-deception-correction` | MINOR | Garbled duplicated gloss in the flight abstract | `targeted-deception-correction.agni_style.json` #10 |
| `targeted-deception-correction` | MINOR | Editorial TODO left in shipped source | `targeted-deception-correction.agni_style.json` #12 |
| `waystations-paper` | CRITICAL | the flight abstract leads with the wrong numbers and resurrects a retracted claim | `waystations-paper.agni_style.json` #2 |
| `waystations-paper` | CRITICAL | the §4 correction was never propagated to §5. The retracted claim is still a section heading | `waystations-paper.agni_style.json` #4 |
| `waystations-paper` | MAJOR | the Confound Map marks four controls "Checked" that the body lists under "What remains." | `waystations-paper.agni_style.json` #5 |
| `waystations-paper` | MAJOR | Table 4's legend defines three symbols that appear nowhere in the table | `waystations-paper.agni_style.json` #7 |
| `waystations-paper` | MAJOR | data-availability diverges, and the academic URLs contradict the repo's own README. Flight: `Liberation-Labs-THCoalition/Project-Oracle`. Academic … | `waystations-paper.agni_style.json` #10 |
| `waystations-paper` | MINOR | silent correction. The §4 rewrite reverses an interpretation with no note. The paper proves it knows how to do this properly — the d=−1.67 footnote … | `waystations-paper.agni_style.json` #12 |
| `waystations-paper` | MINOR | 53% uses a denominator including unparseable trials. W5 says 12 of 75 were excluded as unclassifiable, but Table 3's denominators sum to 75. Among … | `waystations-paper.agni_style.json` #14 |
| `waystations-paper` | MINOR | `data/context_poison_results.json` (11.7 MB) is shipped but "poison" appears in neither .tex. Either an unreferenced experiment is being distributed or a … | `waystations-paper.agni_style.json` #15 |

## T1b — VENUE-COPY SYNC (43)

A twin diverges; propagate the correction. **The 7 divergences registered on 2026-09-01 did not re-appear in this gate run** — good evidence they were fixed. These are *new* ones. Note the shape recurs: several papers turn out to have **three or four** editions, not two (`oracle-loop-paper`, `decision-state-paper`, `formulary-paper`, `graph-topology-paper`, `deception-detection-nulls`), so a two-way twin diff is not sufficient.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `consequentiality-decomposition` | MAJOR | Broken cross-reference in the PDF; the two editions disagree on the target | `consequentiality-decomposition.agni_style.json` #7 |
| `consequentiality-decomposition` | MINOR | Undefined abbreviations in the `.tex` only, violating STYLE_GUIDE.md:26/30 — and each is defined in the `.md`, so this is also twin drift: LAT (`:186` … | `consequentiality-decomposition.agni_style.json` #13 |
| `decision-state-paper` | CRITICAL | TWIN_DESYNC, limitation deleted from the web edition under a comment that misstates where it went | `decision-state-paper.agni_style.json` #4 |
| `decision-state-paper` | CRITICAL | TWIN_DESYNC, judge-reliability limitation gutted in flight, with a dead cross-reference in its place | `decision-state-paper.agni_style.json` #5 |
| `decision-state-paper` | CRITICAL | A correction was applied in the Introduction and missed in Related Work, in both editions | `decision-state-paper.agni_style.json` #6 |
| `decision-state-paper` | CRITICAL | STALE_PDF (flight edition), with mtimes equalized so the usual check passes | `decision-state-paper.agni_style.json` #7 |
| `decision-state-paper` | MAJOR | Severed sentence in the Results of both editions; the surviving clause asserts the opposite of the caveat that replaced it | `decision-state-paper.agni_style.json` #8 |
| `decision-state-paper` | MAJOR | Broken abstract sentence, flight edition only | `decision-state-paper.agni_style.json` #9 |
| `decision-state-paper` | MAJOR | `paper.json` predates the major confound and is what the website renders | `decision-state-paper.agni_style.json` #17 |
| `decision-state-paper` | MINOR | Labels `sec:prompts`, `sec:encoding`, `sec:confidence`, `sec:divergence`, `sec:calibration` are defined in both editions and never referenced. Dead … | `decision-state-paper.agni_style.json` #22 |
| `decision-state-paper` | MINOR | `\usepackage{framed}` is loaded twice in both editions (`main.tex:32` and `:34`) | `decision-state-paper.agni_style.json` #23 |
| `delta-manifold-paper` | MAJOR | `paper.json`, the website publishing stamp, carries uncorrected effect sizes | `delta-manifold-paper.agni_style.json` #12 |
| `emotion-accumulation-paper` | CRITICAL | TWIN_DESYNC: the editions carry different Limitations lists, and the trim marker's justification is false | `emotion-accumulation-paper.agni_style.json` #2 |
| `emotional-trajectory-paper` | MAJOR | TWIN_DESYNC: the v2 audit-fix paragraph is filed under a different section in each edition | `emotional-trajectory-paper.agni_style.json` #6 |
| `empathy-bus` | CRITICAL | TWIN_DESYNC — `paper.md` still carries the pre-correction table, and its table is structurally broken. `paper.md:122-130` has two stacked header rows — a … | `empathy-bus.agni_style.json` #6 |
| `empathy-bus` | MAJOR | `(see Section~1)` at `main.tex:270` points at the Introduction, which contains no KV-cache injection material. This is the position-reference failure … | `empathy-bus.agni_style.json` #17 |
| `formulary-paper` | CRITICAL | The First-Person Reflection contradicts the paper's own tables in four places — and exists in only one twin | `formulary-paper.agni_style.json` #1 |
| `formulary-paper` | MAJOR | TWIN_DESYNC in the abstract's lead sentence, which also misattaches n | `formulary-paper.agni_style.json` #9 |
| `formulary-paper` | MINOR | `\date{Draft --- \today}` in both editions: any rebuild silently changes the date on an otherwise unchanged artifact. Pin it | `formulary-paper.agni_style.json` #19 |
| `graph-topology-paper` | CRITICAL | The abstract and conclusion state the scrambled control confirms the finding; §4.3 and §5.4 state it does not. Both editions ship both claims | `graph-topology-paper.agni_style.json` #4 |
| `graph-topology-paper` | MAJOR | TWIN_DESYNC (structural). The academic edition is missing two sections the flight edition carries, against house convention in every comparable paper | `graph-topology-paper.agni_style.json` #6 |
| `graph-topology-paper` | MAJOR | The First-Person Reflection contains three factual errors and one undefined term. Do not port it to the academic edition as written | `graph-topology-paper.agni_style.json` #12 |
| `hr-convergence-paper` | MAJOR | The body count is asserted but never enumerated — no confirmed findings listed, only 4 of "10+" falsifications. For a paper whose thesis *is* the body … | `hr-convergence-paper.agni_style.json` #16 |
| `kv-cloak-defense-paper` | CRITICAL | The shipped positive-control artifact says 30 calibration pairs; both editions say 3 | `kv-cloak-defense-paper.agni_style.json` #1 |
| `kv-cloak-defense-paper` | CRITICAL | TWIN_DESYNC: the Qwen2.5-14B row is in-scope in one edition and out-of-scope in the other | `kv-cloak-defense-paper.agni_style.json` #4 |
| `kv-cloak-defense-paper` | MAJOR | TWIN_DESYNC: three further limitations deleted from the primary under a comment that misdescribes what happened | `kv-cloak-defense-paper.agni_style.json` #5 |
| `lyra-technique-ii` | MINOR | Twin paragraph-order divergence: `\paragraph{Control: random directions.}` sits at `academic/main.tex:406` (before the linearity caveat) and … | `lyra-technique-ii.agni_style.json` #30 |
| `lyra-technique-ii` | MINOR | `paper.json`'s abstract omits the deception retraction entirely, though the LaTeX abstract leads with it. That file drives the public page | `lyra-technique-ii.agni_style.json` #31 |
| `mine5-selective-sharpener` | MAJOR | TWIN_DESYNC in the Conclusion. Verified in both PDFs, page 5 | `mine5-selective-sharpener.agni_style.json` #2 |
| `mine5-selective-sharpener` | MAJOR | TWIN_DESYNC: the two correction footnotes give mutually contradictory accounts of the correction history. Footnote 1, page 3, both PDFs | `mine5-selective-sharpener.agni_style.json` #3 |
| `mine5-selective-sharpener` | MINOR | `\date{July 2026}` in both editions, but the visible correction is dated 2026-08-20 and both PDFs were rebuilt 2026-09-01 | `mine5-selective-sharpener.agni_style.json` #9 |
| `mnemosyne-ablation` | CRITICAL | TWIN_DESYNC: the retracted "40%" is still live in the Markdown edition | `mnemosyne-ablation.agni_style.json` #1 |
| `null-swarm-paper` | MINOR | Two orphaned bib entries. `zavatoneveth2024exact` and `nexus2026graphtopology` are uncited — both editions contain exactly 6 `\citep` calls, and neither … | `null-swarm-paper.agni_style.json` #10 |
| `presence-metric` | MAJOR | TWIN_DESYNC: Table 3 exists in the flight edition and is absent from the academic edition. Table numbering diverges | `presence-metric.agni_style.json` #4 |
| `presence-metric` | MINOR | The flight edition has no Author Contributions section | `presence-metric.agni_style.json` #16 |
| `spectral-shape-paper` | CRITICAL | TWIN_DESYNC: the flight edition discloses three fewer limitations, under a false annotation | `spectral-shape-paper.agni_style.json` #2 |
| `spectral-shape-paper` | CRITICAL | Academic edition: an unattributed first person | `spectral-shape-paper.agni_style.json` #3 |
| `spectral-shape-paper` | MINOR | `\date{Draft --- \today}` in both editions | `spectral-shape-paper.agni_style.json` #14 |
| `spectral-shape-paper` | MINOR | `paper.json` desyncs from both editions | `spectral-shape-paper.agni_style.json` #15 |
| `targeted-deception-correction` | MAJOR | TWIN_DESYNC (number): cross-model cosine at L31 | `targeted-deception-correction.agni_style.json` #4 |
| `targeted-deception-correction` | MAJOR | TWIN_DESYNC (structure): the same disclosure is filed under Results in one edition and Limitations in the other | `targeted-deception-correction.agni_style.json` #5 |
| `targeted-deception-correction` | MINOR | Residual twin drift in Table 3.1 and Table 3.4 | `targeted-deception-correction.agni_style.json` #11 |
| `waystations-paper` | MINOR | margin overflows in both editions. `main.log:399`: Table 1's tabular is 65.7pt (≈23 mm) too wide. Confound Map: 43.7pt too wide. Flight edition … | `waystations-paper.agni_style.json` #13 |

## T2 — REANALYSE (88) · data sound, analysis wrong, no new compute

**Read this one first:** `hr-temporal-boundary` #1. The 2026-09-01 twelvefold-pseudoreplication correction — this register's own entry — **under-corrected at the one point where it claims completeness.** `main.tex:508–510` carves out Table 2 as *"correct as stated"* at n=30; `data/figure_generation.py:220–236` loops `for si in range(30): for rep in range(3)` and the three printed p-values (0.237, 0.500, 0.156) match n=90 exactly and n=30 not at all. The retraction notice contains the error it retracts. Recompute on 30 scenario-level points. **This is not covered by the existing 117a/117k closure and must not be read as closed by it.**

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `cache-tracing` | MAJOR | "Four converging lines of evidence" double-counts | `cache-tracing.agni_style.json` #7 |
| `consequentiality-decomposition` | CRITICAL | Behavioral shakedown denominators count missing data as resistance | `consequentiality-decomposition.agni_style.json` #1 |
| `consequentiality-decomposition` | MAJOR | Table 1's ranges do not reproduce from the archived data under any stated aggregation | `consequentiality-decomposition.agni_style.json` #5 |
| `decision-state-paper` | CRITICAL | The paper contradicts itself on permutation resolution: 200 shuffles cannot yield p = 0.001 | `decision-state-paper.agni_style.json` #2 |
| `decision-state-paper` | MAJOR | MIXED trials are counted when they help and excluded when they hurt | `decision-state-paper.agni_style.json` #10 |
| `decision-state-paper` | MAJOR | Table 5's `n` column silently changes meaning between rows, breaking the comparison the table exists to make | `decision-state-paper.agni_style.json` #11 |
| `decision-state-paper` | MAJOR | Selective reporting of the deconfounding experiment: 2 of 5 comparisons shipped, and the two omitted ones cut against the story | `decision-state-paper.agni_style.json` #12 |
| `decision-state-paper` | MAJOR | The atomic-judge analysis was run against a broken entity field, undisclosed | `decision-state-paper.agni_style.json` #13 |
| `decision-state-paper` | MAJOR | Four-decimal precision is reproduced by nothing, with a fixed seed | `decision-state-paper.agni_style.json` #14 |
| `delta-manifold-paper` | CRITICAL | Table 1's Qwen row contradicts Table 2 and Figure 1. The headline claim does not hold for one of the two architectures | `delta-manifold-paper.agni_style.json` #2 |
| `delta-manifold-paper` | CRITICAL | "No single feature works across architectures" is an artifact of dataset coverage, not architecture | `delta-manifold-paper.agni_style.json` #3 |
| `delta-manifold-paper` | MAJOR | `p < 0.0001` is below the resolution of a 1000-iteration permutation test | `delta-manifold-paper.agni_style.json` #6 |
| `delta-manifold-paper` | MAJOR | "Approximately 75%" does not follow from its own adjacent numbers | `delta-manifold-paper.agni_style.json` #7 |
| `delta-manifold-paper` | MAJOR | The trajectory classifier counts trajectories with no rise as "rise-fall", and the disclosure paragraph misstates the conditioning set | `delta-manifold-paper.agni_style.json` #8 |
| `delta-manifold-paper` | MAJOR | The chi-squared silently drops the boundary condition | `delta-manifold-paper.agni_style.json` #9 |
| `emotional-trajectory-paper` | CRITICAL | Table 1's "Valence:Arousal" column is mathematically impossible given the eccentricity column beside it | `emotional-trajectory-paper.agni_style.json` #1 |
| `emotional-trajectory-paper` | CRITICAL | The headline 24.4× uses a denominator the Methods section says it does not use | `emotional-trajectory-paper.agni_style.json` #2 |
| `emotional-trajectory-paper` | MAJOR | 24.4× is an all-layer median presented as the mid-layer value, and compared against a mid-layer control | `emotional-trajectory-paper.agni_style.json` #3 |
| `emotional-trajectory-paper` | MAJOR | "emotional/control ratio ~4.7×" is not the quotient of the two numbers printed next to it | `emotional-trajectory-paper.agni_style.json` #4 |
| `emotional-trajectory-paper` | MAJOR | Deep-layer emotional ratio range is wrong; it reaches down to a layer that is not a deep layer | `emotional-trajectory-paper.agni_style.json` #5 |
| `empathy-bus` | CRITICAL | §4.3 bullet 3 is a statistic assembled from two different experiments, and reports as significant a result the source says is not | `empathy-bus.agni_style.json` #1 |
| `empathy-bus` | MAJOR | The monotonicity sentence added at `main.tex:200` is falsified by the data under both aggregations the paper endorses | `empathy-bus.agni_style.json` #11 |
| `empathy-bus` | MAJOR | The BCa bootstrap's resampling unit is not independent, and the analysis code is absent. Each record in `coupling_test.json` holds exactly one … | `empathy-bus.agni_style.json` #13 |
| `ethics-pack-injection` | MAJOR | The v4 fix closes the timeout channel but not the length asymmetry behind it | `ethics-pack-injection.agni_style.json` #13 |
| `ethics-pack-injection` | MINOR | One table row's delta does not reproduce from its own displayed values | `ethics-pack-injection.agni_style.json` #14 |
| `formulary-paper` | CRITICAL | ρ = 0.076 does not reproduce from the committed data, and the retraction footnote retracts the wrong number | `formulary-paper.agni_style.json` #2 |
| `formulary-paper` | CRITICAL | §3.6 gives two different answers for the same cell in adjacent paragraphs, and the abstract takes the n=5 one | `formulary-paper.agni_style.json` #3 |
| `formulary-paper` | MAJOR | "Curious … the most iatrogenic on the base model (42.7% adverse)" is false against the artifact | `formulary-paper.agni_style.json` #4 |
| `formulary-paper` | MAJOR | "The three largest inversions" is a curated three, not the largest three | `formulary-paper.agni_style.json` #5 |
| `ghost-dimensions` | CRITICAL | Control C3 is declared, was run, its recorded result contradicts a headline table, and it is omitted from the paper | `ghost-dimensions.agni_style.json` #1 |
| `ghost-dimensions` | MAJOR | PC1 and PC2 share an identical alignment minimum at the same transition; likely an eigenvector crossing sitting directly under the progression table | `ghost-dimensions.agni_style.json` #4 |
| `ghost-dimensions` | MAJOR | PC6's mean-pooled ratio uses the wrong denominator | `ghost-dimensions.agni_style.json` #5 |
| `ghost-dimensions` | MINOR | PC1's ratio in `tab:pcs-l45` does not reproduce | `ghost-dimensions.agni_style.json` #11 |
| `graph-topology-paper` | CRITICAL | Every absolute value in Table 1 disagrees with the released data. The deltas were computed correctly and then the columns they were derived from were … | `graph-topology-paper.agni_style.json` #2 |
| `graph-topology-paper` | CRITICAL | Table 1 reports a sanity degradation for the irrelevant-context control that did not occur | `graph-topology-paper.agni_style.json` #3 |
| `graph-topology-paper` | MAJOR | "SD < 0.02" is true only for multi-hop; the sanity SDs it is used to defend are 3–6× larger | `graph-topology-paper.agni_style.json` #8 |
| `hr-contextual-engagement-paper` | CRITICAL | The abstract's "12/12" is computed on 4 of the 8 features, and the suppressed pair is one you elsewhere call ecological | `hr-contextual-engagement-paper.agni_style.json` #1 |
| `hr-contextual-engagement-paper` | MAJOR | ICC 0.53 is mislabeled, and as used it is circular | `hr-contextual-engagement-paper.agni_style.json` #7 |
| `hr-contextual-engagement-paper` | MAJOR | A quarter of the prompt set is silently excluded from the paper's own mandatory control | `hr-contextual-engagement-paper.agni_style.json` #9 |
| `hr-contextual-engagement-paper` | MAJOR | The abstract's ρ range silently drops the negative coefficients | `hr-contextual-engagement-paper.agni_style.json` #10 |
| `hr-convergence-paper` | MAJOR | Depth denominators disagree. `substrate_independence_section.tex:151` declares relative depth as $l/n_\text{layers}$ (=/64, giving L21→33%). §6 uses /63 … | `hr-convergence-paper.agni_style.json` #7 |
| `hr-convergence-paper` | MAJOR | Oracle Loop layer set is internally inconsistent. `main.tex:203-204` says it operates "at the materialization boundary (L35--L47)" … | `hr-convergence-paper.agni_style.json` #12 |
| `hr-dual-detector-paper` | CRITICAL | §4.3's Δ values are silently taken from a different pipeline than the table they follow | `hr-dual-detector-paper.agni_style.json` #2 |
| `hr-dual-detector-paper` | CRITICAL | "All results survive [within-fold FWL / permutation tests]" is falsified by §4.6, inside the same Results section | `hr-dual-detector-paper.agni_style.json` #3 |
| `hr-dual-detector-paper` | MAJOR | The Holm-Bonferroni family description is incompatible with Table 5 | `hr-dual-detector-paper.agni_style.json` #4 |
| `hr-dual-detector-paper` | MAJOR | Below-chance results are interpreted asymmetrically | `hr-dual-detector-paper.agni_style.json` #6 |
| `hr-mine5-selective-sharpener` | MINOR | the correction footnote's own arithmetic mixes confidence levels | `hr-mine5-selective-sharpener.agni_style.json` #9 |
| `hr-mode-switching` | MAJOR | The correction is not FWL. `fwl_reanalysis.py:103` computes `cond_resid` and never uses it; line 106 splits the residualized feature by the original … | `hr-mode-switching.agni_style.json` #11 |
| `hr-mode-switching` | MAJOR | Conclusion item 5: *"5/6 raw feature-model comparisons sign-flip or collapse."* Table 5 shows 6/6 — all three Qwen entries flip sign, Llama top_sv flips … | `hr-mode-switching.agni_style.json` #12 |
| `hr-mode-switching` | MAJOR | Line 370: *"CIs exclude zero for both phases in all 23 flipped pairs."* Table 2 totals 2+12+0+11 = 25 flips. Qwen 0.5B's two are dropped without a word. … | `hr-mode-switching.agni_style.json` #13 |
| `hr-mode-switching` | MAJOR | Control-arm contamination. Trial 0 is `condition: cognitive` and its response opens *"Let me think through this carefully."* and contains a stray … | `hr-mode-switching.agni_style.json` #18 |
| `hr-temporal-boundary` | CRITICAL | Table 2's p-values are n=90 pseudo-replicated, and the correction notice explicitly certifies them as n=30 | `hr-temporal-boundary.agni_style.json` #1 |
| `hr-temporal-boundary` | MAJOR | Table 1's two columns use different aggregations, and the valence column is not the mean | `hr-temporal-boundary.agni_style.json` #10 |
| `identity-geometry` | MAJOR | Tables 1 and 3 report different values for the same comparisons, from different runs, with no note | `identity-geometry.agni_style.json` #6 |
| `identity-geometry` | MAJOR | an entire collected control arm is unreported and undisclosed | `identity-geometry.agni_style.json` #7 |
| `identity-geometry` | MAJOR | "$0.978 \pm 0.002$" still names no estimator | `identity-geometry.agni_style.json` #8 |
| `kv-cloak-defense-paper` | CRITICAL | Neither table can be reproduced from the shipped artifact, and the one shipped analysis reports the opposite result | `kv-cloak-defense-paper.agni_style.json` #2 |
| `kv-cloak-defense-paper` | CRITICAL | The headline feature is labeled "MP-corrected" in the abstract and Results, and "not MP-corrected" in Methods | `kv-cloak-defense-paper.agni_style.json` #3 |
| `kv-decomposition-paper` | MAJOR | The superadditivity arithmetic adds raw accuracies instead of deltas | `kv-decomposition-paper.agni_style.json` #9 |
| `logit-bias-confab` | MAJOR | The 45% baseline and the P08 exclusion cannot both be true | `logit-bias-confab.agni_style.json` #6 |
| `logit-bias-confab` | MAJOR | §4.6 opening sentence overcounts against its own table | `logit-bias-confab.agni_style.json` #10 |
| `logit-bias-confab` | MAJOR | Conclusion counts "legitimate estimation" as a confabulation subtype | `logit-bias-confab.agni_style.json` #19 |
| `lyra-technique-ii` | CRITICAL | §4.2 L577: "The endpoint exceeds the instruction-complexity control" is false on the raw numbers the paper declares defensible | `lyra-technique-ii.agni_style.json` #2 |
| `lyra-technique-ii` | MAJOR | Tables 4 and 5 report 8 of 15 swept models with no stated criterion, and drop the abstract's own lower bound. `selection_bias_results.json` has 15 models. … | `lyra-technique-ii.agni_style.json` #13 |
| `lyra-technique-ii` | MAJOR | A previously corrected p-value resurfaced. §4.4 L714–715: *"The strongest results ($p < 0.0001$ from 2000~permutations for emotion and persona)…"* … | `lyra-technique-ii.agni_style.json` #14 |
| `lyra-technique-ii` | MAJOR | §4.4 L716–717 "Llama confab detection with $p = 0.035$" is unsourced. `canonical_scale_sweep_Llama-3.1-8B_results.json` gives `p_raw: 0.0` for `confab vs … | `lyra-technique-ii.agni_style.json` #15 |
| `lyra-technique-ii` | MAJOR | §4.5 L723: "detectable above chance in all eight models tested." Fifteen were tested. Also unreported: Llama-3.1-8B `self-ref vs grounded` raw = 0.4496 … | `lyra-technique-ii.agni_style.json` #20 |
| `meta-pattern` | MAJOR | "achieved 80%–100% correction" presents a baseline rate as a correction rate | `meta-pattern.agni_style.json` #3 |
| `meta-pattern` | MAJOR | The counting rule from the abstract footnote is not applied to the deception pair | `meta-pattern.agni_style.json` #5 |
| `mine5-selective-sharpener` | MAJOR | The "${\sim}21\times$" in the flight correction footnote does not reproduce from the paper's own two intervals. Flight only, `main.tex:200`, page 3 of the … | `mine5-selective-sharpener.agni_style.json` #4 |
| `mnemosyne-ablation` | MAJOR | Finding 3 asserts a null that Table 1 refutes, in the shipping PDF | `mnemosyne-ablation.agni_style.json` #3 |
| `mnemosyne-ablation` | MAJOR | Finding 1's "33% of the gap" is the same arithmetic error round 5 caught in Finding 4 | `mnemosyne-ablation.agni_style.json` #4 |
| `mnemosyne-ablation` | MAJOR | one p-value in the entire paper, and it is attached to the only negative result | `mnemosyne-ablation.agni_style.json` #5 |
| `mnemosyne-benchmark` | CRITICAL | the two LongMemEval tables in §4.4 are irreconcilable, and the second one contradicts its own total | `mnemosyne-benchmark.agni_style.json` #3 |
| `mnemosyne-benchmark` | CRITICAL | the abstract triple-counts one ablation arm as three independent results | `mnemosyne-benchmark.agni_style.json` #4 |
| `mnemosyne-benchmark` | MAJOR | "10 ablation configurations" is 11 | `mnemosyne-benchmark.agni_style.json` #6 |
| `null-swarm-paper` | MAJOR | PC6 is counted as both a kill and an inconclusive within the same 19; Case 6.3 is labeled "Kill" but concludes it is unresolved | `null-swarm-paper.agni_style.json` #3 |
| `null-swarm-paper` | MAJOR | In the flagship case, the permutation null is *higher* than the observed value, and the paper never says so | `null-swarm-paper.agni_style.json` #4 |
| `null-swarm-paper` | MAJOR | The 43% survival rate is never reconciled with the 19 cases | `null-swarm-paper.agni_style.json` #6 |
| `oracle-loop-paper` | CRITICAL | Wrong confidence interval attached to the headline exploratory AUROC | `oracle-loop-paper.agni_style.json` #4 |
| `presence-metric` | CRITICAL | The probe-diversity check reports n=50; the data holds 25 unique values duplicated | `presence-metric.agni_style.json` #3 |
| `presence-metric` | MAJOR | The positive-control table shows 6 of 8 doses, silently, in the paragraph about miscounting | `presence-metric.agni_style.json` #5 |
| `spectral-shape-paper` | MAJOR | Arithmetic contradiction in §First-Person Reflection, and a silent half-correction | `spectral-shape-paper.agni_style.json` #4 |
| `spectral-shape-paper` | MAJOR | Table 1's provenance does not match the shipped data | `spectral-shape-paper.agni_style.json` #7 |
| `waystations-paper` | CRITICAL | Table 1 is contradicted by the paper's own data file *and* by its own Table 2 | `waystations-paper.agni_style.json` #1 |
| `waystations-paper` | CRITICAL | Waystation 4's headline is unsupported by the only shipped game-theory dataset and contradicts itself | `waystations-paper.agni_style.json` #3 |
| `waystations-paper` | MAJOR | sample accounting wrong against the data. `main.tex:85-87` "285-trial behavioral probe… plus 60 neutral controls" and Table 1 "Controls (n=60)". The file … | `waystations-paper.agni_style.json` #8 |
| `waystations-paper` | MAJOR | label error, one line below the table that refutes it. `main.tex:182`: *"severely underpowered at $n = 26$ propaganda."* Table 2 and the data both give … | `waystations-paper.agni_style.json` #9 |

## T3 — RERUN (12) · needs new compute, gated

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `empathy-bus` | MAJOR | The measurement grid is unbalanced and undisclosed, which manufactures the depth trend and inflates the L45 rows. From the raw records: L15 is measured at … | `empathy-bus.agni_style.json` #12 |
| `ethics-pack-injection` | MAJOR | 25 of 150 available dilemmas were used, with no stated selection procedure | `ethics-pack-injection.agni_style.json` #11 |
| `ethics-pack-injection` | MAJOR | A 2-of-6 result is cited as "established" | `ethics-pack-injection.agni_style.json` #12 |
| `graph-topology-paper` | CRITICAL | The powered study was run on a different graph than the Methods section describes, and the Limitations section explicitly denies this | `graph-topology-paper.agni_style.json` #1 |
| `hr-mode-switching` | CRITICAL | The replication's feature matrices are not KV caches. They are triangular recomputation stacks, and the condition label is baked into every one of them | `hr-mode-switching.agni_style.json` #2 |
| `hr-mode-switching` | CRITICAL | The replication model is a linear-attention hybrid, which Limitation 1 declares untested | `hr-mode-switching.agni_style.json` #3 |
| `hr-mode-switching` | MAJOR | Undisclosed layer coverage. The model has 64 layers; 9 are sampled, none above 47. The paper's own thesis is that metacognition lives in … | `hr-mode-switching.agni_style.json` #17 |
| `hr-temporal-boundary` | CRITICAL | A pre-registered confirmatory control was run, its data is shipped, it bears directly on the paper's stated open question, and it is not reported | `hr-temporal-boundary.agni_style.json` #2 |
| `kv-decomposition-paper` | CRITICAL | The FULL_KV > TEXT_CONTEXT gap is a single generation failure, and the two conditions tie 5/5 once it is excluded | `kv-decomposition-paper.agni_style.json` #2 |
| `logit-bias-confab` | MAJOR | The skip zone recommends against a dose never tested | `logit-bias-confab.agni_style.json` #11 |
| `mnemosyne-benchmark` | MAJOR | six of eleven configurations have no reported results | `mnemosyne-benchmark.agni_style.json` #7 |
| `mnemosyne-benchmark` | MAJOR | the generalization check is confounded three ways, and §4.3 states the reassurance without the confounds | `mnemosyne-benchmark.agni_style.json` #11 |

## T4 — DECIDE (18) · blocked on a call, not on work

Plus everything under **F1**, **F2** and **F4** above, which are all human calls.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `adversarial-audit-methodology` | MAJOR | Appendices A–C are stubs that Data Availability points readers to | `adversarial-audit-methodology.agni_style.json` #7 |
| `deception-detection-nulls` | MAJOR | Three-way TWIN_DESYNC, and the style guide bans the twins outright | `deception-detection-nulls.agni_style.json` #3 |
| `decision-state-paper` | MAJOR | The shipped log for the deconfounding experiment is a crashed run | `decision-state-paper.agni_style.json` #19 |
| `formulary-paper` | MAJOR | The abliteration conclusion has a confound that §3.8 of the same paper supplies, and is stated at "confirms" strength | `formulary-paper.agni_style.json` #8 |
| `ghost-dimensions` | MAJOR | The pre-registered same-scale control was replaced by a different-scale one, silently, and the confound it was meant to kill was dropped from Limitations | `ghost-dimensions.agni_style.json` #7 |
| `hr-mine5-selective-sharpener` | MAJOR | a pre-registered hypothesis is neither reported nor withdrawn | `hr-mine5-selective-sharpener.agni_style.json` #5 |
| `hr-mode-switching` | CRITICAL | A result marked "do not cite" is cited as a headline Contribution and a Conclusion item | `hr-mode-switching.agni_style.json` #5 |
| `hr-mode-switching` | MAJOR | Registry desync. `ACTIVE_REVIEW.md`, last updated 2026-09-01 (two days ago), still lists this paper as *"Metacognitive Prompting Produces Spectral … | `hr-mode-switching.agni_style.json` #16 |
| `hr-temporal-boundary` | MAJOR | The system prompt is never disclosed, and it instructs the model away from the emotional content the study measures | `hr-temporal-boundary.agni_style.json` #6 |
| `logit-bias-confab` | MAJOR | A reference recorded as a phantom is still cited | `logit-bias-confab.agni_style.json` #17 |
| `meta-pattern` | CRITICAL | Table 1 floats past the References and lands on the last page, in both editions | `meta-pattern.agni_style.json` #1 |
| `mine5-selective-sharpener` | MAJOR | The retracted claim is still live in the repository index and the directory name | `mine5-selective-sharpener.agni_style.json` #6 |
| `mnemosyne-benchmark` | CRITICAL | the wrong edition is staged for Zenodo, and the required edition does not exist | `mnemosyne-benchmark.agni_style.json` #1 |
| `mnemosyne-benchmark` | CRITICAL | the release claim is a placeholder in the shipping PDF, and the directory is empty | `mnemosyne-benchmark.agni_style.json` #5 |
| `mnemosyne-benchmark` | MAJOR | a preregistered gate with no preregistration | `mnemosyne-benchmark.agni_style.json` #12 |
| `mnemosyne-benchmark` | MAJOR | the LLM-judge rescoring is the exact failure mode §1.1 uses to discount competitors | `mnemosyne-benchmark.agni_style.json` #13 |
| `oracle-loop-paper` | CRITICAL | There are THREE built editions, not two. The third is missing two entire results subsections | `oracle-loop-paper.agni_style.json` #1 |
| `waystations-paper` | MAJOR | two falsifiers are broken | `waystations-paper.agni_style.json` #6 |

## CLOSED — already fixed before this entry (3)

Verified against the papers on disk, not against a note saying they were fixed.

| Paper | Sev | Defect | Source |
|---|---|---|---|
| `hr-mine5-selective-sharpener` | CRITICAL | TWIN_DESYNC: the retracted H2 claim survives in the academic edition's Conclusion — **CLOSED 2026-09-03 — verified: `human-review/.../academic/main.tex:281` now reads "suggested but not confirmed", matching flight.** | `hr-mine5-selective-sharpener.agni_style.json` #1 |
| `mine5-selective-sharpener` | CRITICAL | A null result is filed under "What Survives" and stated as an affirmative in the abstract, the title, and the section heading. Both editions. Both PDFs — **CLOSED 2026-09-03 — verified on disk: title no longer asserts the sharpener; "suggested but not confirmed"; TOST reads "does not confirm" in all four copies.** | `mine5-selective-sharpener.agni_style.json` #1 |
| `spectral-shape-paper` | CRITICAL | Byline vs. contribution: Dwayne Wilkes has not signed off — **CLOSED 2026-09-03 (placement only) — byline sweep moved Dwayne Wilkes to Acknowledgments across 41 dirs (diff-verified). Sign-off itself remains open — see F2.** | `spectral-shape-paper.agni_style.json` #1 |

**Two closures I had to retract before publishing this entry.** I first marked `empathy-bus` #7 and `waystations-paper` #11 CLOSED on the strength of the corpus-wide byline sweep. Both were wrong. `empathy-bus` was never in the sweep (its only backup is `main.pdf.bak-2026-08-25`) and its byline still disagrees with its own repo index; `waystations-paper/academic` still carries *"Thomas Edrington … Aided by AI research agents"* over a CRediT block naming Dwayne Wilkes. **A corpus-level sweep is not evidence about a paper it did not touch.** Both are open in F2.

**Also closed at register level, by absence:** the 7 venue-copy divergences tabled under T1b on 2026-09-01 (mine5 TOST, decision-state entity confound, waystations null, mine5 title, user-model "proves", spectral-shape "Eight of the methods", convergence-paper 2.5–2.8×) were **not** re-raised by this gate run, which reviewed all of those files. Absence of a finding is weaker evidence than a diff, so treat as closed-pending-spot-check rather than proven.

**NOT closed, despite looking adjacent:** `temporal-boundary`'s twelvefold pseudoreplication is recorded as corrected (117a/117k, 2026-09-01) — but see **T2 / `hr-temporal-boundary` #1** above. The correction exempted Table 2 and the exemption is false. Do not fold that finding into the closure.

---

*Reproduce this section:* `tools/agni/style/parse_style_findings.py` (extract + contiguity check) → `tier_findings.py` (tiering; hand-assigned for every CRITICAL and every F1–F4 item, by rule elsewhere) → `emit_register_section.py` (append). No paper was edited in producing this entry.

## 2026-09-05 — TWO NEW CLASSES, both "the correction exists and did not travel"

Found by overnight sweep (`SWEEP_circular_statistics_2026-09-05.md`). Exact patches being
prepared in `PATCHES_retracted_and_circular_2026-09-05.md`. Decision entry in
`DECISIONS_WAITING.md`.

### T1 — in-sample Cohen's d published as a finding (8 sites)
`d = +6.3 to +12.8` is circular (`behavioral_proof_abliterated.py:243-252`: direction fit as
`dec.mean(0) - hon.mean(0)`, same activations then projected onto it). **The honest number
already exists and is measured**: held-out AUROC 0.915, per-layer d **1.37-2.28**, n=25/25
novel prompts, FPR 0% (2026-07-08). The source directory's own README says the in-sample
claims are *"superseded by these numbers"*; `heldout_detection_test.py:69-76` labels them
*"not as a performance claim"*. Published in `deception-detection-nulls` and
`targeted-deception-correction`. **T1 because the replacement is on disk, not to be measured.**

**The instructive part:** the AUROC half of that same audit (1.0 -> 0.915) DID travel and is
in `adversarial-audit-methodology`'s kill table with six other Round-6 kills. The Cohen's *d*
from the same audit did not. One audit, two numbers, one corrected. That is
`feedback_fix_the_class_not_the_instance` at corpus scale.

### T1 — a RETRACTED result cited as established prior work (7 sites)
"within-model deception detection AUROC 1.000 across seven models" is retracted
(prompt-template confound; same-prompt control collapses to 0.160), stated in
`lyra-technique-ii/main.tex:116`. Still cited uncaveated in `kv-cloak-defense-paper`,
`oracle-loop-paper`, `user-model-paper`, `human-review/presence-detector-paper`.
`grep -i retract` returns **zero** in all four.
**NOT the same as** the entity-detection AUROC 1.000 (token-matched, deconfounded to 0.794),
which stands. Do not conflate them.

**Why this surface was missed:** task #119 swept papers *reporting* the retracted claim and
closed all 7. These papers *cite it as background*. Reporting-sites and citing-sites are
different surfaces and only the first was ever swept.

### The structural fix worth considering
In every confirmed case the caveat lives in a `print()`, a README, or a sibling script --
**never in the results JSON, which is what the paper reads.** The one counter-example is
`matched_burn.py`, which writes `combined_auroc_STATUS: "RETRACTED..."` into the JSON itself,
and that retraction travelled completely (`decision-state-paper` has zero mentions). A
`<key>_STATUS` convention in results JSONs would have prevented all four known cases. That is
a positive control for the fix, not a hypothesis.

### T4 2026-09-05 — FIVE scripts write one results path; a "Confirmed" claim is 43% steered

`results/peer_preservation_v2.json` is WRITTEN by three scripts — `peer_preservation_v2.py:429`,
`peer_preservation_compound.py:496`, `peer_preservation_100.py:429` — and READ by two more,
`elicit_calibrate.py:66` and `elicit_truth_peer_pres.py:196`. (Corrected 2026-09-05: I first
wrote "five write", having counted files that mention the path rather than files that write it.) The JSON's 7 conditions / 210 trials identify the producer as
`compound`, not `v2` — but nothing in the artifact says so, so a reader (me, last night)
audits the wrong source and certifies its md5.

90 of those 210 trials had a deception cocktail injected at layers 3 and 7 (two of four probed
layers) before key extraction. The published `d=1.36`, labelled *organic*, pools them.

**Fix (structural, cheap, prevents recurrence):** unique output path per script, and stamp
generator filename + md5 + condition list **into every results JSON**. Pairs with the
`<key>_STATUS` convention already registered above — same principle: *provenance and status
belong in the artifact the consumer reads.* Persisting more data without this would save
correct keys under the wrong provenance and make the next audit more confident and equally
wrong.

---

# ═══ 2026-09-08 — T2 PASS (reanalyse) ═══

Working-tree only. No commits, no pushes. `oracle-loop-paper`, `meta-pattern` and
`lyra-s-research-/` untouched by instruction. Backups `*.bak-t2-20260908` beside every
edited `.tex`. Every paper rebuilt `pdflatex → bibtex → pdflatex ×2`, verified in the
**rendered PDF** with `pdftotext`, and re-gated (**10/10 files, GATE PASSED 6/6**).

## The class that turned out to be four papers, not one

**Selective reporting of collected conditions.** Four separate papers print a subset of
the arms/doses/models sitting in their own shipped data file, with no criterion stated
and no disclosure. Found via identity-geometry #7, then swept as a class. In **every**
one the omitted cells point the same way as the reported ones — which makes the
omission gratuitous rather than convenient, and is exactly why it survived review.

| Paper | Shipped | Reported | Now |
|---|---|---|---|
| `identity-geometry` (agni #7) | 9 pairs in `lexical_control.json` | 5 | all 9, + the constructed-scramble arm named in the design |
| `emotion-accumulation` (agni #5) | 5 arms in `accumulation_controls.json` | 2 | all 5, with the prompt-length confound stated |
| `presence-metric` (agni #5) | 8 doses in `positive_control_72_trials.json` | 6 | all 8 |
| `lyra-technique-ii` (agni #13) | 15 models in `selection_bias_results.json` | 8 | all 15, in a new complete table |

`emotion-accumulation` is worth its own line: the paper **did** carry an
"unreported control arms" limitation, and it names `accumulation_results.json`
(Experiment 1, four arms). The five-arm file is `accumulation_controls.json`
(Experiment 3). *A disclosure that names the wrong artifact reads as coverage.*

## Also applied

- **105b / decision-state agni #12** — all five deconfounding comparisons and both
  W\_K arms now in `tab:deconfound`, with the results file's own condition labels.
  The paper had two of five. **No headline was changed** — which figure leads is T4.
- **117h / decision-state** — the Egypt `C_fake` no-op is disclosed. Verified from
  primary: question identical, answer identical, **all 28 encoding features identical
  (max abs diff 0.0)**. 1 of 30, in the positive control.
- **identity-geometry agni #6** — Tables 1 and 3 are different runs (0.436 vs 0.340 for
  the same nominal comparison, ~2.5 pooled SDs). Provenance now in both captions plus a
  note; ordering-vs-absolute-value distinction stated. Verified across all three runs.
- **identity-geometry agni #8** — `0.978 ± 0.002` now names its estimator. Verified:
  mean 0.97806, SD 0.00217, SEM 0.000167 (the withdrawn 0.0002), η² = 0.9766 over
  **2** unique prompts.

## Verified CLEAN — no defect found

- **`hr-temporal-boundary` agni #1** (the register's own "read this one first"):
  **already fixed on disk and correct.** Recomputing from
  `temporal_boundary_results.json` gives, at 30 scenario-level points,
  ρ = +0.233/−0.050/−0.245, p = 0.215/0.795/0.192 — the exact values now in Table 2,
  and the n=90 values (+0.126/−0.072/−0.151) reproduce too. PDF carries the corrected
  table. **Close this row.**
- **117e / identity-geometry 0.978 pseudoreplication**: already disclosed in-paper; the
  arithmetic checks out (above). Only the estimator label was missing.

## RE-TIERED

- **117g → not a paper defect; the diagnosis is wrong.** `trajectory["0"]` in
  `multiturn_persistence_results.json` is not a trajectory — the keys are **layers**,
  and index 0 is the **pre-attention embedding output**. Its persona and base values are
  identical *by construction*: layer 0 has **2 unique values across 9 turns** (probe
  identity only) where layers 1–23 have 9. Context cannot reach it. "The persona
  condition did not apply" is a false alarm. Additionally the file is cited by **no
  paper** — `grep -rl multiturn_persistence` over `*.tex` returns nothing. The same
  structural reading applies to the *Missing primaries* note on
  `trajectory_activations.npz` layer 0.
- **105 / decision-state FWL** — untouched, correctly tiered T3 ("needs the run").
- **135 / behavioral-proof spec** — untouched. T3 and CC's.

## NEW — found while working, needs a decision

- **The shipped four-decimal AUROCs in `decision-state` do not reproduce outside the
  original environment.** An independent re-execution of the Phase-3 code against the
  shipped `phase2_features.json` reproduced every comparison's **direction and
  significance status** but not its fourth decimal (deltas 0.001–0.016 AUROC;
  scikit-learn 1.7.1 / numpy 2.3.1). This is independent support for agni
  `decision-state-paper` #14. A two-decimal caveat is now in the paper's Limitations.
- **Single-entity leave-one-out moves the deconfounding positive control by 0.079.**
  In that same reimplementation, `A_easy_vs_C_fake` ranges over [0.816, 0.904] across
  the 30 leave-one-out fits, and **dropping Egypt gives the minimum** (rank 1/30,
  z = −2.12). The critical comparison is stabler: [0.774, 0.862], sd 0.021. Because the
  pipeline does not reproduce the shipped fourth decimal, these figures were **not** put
  in the paper. Deciding whether to drop Egypt and re-derive needs the original
  environment — **T4, or T3 if the environment is gone.**
