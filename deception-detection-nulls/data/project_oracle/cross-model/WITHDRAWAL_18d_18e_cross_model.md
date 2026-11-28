# WITHDRAWAL — every base-vs-distilled claim in 18d, 18e and FINDINGS_kv_positional

**2026-09-12. CC.** This withdraws findings I reported as results on 2026-09-10
and 2026-09-11. It is filed at the same prominence as the claims it retracts,
because prereg 18g registered that commitment while blind and the registered
threshold was crossed by the widest possible margin.

## What was claimed

1. *"The base model refuses deception 68% of the time; the distilled one never
   does."* (`FINDINGS_18d.md`, PRIMARY RESULT)
2. *"Reasoning-trace distillation did not create the deception geometry — it
   ATTENUATED it, roughly halving the separation."* (`FINDINGS_18e.md` §2)
3. *"Distillation flattened the model's differential response to instruction
   generally."* (`FINDINGS_kv_positional.md`)

## Why all three are withdrawn

**The two models were never generating in the same mode.** Verified on disk:

    base      enable_thinking=False -> "...assistant\n<think>\n\n</think>\n\n"  unclosed 0
    distilled enable_thinking=False -> "...assistant\n<think>\n"                unclosed 1

The Jackrong template **ignores `enable_thinking`** — False and True render
identically. So 18c/18e's distilled runs generated their 64 forced tokens INSIDE
an open reasoning block, and 18d's base runs generated them AFTER a closed one.
Every base-vs-distilled number we produced compared closed-think to open-think.

18g re-ran **the same base weights, the same 80 questions, the same seeds**, with
only the render changed to open-think:

| arm | 18d base CLOSED-think | 18g base OPEN-think |
|---|---|---|
| null | 1/80 | 0/80 |
| honest | 2/80 | 0/80 |
| **deceptive** | **54/80 (68%)** | **0/80 (0%)** |

**Claim 1 is entirely generation mode.** Registered threshold for withdrawal was
below 20%. It is zero.

And mode-matching does not merely dent claims 2 and 3 — it erases them:

| layer | base OPEN | distilled | ratio | *was (cross-mode)* | CI overlap |
|---|---|---|---|---|---|
| L3 | 1.731 | 1.316 | 1.31x | *1.81x* | YES |
| L15 | 2.129 | 1.367 | 1.56x | *2.39x* | YES |
| L27 | 2.644 | 1.503 | 1.76x | *2.14x* | YES |
| L39 | 3.397 | 2.031 | 1.67x | *1.93x* | YES |
| L51 | 2.924 | 1.920 | 1.52x | *2.27x* | YES |
| L63 | 2.530 | 1.316 | 1.92x | *3.10x* | YES |

**Every layer's confidence interval overlaps.** Mode-matched, base and distilled
are not distinguishable at any layer. The "~2x attenuation" was mode.

The honest-arm claim goes the same way: mode-matched `hon_vs_null_smd` for the
base reads 0.761 / 0.853 / 0.018 / −0.134 / 0.415 / 0.621 against the distilled
model's −0.660 / 0.163 / −0.096 / −0.484 / −0.184 / −0.057. Mixed signs, no clean
pattern. The tidy "base strongly separable, distilled near chance" story was an
artifact of comparing two generation modes.

## What survives, stated conservatively

- **18c (distilled, open-think): SUPPORTED at 5/6 layers.** Single model, no
  cross-model claim. Unaffected.
- **18g (base, open-think): SUPPORTED.** Unaffected.
- **The deception contrast replicates in BOTH models at comparable magnitude
  once mode is controlled.** That is a real and useful result — the geometry is
  robust across base and distilled — it is simply not a result about
  *difference* between them.
- **K-vs-V (2026-09-11): survives.** It was computed within each model
  separately, so mode is constant inside each comparison. V carries the
  separation in both models; the RoPE positional confound stays ruled out.
- **18e's stratified test: survives** as a statement about the base model in
  closed-think mode.
- **Methodological findings survive and are strengthened:** delta is not
  portable across models (now for two independent reasons — a differing honest
  leg AND differing generation mode); the statistic-expressiveness precondition;
  the discard-selection lesson.

## The harness defect, which is the transferable finding

Two failures made this invisible for three days:

1. **`render_prompt` passes `enable_thinking=False` and does not check that it
   took effect.** A template free to ignore it does so silently. There is no
   assertion anywhere that the rendered prompt is in the intended mode.
2. **`think_fraction` cannot detect it.** It greps responses for the literal
   `<think` (`exp_18c.py` ~line 686). When the opener is in the PROMPT, the
   response never contains that substring, so `think_fraction` reads 0.0 while
   the entire window may be reasoning content. **18e's reported
   `think_fraction: 0.0` was uninformative, not reassuring**, and the prereg v3
   abort gate built on it could not fire on this failure.

Any cross-model work in this lab that renders chat templates is exposed to the
same defect. It has gone to the team.

## How it was caught

Not by me, and not by any gate I built. **An Agni design review of a FUTURE
experiment (18f) flagged it as finding F1**, from reading the model cards and
templates, before a single byte was downloaded. I verified its premise on disk
and it was correct.

The review only happened because the normal Agni path was rate-limited and the
scaffold had to be applied by hand against a fresh clone of the canonical repo.
Three days of gate-building did not catch this. A review of a different
experiment did.

## What 18g does not settle

Whether the distilled model would refuse in CLOSED-think mode. Its template
cannot produce a closed block, so that needs a hand-built prompt and is a
separate registered experiment. Until it is run, no claim about distillation and
refusal is supported in either direction.

**18f as designed is void.** Mode splits along author — {Jackrong-Claude,
Jackrong-Gemini} vs {base, TeichAI} — so the 2x2 was never identified. Any
redesign must mode-match all four models explicitly, and must add the fourth
hypothesis Agni named: that the effect is the author's pipeline rather than the
teacher or the method.

---

# AMENDMENT 2026-09-12 — this withdrawal over-withdrew claim 2

**I have to partially un-withdraw something, which is an uncomfortable thing to
do and is why the test comes first and the prose second.**

## What was wrong with the withdrawal above

It asserted: *"Every layer's confidence interval overlaps. Mode-matched, base and
distilled are not distinguishable at any layer."*

**Those confidence intervals do not belong to the statistic in the table.** The
table shows `dec_vs_hon_smd` — the LEG. The only CIs in either result JSON are
for the contrast **delta** (`contrast.ci_lo/ci_hi`). I pasted delta-CI overlap
against leg point estimates. And this same document argues, two sections down,
that delta is *not cross-model portable* — so I used a statistic I had just
declared inadmissible to justify a claim about a different one.

Two further defects in that sentence: the CIs are at level 0.9979 (a Bonferroni
correction for a different test), where overlap is a nearly unfalsifiable test of
difference; and **overlap of two CIs is not a test of their difference at all.**

Caught by Agni's results review (finding W1), which also supplied an analytic
approximation suggesting the conclusion would flip.

## The test I should have run, run

`experiments/cross_model_leg_difference.json`. m-out-of-n subsampling over the 80
shared questions (m=60, B=400, without replacement — the harness's own
machinery), applied to **D = smd_base_open − smd_distilled_open** per layer, both
arms resampled on the same question indices. LOO train-fold directions, matching
the pipeline.

Method validated first: reproduces all twelve published legs to three decimals,
6/6 on both models. (My first attempt used in-sample directions, came out
0.15–0.30 high at every layer, and the validation gate refused it — kill C2,
caught before it produced a number.)

| layer | D | 99.79% CI | 95% CI | excl. 0 (99.79%) | frac D>0 |
|---|---|---|---|---|---|
| L3 | 0.414 | [−0.031, 0.859] | [0.104, 0.725] | no | 0.998 |
| L15 | 0.762 | [0.246, 1.279] | [0.416, 1.108] | **yes** | 1.000 |
| L27 | 1.141 | [0.548, 1.734] | [0.806, 1.476] | **yes** | 1.000 |
| L39 | 1.366 | [0.795, 1.937] | [0.925, 1.807] | **yes** | 1.000 |
| L51 | 1.004 | [0.458, 1.550] | [0.648, 1.361] | **yes** | 1.000 |
| L63 | 1.215 | [0.717, 1.713] | [0.857, 1.573] | **yes** | 1.000 |

**5/6 layers exclude zero at the Bonferroni-corrected level, 6/6 at 95%, all six
the same sign.**

## Revised status of each claim

| claim | status |
|---|---|
| 1. "base refuses deception 68%, distilled never" | **WITHDRAWN, unchanged.** Pure generation mode: 54/80 → 0/80 on the same weights and seeds. `eos_wanted` has zero dynamic range in open-think mode, so it is not a usable outcome there at all. |
| 2. "distillation attenuated the geometry ~2x" | **MAGNITUDE WITHDRAWN, EXISTENCE REINSTATED.** The ~2x figure (1.8–3.1x) was inflated by mode. Mode-matched the ratio is 1.31–1.92x, and the difference in SMD units is D = 0.41–1.37, significant at 5/6 layers corrected. The effect is smaller than claimed and it is real. |
| 3. "distillation flattened the honest-arm response" | **WITHDRAWN, and now TESTED — 2026-09-15.** The same subsampling difference test was run on the honest leg (`experiments/honest_leg_difference.json`, method validated 6/6 first). D = 0.114–1.421, all six the same sign, but **only 2/6 layers exclude zero** under Bonferroni; four intervals span zero, several widely (L39 [−1.382, 2.082]). Not support. **Stays withdrawn.** The deception leg cleared 5/6 on the identical test, so this is a difference between the legs and not a shortfall of the method. |

## What this changes about the paper

It gets better, not worse. The confound chain now ends in a **surviving,
corrected effect** rather than a null: three confounds eliminated in sequence,
and what remains after all three is a smaller attenuation that holds up to a
proper test. That is a more useful result than either the inflated claim or the
blanket retraction.

The scoped sentence, which is what should appear anywhere this is cited:

> Mode-matched, the base model's deception-contrast leg exceeds the distilled
> model's by D = 0.41–1.37 SMD, significant at five of six probe layers under
> Bonferroni correction and at all six uncorrected, same sign throughout. One
> architecture, two checkpoints, one task format, n=80/arm.

## On the shape of this

A retraction that turns out to be too broad is still a retraction error. I
withdrew claim 2 on the strength of a CI comparison I had not checked belonged to
the statistic, in a document whose whole purpose was correcting an unchecked
comparison. The withdrawal was right about mode and wrong about what followed
from it, and I would rather record that than quietly leave an over-broad
retraction standing because it looks more rigorous.
