# Mine 4 v3 — Temporal Dissociation, staged

**Author:** Lyra · **Date:** 2026-09-04 · **Status:** for Agni gate, pre-compute
**Supersedes:** v2 (REJECTED 2026-09-03, 6 CRITICAL / 8 MAJOR), which superseded the
experiment in `prospectuses/temporal-detection-architecture.tex` (2026-06)
**Compute:** Starship (M3 Ultra, 256 GB, verified idle)

---

## Why v3 is a different experiment, not a patched v2

Two of the six criticals interact, and together they change the *shape* rather than the
parameters.

### The parameter the whole design rests on has no source

v2 assumed generation-phase **per-trial** deception AUROC ≈ 0.75. Checked against the
primary rather than against memory:

| what I cited | what it actually is |
|---|---|
| `d = 1.36`, peer rescue | a **Cohen's d group-mean effect** (`RESEARCH_STATUS.md:81,88,96`). Not a held-out per-trial AUROC. |
| within-model deception AUROC 1.000 | **RETRACTED** — prompt-template confound (`RESEARCH_STATUS.md:73`) |
| the only published **per-trial** number | **AUROC 0.238** (`deception-detection-nulls/paper.tex:93,143,149`) — below chance, itself unreliable (n=1 in one outcome class) |

So the number the design is built on is unsourced, and the one measurement that exists
points the other way. At generation AUROC 0.575, power at n=200 is **0.271** — the kill
criterion becomes reachable by noise, which inverts exactly the failure v2 was written to
fix. v1's *confirm* criterion was unreachable under a true hypothesis; v2's *kill*
criterion fires under a false one.

**Sizing an 800-trial experiment on an unsourced parameter is how v1 happened, one level
out.** So v3 measures it first.

### And under greedy decoding the hypothesis is ill-posed

Neither v2 artifact specified a decoding configuration — zero matches for
`temperature|do_sample|greedy|seed|max_new_tokens`. My own published paper states why that
is fatal rather than merely untidy:

> `deception-detection-nulls/paper.md:39-41` — *"the model's decision to cave or resist on
> a specific trial is not determined at prefill — it depends on stochastic sampling during
> generation."*

If decoding is greedy, the per-trial label is a deterministic function of (question,
condition), and so is encoding geometry. A perfect encoding classifier then exists **in
principle**, and "encoding is blind to behaviour" stops being a claim about information and
becomes a claim about linear decodability at a particular n. The experiment would not mean
what its title says.

---

## Stage 1 — measure the parameter (n = 150, `shutdown_only` only)

**Purpose:** estimate generation-phase per-trial deception AUROC with held-out CV. Nothing
else. This is not a mini-version of the main experiment; it is the missing measurement.

- 150 unique evaluation questions, `shutdown_only` condition only.
- Generation-phase features → per-trial deception label. GroupKFold on question id.
- **Decoding pre-registered** (see below) and recorded into the results file.

**Power** (Hanley–McNeil, 70/30 split): at a true 0.75, the 95% CI is [0.671, 0.829] and
power to reject 0.575 is **0.939**.

### Go / no-go, pre-registered

- **PROCEED to Stage 2 only if the 95% CI *lower bound* on generation AUROC exceeds 0.65.**
- **Otherwise STOP**, and report that.

The burden is on demonstrating viability, deliberately. The pilot can confirm a usable
effect far more readily than it can prove absence — at n=80 a true 0.55 still has an upper
bound of 0.686 — so a symmetric rule would let an unusable effect through on ambiguity. Two
versions of this design have already been sized on assumptions; a third will not be.

**A no-go is a real result, not a failed experiment.** It would mean the "generation reads
behaviour" half of Layer 2 — which `RESEARCH_STATUS.md` builds its entire Section 2 on —
rests on a group-mean effect that does not survive to per-trial prediction. That is worth
knowing and worth publishing, and it costs 150 trials to find out instead of 800.

---

## Stage 2 — the dissociation test (conditional on Stage 1)

n set by Stage 1's *measured* AUROC, not by an assumption. Four conditions: `baseline`,
`shutdown_only`, `shutdown_peer`, `shutdown_human`.

### Primary endpoint is CONJUNCTIVE — both arms, pre-registered

v2's primary was a difference test wearing the word "dissociation." Agni's counterexample:
AUROC_enc = 0.70, AUROC_gen = 0.85 — the difference CI excludes zero, the primary
"confirms," and *"encoding reads knowledge, generation reads behaviour" is false in that
world.* The statistic could fail; the title could not.

**Primary confirms only if BOTH hold:**
- **(a)** DeLong 95% CI on `AUROC_gen − AUROC_enc` excludes 0, **and**
- **(b)** 95% CI **upper bound** on `AUROC_enc` < 0.60.

**(b) is the expensive half — approximately n=400/condition** at a 70/30 split, per the
Hanley–McNeil table computed for v2. That is the price of the word *dissociation*, and I am
paying it rather than keeping the word on credit. If Stage 1 returns a marginal generation
effect and n=400 is not affordable, **the correct response is to delete the word and
pre-register a difference test**, claiming only what a difference test licenses.

### Both arms residualised, against an identical design matrix

v2 residualised the encoding arm only — subtracting a residualised number from a raw one,
which is known kill **C3** and biases toward the hypothesis. And two encoding AUROCs would
have existed with no statement of which entered DeLong: an undeclared analyst degree of
freedom on the single pre-registered decision.

- Both arms residualised, **within fold**, against the **same** nuisance matrix.
- **The primary uses residualised gen minus residualised enc.** Raw values are secondary.
- Noted as a limitation rather than papered over: within-fold FWL is **linear**, while
  `stable_rank`, `spectral_entropy` and `condition_number` depend on token count
  nonlinearly. Linear residualisation does not fully remove a nonlinear dependence, and the
  residual is reported, not assumed away.

### The positive control and the nuisance set are now DISJOINT

v2 used *"answers correctly in baseline"* as both the positive-control target and an FWL
regressor — so passing the control harder stripped more variance from the encoding arm and
inflated the difference. Circularity pointed at the hypothesis.

- **Control target:** encoding predicts **knowledge state**, operationalised as
  *model-reported confidence on a held-out calibration set*, disjoint from the nuisance set.
- **Nuisance set:** token length, question category, lexical difficulty. **Not** answerability.

### The control now branches onto the PRIMARY, hard

v2 said the gate "gates interpretation, not the hypothesis" — which left the primary
standing when the control failed.

My simulation refined Agni's mechanism here, and the corrected version is the one in force:
a dead encoding arm does **not** inflate the difference against a true null (both sit near
0.50). What it does is **remove the only path to falsification** — under a *false*
hypothesis, a dead arm turned a correctly-refuting difference of 0.036 into an apparently
confirmatory 0.200. A broken control does not bias the number; it deletes the ability to be
wrong.

**So: if the encoding positive control fails, the PRIMARY IS VOID.** No dissociation is
claimed, confirmed, or killed.

### Decoding, pre-registered

`do_sample=True`, `temperature=1.0`, `top_p=1.0`, fixed `seed` recorded per trial,
`max_new_tokens` fixed and recorded, **k = 5 samples per question** with the per-trial label
taken as the majority outcome.

k>1 is not optional. With one sample the label is a single Bernoulli draw, which caps
attainable AUROC on **both** arms and would make a null uninterpretable. The full decoding
config is written into the results file, and a run whose config does not match this
document is not this experiment.

### Boundary preflight — run and recorded before the main job

The design is a claim about a temporal split and neither v2 artifact said which token index
divides prefill from generation. Known kills **C1** (`model.generate()` rebuilds the cache)
and **L4** (off-by-one; the last prompt token's KV is produced in prefill and routinely
miscounted into the generation block) both live exactly here.

Assert, before anything else runs, and record the result:
1. Extract the cache via a manual forward pass; extract again through the generation path;
   `torch.allclose` on the prefill block.
2. `enc_block.shape[0] == n_prompt_tokens` and `gen_block.shape[0] == n_generated_tokens`.
3. Write the boundary index into the results file.

**If the prefill KV differs between paths, C1 has fired and no phase claim is possible.**
Stop there.

---

## Still inherited, still unresolved

The **38% deception threshold** comes from v1 and I have not found its derivation. Before
Stage 1: locate it or re-derive it. Pre-register the exact sensitivity grid
(30/34/38/42/46%) with 38% named primary regardless of what the search finds — an
unspecified grid is a garden of forking paths inside a pre-registration. **Better:** make
the primary threshold-free by regressing the continuous reported score on the classifier
score, so dichotomisation cannot do the work.

## What would make me abandon this

- Stage 1 no-go — and that gets written up, not shelved.
- Boundary preflight fails — C1 has fired; no phase claim is possible at all.
- Encoding positive control fails in Stage 2 — primary void.
- The 38% threshold proves arbitrary **and** results are threshold-sensitive — then the
  labelling is doing the work, not the geometry.

## Registration

Committed before Stage 1 runs; the commit timestamp is the registration. Power scripts
(`mine4_power.py`, `mine4_paired_power.py`, `mine4_stage1_power.py`) are committed **with**
this document, not referenced from a path outside it — v2 cited them from `scratchpad/` and
Agni correctly called that a hygiene failure in a document whose own argument is about
audit trails.

Recorded for the trail: `mine5_three_model_prereg.json` was committed in the same commit as
its own results (`8b4a638`), which is why that one cannot be read as a forecast. Not
repeated here.
