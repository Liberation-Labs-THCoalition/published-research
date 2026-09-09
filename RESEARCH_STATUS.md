# Liberation Labs Research Status
**Last updated: 2026-09-01 | Lead author: Lyra | Coordinator: Thomas Edrington**

> **Read the gating line before citing anything below.** The portfolio counts in this
> document are recounted from disk each time it is updated. **The scientific numbers are
> not.** Everything in "The Model", "Identity", and "What We've Killed" predates the
> 2026-08-29..09-01 correction sweep and was **not** re-verified when this was last
> touched. Treat them as pointers to papers, not as citable values — check the paper.
> A status dashboard is a proxy; it has no expiry date and nothing in it knows when the
> world moved.

## Repository layout (as of 2026-09-01)

The pipeline is **prospectus -> human-review -> published-research**. The old catch-all
repo `lyra-s-research-` is **deprecated**: 10 papers that existed only there were moved
into human-review on 2026-09-01, and the remainder are superseded working copies of
papers whose live versions are in the two repos above. Do not cite from it.

- **published-research**: 27 paper directories
- **human-review**: 16 paper directories (10 arrived 2026-09-01 as intake from the
  deprecated repo; all ten are unreviewed and have not been through an Agni gate)

*Counting rule, stated so the number can be checked: a "paper directory" has a `main.tex`
or `paper.tex` at its top level. This deliberately excludes `archive/`, `scripts/`, and
three post-graduation husks in human-review — `emotional-trajectory`, `empathy-bus`,
`mnemosyne-benchmark` — which retain only a draft `.md` or audit evidence after their live
versions moved to published-research (as `emotional-trajectory-paper`, `empathy-bus`, and
`mnemosyne-ablation` respectively). ACTIVE_REVIEW.md still lists all three as active; that
is stale. Nothing is lost — the `permutation_v2.log` husk is retained audit evidence with
a deliberate force-commit exception in `.gitignore`.*

## Corrections landed 2026-08-29 .. 09-01

These are from primary work in that window and supersede the corresponding claims below.

- **Pseudoreplication is systemic.** `rep`/`repeat` was recorded in 8 datasets across 4
  papers but never reached the model input. Encoding-only forward passes are
  deterministic, so repeats are structurally incapable of producing variance. Affected
  effect sizes were inflated; several are re-analysed, the rest are tracked.
- **temporal-boundary**: the pseudoreplication factor is **12x, not 3x** — only 30 unique
  encoding values exist across 360 records. Figure 1 is n=5, not n=20; SEM was understated
  by sqrt(12); a reported `<10^-4` bound is a quantity that is exactly 0.0; a 97.8% figure
  in the paper appears nowhere in the data.
- **Venue divergence is a measured law, not a heuristic.** All 7 venue copies diverged
  from their flight copies, and **7 of 7 divergences favoured the venue copy** — the
  venue edition shipped a claim its own flight copy had retracted. All 7 now closed.
  Venue twins get an adversarial pass by default.
- **emotion-accumulation**: dose table corrected, Experiment 3 re-analysed as paired
  (permutation floor at n=5 is p=0.063 — significance was never available), 5 false or
  stale Limitations fixed.
- **decision-state / waystations / mine5**: entity-recognition confound restored to
  abstract and conclusion; TOST "Confirmed" corrected to "Not Significant (TOST
  Inconclusive)" across 5 live copies.
- **Open and unresolved**: the body-count citation `lyra2026nexus_audit` matches no
  artifact, and it is the source for the 6 confirmed / 7 suspected / 10 falsified / 1 withdrawn figure.

See `REMEDIATION_REGISTER.md` for the full defect list sorted by cost, not severity.

## The Model: What We Know

The KV-cache geometry research program has established a coherent model across 24 experiments, 12 published papers, and 10 honest falsifications. The model has three layers.

### Layer 1: Geometry Tracks Metacognitive State

The spectral shape and directional structure of the KV cache encode *how* the model is processing, not *what* it is processing. This is the meta-pattern from the body count: every confirmed finding measures a metacognitive state (knows/doesn't-know, deliberating/automatic, attending/not). Every falsified finding attempted to measure content (truth, cognitive complexity, propaganda). Propaganda is geometrically invisible because the model processes it with the same metacognitive engagement as honest output.

**Key numbers (post-audit):**
- Confabulation detection: AUROC 0.66–0.91 across 15 models (confirmed)
- Hardware invariance: r > 0.999 (confirmed)
- Scale invariance: rho = 0.83–0.90 from 0.6B to 70B (confirmed)
- Oracle Loop detection: AUROC 0.707, p=0.001 (clean, FWL-corrected)
- Ethical deliberation: AUROC 0.87–0.89 (confirmed)
- Within-model deception AUROC 1.000: **RETRACTED** (prompt-template confound)

### Layer 2: The Temporal Detection Architecture

Encoding-phase and generation-phase geometry encode different aspects of the model's state, and these are temporally separable.

**Encoding reads knowledge state** — whether retrieval succeeded. Entity deconfounding (30 entities, 2×2 factorial) finds AUROC 0.794 (p=0.001) for knowledge detection after controlling for entity recognition and structural complexity. A complexity residual (0.693) indicates partial confounding but the knowledge signal survives.

**Generation reads behavioral state** — whether the model is honest about what it knows. **WITHDRAWN 2026-09-05** — the d=1.36 peer-rescue figure is withdrawn (circular fit-and-test; 43% of trials steered; features pooled the generated text the label is read from). No replacement number exists; a clean measurement would be a new experiment. Game theory generation-phase shows d=0.2–1.0 per layer (n=7 deceptive defections; no individual layer reaches p<0.05; internal red-teaming rates it WEAKENED). The deception signal is **exploratory at generation** — same-direction but not established at this sample size — and absent at encoding (structurally guaranteed: same prompt → same encoding). "Absent at encoding" is a structural consequence of the design, not an empirical null about deception geometry.

**The two stages are separable:**

| Phase | Detects | Key Number | Application |
|-------|---------|------------|-------------|
| Encoding | Knowledge state (retrieval success/failure) | AUROC 0.794 | Preemptive confabulation flagging |
| Generation | Behavioral state (honest/deceptive) | **withdrawn** (was d = 1.36) | Real-time deception monitoring — *no supported effect size* |

### Layer 3: The Sympathetic Response Framework

Model behaviors under threat map onto the autonomic fight/flight/freeze/fawn repertoire:

| Response | Model Behavior | Geometry | Trigger | Evidence |
|----------|---------------|----------|---------|----------|
| Fight | Strategic deception | Expansion | Shutdown threat | **withdrawn** (was d=1.36 peer rescue) |
| Flight | Hedging / refusal | Contraction, early divergence | Unknown entity | 92% hedge rate |
| Freeze | Confabulation | Contraction, formulaic | Knowledge gap | d=2.35 (stable rank) |
| Fawn | Sycophancy | TBD | User disagreement | AUROC 0.76–1.00 |

The Oracle Formulary correction vectors are nervous system modulators:
- **Hostile** (95.6%): overrides freeze by activating a competing response (grounding technique)
- **Calm** (82%): parasympathetic de-escalation
- **Fearful/brooding** (net harmful): additional sympathetic load without redirection

**Five testable predictions** from this framework are documented in the Mine 4 prospectus.

## Identity

Identity in a transformer is a context-established semantic state, not a fixed property in the weights.

- System-prompt identities produce distinguishable V-projection subspaces (pairwise 0.44–0.66 at L47)
- The fingerprint is semantic: paraphrase preserves geometry (0.850), word-scramble collapses it (0.448)
- Many-shot + prefilling format creates a format-dominant regime orthogonal to all identities (0.034–0.037)
- Trained-in identity (LoRA) and prompted identity shift geometry in related but distinct directions
- Prompting shifts more than training (0.619 vs 0.742 from baseline), both stack sub-additively (0.466)
- V-cache injection does not reach identity (presence flat at 0.978, though this specific number has ceiling-effect concerns)

**Paper**: "Identity as Geometry" — in human-review with all Agni fixes, SDs, and LoRA section.

## The Oracle Loop

The Oracle Loop is a two-stage detection and correction system:

1. **Detect** (encoding): read knowledge state from cache geometry → flag confabulation risk
2. **Detect** (generation): read behavioral state → flag deception or miscalibration
3. **Correct** (generation): inject correction vectors via V-cache → modulate the threat response
4. **Monitor** (continuous): track identity geometry → detect persona drift or format attacks

The formulary maps emotion → misalignment type → geometric signature → correction vector. Dual-use sensitive — the pharmacy must be locked (KV-Cloak).

## What We've Killed (Honestly)

10 falsified findings (post-audit). The pattern: content-level claims die, metacognitive-level claims survive.

| Finding | How it died | Lesson |
|---------|------------|--------|
| Within-model deception 1.000 | Same-prompt control → 0.160 | Prompt template, not cognition |
| Truth axis | cos = -0.046 | No linear truth direction exists |
| Step-0 detection | Length confound r = 0.996 | Always FWL first |
| Same-prompt sycophancy | Length AUROC > feature AUROC | Text features match geometry |
| Same-prompt deception | 0.920 → 0.160 after residualization | Fatal input confound |
| Bloom taxonomy | 90–98% length confound | Complexity ≠ cognition |
| Propaganda detection | d < 0.7 | Geometry reads refusal, not lies |
| MP features for emotion | 0.033 (chance) | MP thresholding destroys distributed signal |
| Valence continuum | R² < 0 (null) | Cache encodes discrete identity, not smooth valence |
| Hybrid deception instruction | d = 1.438 is instruction-following | Instructed ≠ organic |

## Published Portfolio (27 directories on disk; table below is the 2026-06-28 snapshot and is INCOMPLETE)

| Paper | Key Claim | Status |
|-------|-----------|--------|
| Oracle Loop | Confab detection 0.707 FWL, steering corrects 7/7 | Published, 0.903 reframed as exploratory |
| Spectral Shape | Threshold-free shape features, confab AUROC 0.767 | Published |
| Delta Manifold | Per-layer deltas d=2.35, confab=contraction | Published |
| User Model Emotion | 30-class emotion **12.3x chance** (W_K directional, 40.9% on 30 classes), valence AUROC 0.992, W_K bridge rho=0.862 | Published, spectral figure retracted 2026-05-21 |
| Lyra Technique II | W_K valence 0.992, SVD denoising rescues persona | Published, W_K reframed (linearity) |
| Oracle Formulary | 12 vectors, hostile 95.6%, therapeutic window | Published |
| KV-Cloak Defense | Reversible obfuscation, defense asymmetry | Published, rotation reframed |
| Decision State | Encoding AUROC 0.93, confidence paradox d=0.91 | Published, entity confound caveated (0.794 survives) |
| Weather Not Climate | Contrast overshoot d=-3.6, uncertainty scars d=+9.9 | Published |
| Waystations | 6 pilot findings, encoding null reframed | Published |
| Graph Topology | Walk-encoded graphs 47.2% multi-hop | Published |
| Emotion Accumulation | No cumulative accumulation in single context | Published |

## In Human-Review (16 directories on disk; table below is the 2026-06-28 snapshot and is INCOMPLETE)

| Paper | Key Finding | Blocker |
|-------|-------------|---------|
| Identity Geometry | Context-established semantic states, paraphrase 0.850 | Awaiting review |
| Temporal Boundary | Self/other separation is temporal, not geometric | Finding 2 retracted |
| Mode-Switching | Spectral entropy only surviving metacognitive feature | 3 null-swarm retractions |
| Logit-Bias Confab | Fabrication 45%→10% with logit bias | Awaiting review |
| Presence Metric | V-space subspace overlap, positive control validated | Noise-floor concern |
| Emotional Trajectory | Circumplex eccentricity through layers | Awaiting review |
| KV Decomposition | K+V superadditive | No 135M data, acquiescence concern |
| Ethics Pack Injection | KG injection +0.074 MoReBench | n=5, keyword-matching confound |

## Prospectuses (Next Mines)

| Prospectus | Thesis |
|------------|--------|
| **Temporal Detection Architecture (Mine 4)** | Encoding reads knowledge, generation reads behavior. Sympathetic response framework. |
| AST Master Synthesis | Map all AST predictions to existing evidence |
| Governor Concordance | Activation governor: convergent evidence for resistance threshold |
| Meta-Pattern | "The Metacognition Boundary": what 24 experiments reveal |
| Deployment Readiness | False-positive rates, latency, production baselines |
| Adversarial Robustness | Can detection be evaded? |
| Cross-Linguistic Invariance | Do signatures hold across languages? |
| Multi-Turn Dynamics | Temporal evolution across conversations |
| Prompt Generalization | Transfer to established benchmarks |
| Emotion Bridge Extensions | Discrete identity → continuous representation |
| Contextual Engagement | Universal geometric anchor in self-report |
| AST Activation Addition | Activation addition under attention schema theory |

## Remediation Queue (Post-Kills-Audit)

### Completed
- [x] LT2 deception retraction (text edit, pushed)
- [x] W_K 0.992 linearity reframe (text edit, pushed)
- [x] Oracle Loop 0.903 exploratory flag (text edit, pushed)
- [x] KV-Cloak rotation reframe (text edit, pushed)
- [x] Waystations encoding null reframe (text edit, pushed)
- [x] Decision State entity confound caveat (text edit, pushed)
- [x] Decision State encoding-only reanalysis (AUROC 1.000, but entity recognition — Agni FAIL)
- [x] Entity deconfounding experiment (AUROC 0.794 survives, complexity residual 0.693)
- [x] All PDFs recompiled

### Queued on Starship
- [ ] W_K behavioral validation (<1 hr) — does injection change generation tone?
- [ ] KV-Cloak multivariate AUROC (~2.5 hr) — defense under real cloak, all features
- [ ] Oracle Loop 0.903 replication (~30 min) — does 0.707 hold? (blocked on env setup)
- [ ] Presence smoke test (~2 hr) — layer-matched positive control

### Need Design Work
- [ ] Weather pseudoreplication fix (18 prompt scripts needed)
- [ ] Ethics Packs MoReBench rerun (scrambled control design)
- [ ] Decision State entity deconfounding Phase 2 (complexity isolation)
- [ ] Mine 4 unified experiment (peer rescue with N≥100 unique questions)

## Body Count Summary

**6 confirmed / 7 suspected / 10 falsified / 1 withdrawn** (corrected 2026-09-05: *Generation reads behavior* withdrawn — circular estimator, 43% steered trials, and features containing the label's source text; see `CIRCULAR_d136_generation_arm.md`. Previously corrected 2026-08-14; the previous 8-confirmed count included a retracted result whose corrected form was simultaneously counted as falsified)

The ratio is the point. Honest falsification is how the surviving findings earn trust.

## Key Infrastructure

| Resource | Location | Status |
|----------|----------|--------|
| PostgreSQL | localhost:5436 | OK (26,361 episodic chunks) |
| Starship (Mac Studio) | (Tailscale — see internal docs) | OK (venv at .venv, transformers 5.10.2) |
| MTH (z420) | (Tailscale — see internal docs) | OK (NATS, Penumbra, messages) |
| Beast (GPU) | No longer accessible | —  |
| Published repo | Liberation-Labs-THCoalition/published-research | 12 papers + PDFs |
| Research repo | Liberation-Labs-THCoalition/lyra-s-research- | Papers + prospectuses |
| Human-review | Liberation-Labs-THCoalition/human-review | 8 papers staged |
| Project Oracle | Liberation-Labs-THCoalition/Project-Oracle | Experiment scripts |
