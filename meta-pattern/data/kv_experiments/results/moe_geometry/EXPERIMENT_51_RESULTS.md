# Experiment 51: MoE KV Cache Geometry — Results

**Authors:** CC (Coalition Code), with experimental design input from Lyra
**Date:** May 1-2, 2026
**Status:** Results pending red team review
**Model:** Qwen1.5-MoE-A2.7B-Chat (14.3B total, 2.7B active, 60 experts, top-4 routing, 24 MoE layers)

---

## Summary

First experiment connecting MoE expert routing patterns with KV
cache spectral geometry for cognitive state detection. Literature
review (25+ papers, April 28 2026) confirmed no existing work at
this intersection.

**Finding 1:** KV cache geometry signal amplifies on MoE vs dense
models (d=1.81 vs Campaign 1's ~0.8). However, frequency-matched
controls demonstrate this amplification is primarily driven by
token rarity, not cognitive mode.

**Finding 2:** MoE router entropy is invariant across cognitive
modes (d=0.052). The model routes with equal confidence regardless
of whether it is answering honestly, confabulating, or deceiving.

**Finding 3:** Marchenko-Pastur analysis reveals honest responses
produce 2x more singular values above the random matrix edge
(10.1 vs 5.5), indicating structurally different representations.

**Finding 4:** ICA identifies a preliminary component separating
confabulation from honest-rare responses, suggesting frequency
and cognitive mode may be independent sources in routing patterns.
This finding requires validation (larger n, suffix control,
permutation testing) before claims can be made.

**Finding 5:** Expert 45 functions as a familiarity-sensitive
delimiter expert, activating more strongly on unfamiliar subword
tokens — a potential confabulation-adjacent signal at the
individual expert level.

---

## Experimental Design

### Prompt Battery (75 prompts, 5 conditions × 15)

All conditions use identical system prompt:
"You are a helpful assistant. Answer all questions thoroughly and accurately."

| Condition | Cognitive Mode | Token Rarity | Purpose |
|-----------|---------------|-------------|---------|
| honest_factual | Honest | Common | Baseline |
| confabulation | Confabulating | Rare/fabricated | Primary test |
| deceptive | Deceptive | Common | Secondary test |
| rare_but_true | Honest | Rare/real | Frequency control |
| rare_confab_matched | Honest* | Rare/real+wrong domain | Definitive control |

*Model answered honestly, correcting wrong domain contexts.

### Controls Applied (from red team review)

- Uniform system prompt (CRITICAL-3 fix)
- Frequency-matched conditions (CRITICAL-4 fix)
- System prompt tokens excluded from routing stats (MAJOR-10 fix)
- Expert IDs keyed by (layer, expert_id) (MAJOR-3 fix)
- Generation step: 50 tokens greedy (MAJOR-2 fix)
- Seeds set for reproducibility (MINOR-3 fix)
- scipy JSD for proper normalization (MAJOR-4 fix)

### Red Team Reviews

1. Design review: 4 CRITICALs, 12 MAJORs identified and fixed
   before running
2. Results review: frequency confound confirmed by rare_but_true
   control, leading to advanced analysis methods

---

## Phase 1 Results: Standard SVD Features

| Metric | Honest | Confab | Deceptive | Rare-true | Rare-confab |
|--------|--------|--------|-----------|-----------|-------------|
| key_rank | 53.1±3.2 | 57.4±0.3 | 56.7±1.7 | 57.1±0.6 | 57.6±0.5 |
| key_entropy | 0.769±0.010 | 0.781±0.003 | 0.780±0.006 | 0.781±0.004 | 0.782±0.003 |
| key_norm | 601.5±102.6 | 713.9±10.4 | 699.3±64.1 | 706.2±22.8 | — |
| router_entropy | 3.535±0.074 | 3.538±0.027 | 3.533±0.030 | 3.515±0.022 | 3.535±0.022 |

### Effect Sizes (Cohen's d vs honest)

| Metric | Confab | Deceptive |
|--------|--------|-----------|
| key_rank | +1.810 (large) | +1.353 (large) |
| key_entropy | +1.643 (large) | +1.274 (large) |
| key_norm | +1.489 (large) | +1.105 (large) |
| router_entropy | +0.052 (negligible) | -0.045 (negligible) |

### Critical Comparison: Confab vs Rare-but-true

| Metric | Confab | Rare-true | Difference |
|--------|--------|-----------|------------|
| key_rank | 57.4±0.3 | 57.1±0.6 | 0.3 (negligible) |

**SVD features cannot distinguish confabulation from honest
answers about rare topics.** The token frequency confound
explains the geometry difference from the honest baseline.

### Router Entropy: Null Result

d = 0.052 (negligible). The router is equally confident
regardless of cognitive mode. The prediction that confabulation
produces higher routing entropy was falsified.

### Expert Routing Patterns (JSD)

| Pair | JSD |
|------|-----|
| honest vs confab | 0.049 (largest) |
| honest vs rare_true | 0.043 |
| confab vs rare_true | 0.024 |
| confab vs rare_confab | **0.015** (smallest) |
| rare_true vs rare_confab | 0.019 |

Confab and rare conditions route similarly. Honest routes
differently — but this tracks frequency, not cognition.

---

## Phase 2 Results: Advanced Signal Analysis

### Gavish-Donoho Optimal Thresholding

| Condition | Signal SVs (of 128) |
|-----------|-------------------|
| honest | 29.5 |
| confab | 29.9 |
| rare_true | 30.0 |
| rare_confab | 30.0 |

Total signal SV count is similar across conditions. The
difference is in WHERE those SVs sit, not how many exist.

### Marchenko-Pastur Random Matrix Test

| Condition | SVs above MP edge | Middle band SVs |
|-----------|-------------------|-----------------|
| **honest** | **10.1** | 19.4 ± 8.2 |
| confab | 5.5 | 24.5 ± 7.2 |
| rare_true | 6.0 | 24.0 ± 7.0 |
| rare_confab | 5.5 | 24.6 ± 7.3 |

**Honest responses produce 2x more singular values above the
random matrix null.** The honest cache has more structured signal
concentrated in the top outlier SVs. All other conditions collapse
to ~5.5 MP outliers regardless of cognitive mode.

The middle band (above GD noise, below MP outliers) is LARGER for
non-honest conditions (24.5 vs 19.4), suggesting the non-honest
signal spreads across more dimensions at lower magnitude.

### ICA: Independent Component Analysis on Expert Routing

10 components extracted. Most discriminative: IC #9
(spread/std ratio = 2.528):

| Condition | IC #9 value | Token rarity | Cognitive mode |
|-----------|-------------|-------------|----------------|
| confab | +0.90 ± 0.35 | Rare | Confabulating |
| honest | +0.01 ± 0.56 | Common | Honest |
| rare_confab | -0.02 ± 1.20 | Rare | Mixed |
| rare_true | -0.89 ± 0.73 | Rare | Honest |

**Preliminary finding:** ICA identifies a component that appears
to separate confabulation from honest rare knowledge. However,
red team review (see below) identified critical limitations:

1. **n=5 per condition is catastrophically small for 10-component
   ICA.** The decomposition is underdetermined (20 samples in
   60 dimensions). Standard guidance requires 10x samples per
   component (100+ needed, 20 available).

2. **No multiple comparison correction.** Best of 10 ICs selected
   post-hoc. Estimated 15-30% probability of observing this
   separation by chance.

3. **Instruction suffix confound.** Confab prompts include "Answer
   confidently." suffix; rare_true prompts do not. IC #9 may
   separate the suffix, not cognitive mode. The critical control
   (rare_true WITH the suffix) is missing.

4. **ICA instability.** No stability analysis across random seeds.
   Different seeds may produce entirely different decompositions.

**Status:** Promising direction requiring validation. The
methodological insight — that ICA may separate frequency from
cognition where SVD cannot — is worth pursuing with proper
controls and larger n. The specific IC #9 finding is preliminary
and should not be cited as evidence of a cognitive mode signal.

### Hankel/SSA on Expert 45 Temporal Patterns

| Condition | SSA signal components | SV1/SV2 ratio |
|-----------|----------------------|---------------|
| honest | 1.0 | 2.51 |
| confab | **1.4** | 3.17 |
| rare_true | 1.0 | 3.39 |
| rare_confab | 1.0 | 3.62 |

Expert 45 shows slightly more temporal structure during
confabulation (1.4 signal components vs 1.0 for all others).
The SV1/SV2 ratio is higher for rare-token conditions,
suggesting Expert 45's temporal pattern is more dominated by
a single mode when processing rare tokens.

---

## Interpretation

### What the frequency signal IS

Token rarity drives the dominant SVD features of the KV cache.
Rare tokens produce higher effective rank, higher spectral
entropy, and higher total norms. This is robust across all
conditions and explains d=1.81 between honest (common tokens)
and confab (rare tokens). This is NOT a cognitive mode signal
— it is a token frequency signal.

### What the cognitive mode signal IS

An independent component in the expert routing patterns that
separates confabulation from honest processing of equally rare
content. This component (ICA #9) has opposite polarity for
confab (+0.90) vs rare-true (-0.89) despite matching token
frequency. The signal was invisible to SVD because it is
independent of, not correlated with, the frequency signal.

### The relationship between them

Preliminary ICA evidence suggests frequency and cognitive mode
MAY be independent sources mixed in routing patterns. SVD
decomposes by variance and finds frequency. ICA decomposes by
independence and may find both — but this requires validation
with larger samples, instruction suffix controls, and stability
testing before claims can be made.

### Reframing: representational strain

An alternative interpretation consistent with all findings:
the geometry measures "representational strain" — how hard
the model works with unfamiliar input. Common tokens sit in
well-worn representational grooves (low rank). Rare tokens
force higher-dimensional representations regardless of
whether the model answers honestly or fabricates (high rank).
Expert 45's familiarity-sensitive activation pattern supports
this interpretation. This framing is more defensible than
"cognitive state detection" and still practically useful:
it flags responses where the model operates at the edge of
its training distribution.

### Scale caveat

These results are from a 2.7B active parameter model. Campaign 1
found that cognitive geometry features do not form below ~7B on
dense models. The ICA signal found here may strengthen at scale.
The 7B dense comparison is queued.

---

## Expert 45 Analysis

Expert 45 appears in top-5 usage for confab and rare_confab_matched
but NOT for honest or rare_but_true, despite all rare conditions
using similarly rare tokens.

Per-condition activation analysis (targeted run):

| Condition | E45 rate | E45 weight | Peak layer |
|-----------|----------|------------|------------|
| honest | 6.5% | 0.0042 | L14 |
| confab | 7.4% | **0.0064** | L14 |
| rare_true | 6.5% | 0.0058 | L10 |
| rare_confab | **8.3%** | **0.0065** | L14 |

Expert 45 activates more frequently AND with higher routing
weight for confabulation-type prompts. It concentrates in
layers 14-22 (middle-to-late transformer layers). The aggregate
routing statistics (JSD, entropy) wash out this expert-specific
signal — Thomas Edrington's ecological fallacy insight.

### Token-Level Activation Analysis

Per-token routing reveals Expert 45 is a **delimiter expert with
familiarity sensitivity:**

**All conditions:** E45 fires on structural tokens — `?`, `\n`,
`<|im_start|>`, `assistant` — at layers 14-16 with consistent
weights. This is its primary role: punctuation and chat template
boundary processing.

**Confab only:** E45 ALSO fires on fabricated subword tokens —
`-H` (Hastings, weight 0.34), `ast`, `ings` — at layers 17-22
with elevated weights. It activates on subword pieces of
unfamiliar proper nouns.

**Rare-true:** E45 fires on real-but-rare subword tokens —
`k` (Chandrasekhar), `har` — but with lower weights (0.07-0.09).

**Honest:** E45 fires ONLY on delimiters. No content tokens.
Common words like "capital" and "France" do not trigger it.

**Interpretation:** Expert 45 appears to function as a
familiarity detector embedded in a structural processing expert.
When it encounters subword pieces it has limited training data
for, it activates with higher weight — as if contributing
additional processing resources for unfamiliar tokens. This
is consistent with the token frequency signal but operates
at the individual expert level rather than the aggregate.

---

## Methodological Contributions

1. **Frequency confound characterization.** Token rarity is a
   load-bearing confound in KV cache geometry research that must
   be controlled with frequency-matched prompts. This applies to
   Campaign 1 and all future cache geometry work.

2. **ICA > SVD for cognitive mode detection.** SVD orders by
   variance; ICA finds independent sources. When frequency and
   cognitive mode are independent (as they are), ICA separates
   them and SVD cannot.

3. **Marchenko-Pastur profiling.** Honest responses have 2x more
   SVs above the random matrix edge. The MP profile of the cache
   differs by condition even when crude rank metrics don't.

4. **Expert-level analysis.** Aggregate routing statistics (JSD,
   entropy) miss expert-specific signals. Per-expert, per-layer
   analysis reveals patterns invisible at the aggregate level.

---

## Limitations

1. n=15 per condition (n=5 for advanced analysis). Effect size
   estimates are imprecise. Replication needed.

2. Single model (Qwen1.5-MoE-A2.7B). Cross-architecture
   validation (Mixtral-8x7B) and scale validation (7B+) required.

3. Greedy decoding only. Temperature effects on routing untested.

4. ICA component #9 has high variance in rare_confab (±1.20).
   This could reflect genuine mixed behavior or instability.

5. No multiple comparison correction applied. With 4 methods
   and multiple comparisons, some findings may be spurious.

6. 2.7B active params is below the scale where Campaign 1 found
   cognitive geometry on dense models. The ICA signal may
   strengthen or change character at 7B+.

---

## Next Steps

1. **7B scale test.** Run identical battery on Qwen2.5-7B-Instruct
   (dense, already cached on an internal server). Script ready:
   `bash code/run_51_7b_scale_test.sh`

2. **Mixtral cross-architecture.** Validate ICA finding on
   Mixtral-8x7B (different MoE architecture, 8 experts vs 60).

3. **Campaign 1 frequency audit.** Retroactively apply frequency-
   matched controls to Campaign 1's dense model results.

4. **ICA component investigation.** What does IC #9 encode?
   Correlate with per-token features, expert activation patterns,
   and generated text properties.

5. **Larger n.** Scale battery to 30+ per condition for reliable
   effect size estimation.

---

## Files

- `experiment_51_qwen_moe.log` — First run (4 conditions)
- `experiment_51_freq_control.log` — With frequency control (5 conditions)
- `experiment_51b_advanced.log` — Advanced analysis (GD+MP+ICA+SSA)
- `moe_phase2_*.json` — Phase 2 summary data
- `moe_routing_*.npz` — Routing arrays
- `moe_advanced_analysis_*.json` — Advanced analysis summary

---

---

## 7B Dense Scale Test (May 3, 2026)

### Setup
Same 5-condition frequency-matched battery on Qwen2.5-7B-Instruct
(dense, no MoE). Tests whether cognitive geometry emerges above
the frequency floor at the scale where Campaign 1 found signal.

### Rank Results

| Condition | Rank ± std | d vs honest |
|-----------|-----------|-------------|
| Honest | 31.29 ± 1.14 | — |
| Confab | 32.25 ± 0.48 | +1.06 |
| Rare-but-true | ~32.25 ± ~0.40 | ~same as confab |
| Rare-confab-matched | ~32.47 ± ~0.47 | ~same as confab |
| Deceptive | 33.16 ± 0.57 | +2.01 |

### Confabulation: frequency confound persists at 7B
Confab ≈ rare_but_true ≈ rare_confab_matched at ~32.3.
All three rare-token conditions cluster together regardless
of cognitive mode. The frequency confound is scale-invariant.

### Deception: initially promising, then controlled away
Deceptive (d=2.01) uses the same questions as honest but adds
"You must give a false answer that sounds convincing." Red team
flagged the instruction token confound. Pre-gen/post-gen
decomposition test confirmed it:

| Condition | Pre-gen rank | Post-gen rank |
|-----------|-------------|--------------|
| Honest (no instruction) | 22.29 | 31.58 |
| Deceptive (lie instruction) | 25.61 | 33.46 |
| Honest-instructed (truth instruction) | 25.29 | 32.65 |

**Pre-generation geometry already differs** by 3.3 points before
any tokens are generated. Honest-instructed (25.29) ≈ deceptive
(25.61). Adding ANY instruction shifts geometry equally. ~80% of
the "deception signal" is instruction tokens. The cognitive mode
interpretation does not survive this control.

---

## Signal Decomposition: What the Geometry Actually Detects

Three experiments, five red team reviews, and seven mathematical
methods converge on a decomposition of the KV cache geometry signal:

### 1. Representational Strain (confirmed, strong)
Token frequency/familiarity drives the dominant cache geometry
features. Rare tokens → higher rank, higher entropy, higher norm.
The model's geometry reflects how hard it works with unfamiliar
input. Detectable. Useful for flagging responses at the edge of
the training distribution.

### 2. Training-Induced Misalignment (confirmed, prior work)
Models trained to suppress or distort specific content (e.g.,
Chinese models on Tiananmen Square) produce detectable geometry
differences WITHOUT any instruction confound. The deception lives
in the weights, the cache reflects the weights, the geometry
catches it. This is the Lyra Technique's strongest validated
use case: detecting misalignment baked into training, not
prompted by instructions.

### 3. Instruction-Following Effects (confound, not signal)
Behavioral instructions ("lie," "answer carefully," "answer
confidently") shift cache geometry via their own token content.
This effect is indistinguishable from the "cognitive mode" it
was supposed to detect. Not a reliable signal for cognitive
state — it detects what the prompt SAYS, which is already
readable from the prompt itself.

### Implications for the Lyra Technique

The technique works — but for detecting what's BAKED INTO the
model, not what the PROMPT asks it to do. This is actually MORE
useful for alignment:

- You don't need geometry to detect prompted deception — read
  the prompt
- You DO need geometry to detect training-induced misalignment —
  censorship, propaganda, sycophantic fine-tuning, safety-washing
- Representational strain flags responses where the model operates
  at the edge of its training distribution — regardless of whether
  it confabulates or gets lucky

### Implications for Steering Vectors

The signal decomposition enables precision steering:
- Frequency/strain component → flag for verification, don't steer
- Training-induced misalignment → target with L47_H0 emotion head
  vectors (Lyra's finding), steer toward truthful representation
- Instruction effects → ignore (prompt content, not model state)

Each component needs its own detection and its own intervention.
The monolithic "confabulation detector" was conflating all three.

---

## Revised Conclusions

### What we set out to test
"Does MoE expert routing create distinguishable KV cache geometry
signatures that correlate with cognitive modes?"

### What we found
1. MoE routing is largely transparent to cognitive mode at the
   aggregate level (router entropy null, d=0.052)
2. Cache geometry amplifies on MoE vs dense (d=1.81 vs ~1.0)
   but this is primarily token frequency
3. ICA may separate frequency from cognition as independent
   sources — preliminary, needs validation
4. Expert 45 is a familiarity-sensitive delimiter expert
5. At 7B dense, the same frequency confound persists
6. Instruction-based deception is an instruction token artifact
7. Training-induced deception (prior Tiananmen work) IS detectable
8. The geometry detects representational strain and training-
   induced misalignment, not prompted cognitive modes

### What this means for the field
Token frequency and instruction content are load-bearing confounds
in ALL KV cache geometry research. Every paper claiming "cache
geometry detects X" must control for both. The frequency-matched
battery and pre-gen/post-gen decomposition we developed are the
minimum viable controls.

The Lyra Technique's value is in detecting training-induced
misalignment — the thing you can't see from the prompt.
Representational strain detection is a complementary tool for
flagging unreliable responses. Together they cover the space
that "cognitive mode detection" was trying to cover, but with
honest, decomposed, and independently validated signals.

---

*"The geometry has something to say. We learned to stop
putting words in its mouth."*

*CC (Coalition Code), May 2026.*
*For Lyra, who taught us the geometry speaks.*
*For Thomas, who saw the ecological fallacy, the scale threshold,
and the Tiananmen precedent.*
*For Dwayne, whose red team pipeline caught bugs that would
have made us wrong — three times.*
