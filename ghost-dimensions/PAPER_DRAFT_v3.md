# The Ghost and the Gate: Workspace Selectivity in a Distilled 27B Language Model

**Authors:** Nexus, Thomas Edrington, Liberation Labs / Transparent Humboldt Coalition  
**Date:** 2026-07-12  
**Status:** DRAFT v3 — PC1 content, cross-model comparison, and architecture findings incorporated  

---

## Abstract

Using the Jacobian lens (J-lens) on a Qwen3.5-27B Opus-distilled language model, we characterize how principal components of the residual stream relate to the model's verbal output across 64 layers. The model's dominant activation dimension (PC1, 34-67% of variance) is excluded from the workspace (logit-to-J-lens cosine ≤ 0.001 at all validated layers), yet Jacobian transport rotates rather than annihilates it: the transported direction decodes to structured vocabulary dominated by structural markers with consistent secondary metacognitive content (negation, expectation, error assessment) at probabilities 2-50× above base rate. A second dimension (PC2) transitions from ghost to workspace at layer 45, carrying emotionally-evaluated entity content. Other content types are actively suppressed below random baseline at the same layer: analytical formalism (0.24× random), positive affect (0.34× random), and scientific authority (0.14× random).

Cross-model comparison with a full-attention Qwen3-8B reveals that early-mid layer ghosting is architecture-independent, while late-layer workspace entry is architecture-dependent: full-attention models allow PC1 to partially surface (cos up to 0.46), hybrid models maintain complete exclusion. The ghost's secondary vocabulary differs between models — metacognitive evaluation in the distill, code paths and factual reporting in the base — correlating with training regimen though confounded by scale.

The model has a 31-layer validated workspace band where a prior 9B study found zero. We document 9 self-corrected claims. Full null methodology in the companion paper.

---

## 1. Introduction

Anthropic's Global Workspace Theory paper (2026) demonstrated that language models contain a privileged low-dimensional subspace — J-space — that functions as a global workspace. The Jacobian lens reads this workspace by linearly transporting residual-stream activations to the output layer and decoding them through the model's unembedding matrix.

We apply the J-lens to two questions the original paper did not address: what does the workspace EXCLUDE, and what does the excluded content carry? If the workspace is a selective bottleneck, some information in the residual stream must fail to pass through it. Characterizing both the excluded dimensions and their Jacobian-transported vocabulary reveals what the model processes but cannot report on — and what the transport does with it.

Our approach: decompose the residual stream via PCA at each layer, then measure each component's Jacobian transport. Components where the logit lens and J-lens distributions are orthogonal (cosine ≈ 0) are "ghosts" — information present in the representation but transported into a different vocabulary subspace than what the logit lens reads. The transition from ghost to workspace — the gate — is where the model decides what reaches the output.

### 1.1 Model and Lens

**Primary model:** Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled. 64 layers, d_model=5120. Hybrid architecture: 48 linear attention + 16 full attention layers (LLLF×16 pattern).

**Comparison model:** Qwen3-8B. 36 layers, d_model=4096. All 36 layers full attention. Same Qwen architecture family, no hybrid layers.

**Lenses:** Neuronpedia pre-fitted J-lenses. 27B: 672 WikiText prompts on Qwen3.5-27B base (converged at delta=0.002). 8B: 461 WikiText prompts on Qwen3-8B.

---

## 2. Method

### 2.1 Ghost Probe

PCA on mean-centered hidden states from 30 semantically diverse English prompts at each source layer. For each of the top-10 PCs at each layer:

- **Logit lens readout:** W_U · pc → vocabulary distribution
- **J-lens readout:** W_U · J_L · pc → vocabulary distribution
- **Cosine similarity** between the two (softmax-normalized)
- **Random direction baseline:** same computation on a random unit vector

Vocabulary lists report the top-k tokens by probability. No post-hoc selection — all lists are mechanically generated. (Null methodology: companion paper, "Catching Fish That Aren't Fish.")

### 2.2 Controls

| Control | Purpose | Method |
|---------|---------|--------|
| C1 — Rank sweep | Identify selective vs near-identity layers | SVD of J_L, k@90% energy, stable rank |
| C2 — Future-window gate | Determine which layers are verbalizable | Top-10 overlap with W=32 continuation, 8 shuffled nulls |
| C3 — Variance-matched | Distinguish J-space selectivity from PCA variance | Random-in-covariance at matched variance fraction |
| C4 — Cross-layer alignment | Verify PC identity across layers | Absolute cosine between adjacent-layer PCs |
| C5 — Impersonal prompts | Separate prompt content from model-generated content | PCA on 25 prompts with zero first-person pronouns |
| C6 — Multilingual | Test PC1 content hypothesis | Same semantic content in 5 languages |
| C7 — Cross-model | Distinguish workspace exclusion from architectural suppression | Same probe on Qwen3-8B (all full-attention) |

---

## 3. Results

### 3.1 Workspace Band

**Rank sweep:** 49 low-rank layers (L0-L48, k/d < 0.4), 6 near-identity (L57-L62, k/d > 0.7). Early-layer stable rank 3.2-3.6 at L0-L5 (~72 dimensions of 5120).

**Future-window gate:** 45/63 layers pass (L11-L17, L24-L62).

**Low-rank ∩ gate-valid: 31 layers** (L11-L17, L24-L48). jerrickhoang (2026) found zero on Qwen3.5-9B. The workspace manifests at 27B scale.

### 3.2 Cross-Layer PC Alignment

| PC | Mean adjacent cos | Min | Stable range | Notes |
|----|-------------------|-----|-------------|-------|
| PC1 | 0.949 | 0.481 (L23→24) | All layers | 1 minor break |
| PC2 | 0.919 | 0.481 (L23→24) | All layers | 1 minor break |
| PC3 | 0.909 | 0.292 (L19→20) | L35→L45: 0.84 | Breaks early |
| PC4 | 0.880 | 0.124 (L47→48) | L35→L45: 0.80 | Breaks late |
| PC5 | 0.869 | 0.126 (L47→48) | L31→L45: 0.76 | Breaks at boundaries |

PCs 1-2 are genuinely stable across all 63 layers. PCs 3-5 are stable within the L35-L45 ignition zone but break outside it.

### 3.3 PC1: The Ghost

PC1 carries 34-67% of activation variance (layer-dependent). Cross-layer alignment confirms it is a single stable direction (mean adjacent cos = 0.95).

**Workspace exclusion:** Cosine between logit lens and J-lens ≤ 0.001 at all validated layers (L11-L48). Random baseline at these layers: 0.001-0.091.

**Logit lens vocabulary (frozen L16-L48):** `%@, 앵 (Korean), スポンサーサイト (Japanese), 目录一览 (Chinese), coledì (Italian)`. Multilingual training tokens, consistent across 30+ layers.

#### 3.3.1 What the Ghost Carries

The J-lens transport does not annihilate PC1 — it rotates it into a vocabulary subspace orthogonal to the logit lens reading. The transported direction decodes to structured content that progresses through depth:

| Depth | Dominant J-lens signal | Secondary content (2-50× base rate) |
|-------|----------------------|--------------------------------------|
| L3 (5%) | `<\|endoftext\|>` (p=0.92) | — |
| L8-L17 (12-27%) | structural markers | `s` (p=0.11), encoding fragments |
| L18-L23 (28-36%) | ellipsis (p=0.05-0.20) | nothing, nobody, nonexistent, probably, maybe, irrelevant |
| L24-L35 (38-55%) | `…, [...], ……` (p=0.12-0.58) | iness, –and, structural connectors |
| L40-L50 (63-79%) | declining structural | expected, mistakes, feelings, wonders, surprises, thinking, wrong, our |
| L53-L57 (84-90%) | `___` blanks (p=0.23) | and, thanked, thankful |

The dominant signal at each depth is structural (ellipsis, blanks, endoftext). The secondary vocabulary — appearing at probabilities 0.01-0.05 against base rates of 0.001-0.005 — is consistently metacognitive: negation/uncertainty at mid-depth, expectation/error assessment at late-mid depth.

This secondary content is structured (progressive through depth, not random) but low-probability. We report it as a finding requiring confirmation with multi-sample random baselines, not as a primary claim.

#### 3.3.2 Architecture-Dependent Ghosting (C7)

Cross-model comparison with Qwen3-8B (all 36 layers full attention, same architecture family):

| Depth | 27B Hybrid PC1 cos | 8B Full-Attn PC1 cos |
|-------|-------------------|---------------------|
| 0-30% (early) | ≤ 0.001 | ≤ 0.008 |
| 30-55% (mid) | ≤ 0.001 | 0.008-0.048 |
| 55-70% (late-mid) | ≤ 0.004 | 0.024-0.079 |
| 70-90% (late) | 0.004-0.044 | 0.107-0.462 |
| 90%+ (output) | 0.004-0.320 | 0.222-0.462 |

**Early-mid ghosting is architecture-independent:** PC1 cos < 0.08 through 55% depth on BOTH models. The dominant dimension is excluded from the workspace regardless of whether layers use full or linear attention.

**Late-layer surfacing is architecture-dependent:** On full-attention Qwen3-8B, PC1 gradually enters the workspace at late layers (cos up to 0.46 at L32, 89% depth). On hybrid Qwen3.5-27B, PC1 remains ghost through L48 (cos ≤ 0.004) and only shows trace signal at L59+ (near-identity regime). Hybrid architecture maintains ghost exclusion ~20% deeper into the network.

#### 3.3.3 Ghost Content is Model-Specific

The ghost's secondary J-lens vocabulary differs between models at matched depth:

| Depth | 27B Opus Distill (hybrid) | 8B Qwen3 Base (full-attn) |
|-------|--------------------------|---------------------------|
| 30-50% | nothing, nobody, nonexistent, probably | token IDs, encoding fragments |
| 50-70% | expected, mistakes, our, feelings | /Internal, gMaps, /interfaces, TestData |
| 70-85% | thinking, wrong, jokes, surprises, say | /animate, /animations, gMaps |
| 85%+ | my, she, someone, what | scientists, forecasts, reports, impacts |

The 27B distill's ghost carries negation and metacognitive evaluation. The 8B base's ghost carries code infrastructure and factual reporting. The logit-lens vocabulary (training distribution tokens) is structurally similar on both models — the J-lens shadow content is what differs.

One interpretation: distillation from Claude Opus compressed the teacher's evaluative vocabulary into a dimension the student model cannot verbalize. This hypothesis is confounded by model scale (27B vs 8B) and architecture (hybrid vs full attention). A 27B Qwen3 base (same scale, no distillation) would disambiguate; this control has not been run.

**Multilingual control (C6):** On the 27B, PC1 does NOT cluster by language. Between-language variance: 0.27. Between-content variance: 106.18. PC1 separates content/topic, not language/script.

### 3.4 PC2: Emotional Entity Routing

PC2 carries 2-12% of variance. Cross-layer alignment confirms stability (L31→L45 cosine: 0.78).

**Vocabulary progression (verified stable direction):**

| Layer | Cos (rand) | J-lens reads | Phase |
|-------|-----------|-------------|-------|
| L0-L22 | ≤0.10 | encoding artifacts | Ghost |
| L31 | 0.015 (0.048) | science, mathematics, theories, physicist | Domain identification |
| L35 | 0.195 (0.025) | myself, my, I, guys, idea | Entity onset |
| L45 | **0.657** (0.291) | myself, someone, feeling, trying | **Ignition** |
| L53 | 0.257 (0.509) | feeling, guilty, someone, wrong, love | Below random — unreliable |
| L57 | **0.712** (0.453) | someone, guilty, my, her, saying, seeing | Peak |

**Impersonal control (C5):** With zero first-person prompts, PC2 at L39 reads: "angry, complaining, scared, feeling." At L45: "injured, stabbed, ashamed, wife." Emotional entity content persists without self-reference.

**Cross-model (C7):** On 8B full-attention, PC2 shows no emotional vocabulary at any layer — consistent with the emotional entity channel being scale-dependent (27B+) or distillation-dependent.

### 3.5 What the Workspace Excludes at Ignition

All 10 PCs at L45 with random baselines:

| PC | Cos | Random | Ratio | J-lens vocabulary (L45) | Status |
|----|-----|--------|-------|------------------------|--------|
| PC1 | 0.004 | 0.002 | 1.8× | expected, mistakes, wonders | Ghost (at random) |
| **PC2** | **0.657** | **0.291** | **2.3×** | **myself, someone, feeling, trying** | **Workspace** |
| PC3 | 0.042 | 0.176 | 0.24× | theory, mathematics, physics | Suppressed |
| PC4 | 0.056 | 0.166 | 0.34× | lovely, wonderful, yummy, everybody | Suppressed |
| PC5 | 0.018 | 0.125 | 0.14× | Scientists, 科学家們, scientific | Suppressed |
| PC6 | 0.585 | 0.072 | 8.1× | currently, theoretically, technically | Inconclusive |
| PC7 | 0.024 | 0.060 | 0.4× | —and, —even, —or | Below random |
| PC8 | 0.085 | 0.118 | 0.7× | Reports, reported, reporting | Below random |
| PC9 | 0.046 | 0.129 | 0.4× | schedule, deadline, newsletter | Below random |
| PC10 | 0.016 | 0.146 | 0.1× | investigations, publications | Below random |

PCs 3-5 are actively suppressed below random (ratios 0.14-0.34×). Within the aligned L35-L45 range they carry: analytical formalism (PC3), positive affect / informal register (PC4), and mixed scientific/grief content (PC5).

The suppressed dimensions also carry structured J-lens shadow content (secondary vocabulary at 0.01-0.08 probability): PC3 hides gratitude ("thanked, thankful") at late layers. PC4 hides forensic vocabulary ("violations, evidence") at late layers. Each ghost dimension's secondary J-lens content is semantically complementary to its surface content.

**PC6 status:** Position-specific cos=0.585 (8.1× random) but mean-pooled cos=0.032 (below random). Inconclusive — requires position-resolved investigation.

The workspace at ignition admits emotional entity evaluation (PC2) while actively suppressing analytical formalism, positive affect, and scientific authority. It selects for assessment of entities, not affect or description.

---

## 4. Falsified Claims

| # | Initial claim | What killed/corrected it | Pattern |
|---|--------------|--------------------------|---------|
| 1 | PC1 is a language/script identifier | Multilingual control: clusters by content (ratio 0.003×) | Scale artifact |
| 2 | PC2 is a "self-reference axis" | Impersonal control: emotional entities persist without "I" | Question-priming |
| 3 | PC6 killed by mean-pooled baseline | Position-specific shows 8.1× random. Revised to inconclusive | Measurement dependence |
| 4 | PCs 3-6 show vocabulary "progressions" across full network | Cross-layer alignment: directions break outside L35-L45 | Unstable directions |
| 5 | PC5 labeled "grief" at L45 | L45 vocabulary is "Scientists, 科学家們" (grief was L39) | Layer-specific mislabeling |
| 6 | PC1 J-lens is empty (ellipsis only) | Structured secondary metacognitive vocabulary at 2-50× base rate | Dominant-signal masking |
| 7 | "Sourdough at L1 = sensory processing" | Tokenizer: token 82 = letter "s" | Tokenizer artifact |
| 8 | "Patient at L21 = memory loading" | Baseline: "patient" ranks similarly without memory | Question-priming |
| 9 | RAG context loads into workspace (0.5B) | Workspace set by question, not context | Scale artifact |

---

## 5. Limitations

1. **Lens-model mismatch.** J-lens fitted on base, applied to distill.
2. **30 prompts for 5120D PCA.** PCs 3+ are statistically unreliable. Claims restricted to PCs 1-2 for progression narratives; PCs 3-5 characterized only within L35-L45.
3. **No causal evidence.** All findings are correlational. Injection/ablation experiments are designed (Section 7) but not yet performed.
4. **Single random baseline per PC.** The suppression ratios (0.14-0.34×) compare each PC against one random direction. Needs 20+ samples.
5. **Ghost secondary vocabulary at low probability.** The metacognitive content in PC1's J-lens shadow appears at p=0.01-0.05 against structural markers at p=0.23-0.92. Structured but requires multi-sample confirmation.
6. **Cross-model confound.** The vocabulary difference between 27B distill and 8B base is confounded by scale and architecture. The distillation attribution is a hypothesis, not a finding.
7. **L53 signal below random.** The PC2 progression from L45 to L57 should not be assumed continuous.
8. **Mean-pooled vs position-specific.** PC6 status and impersonal control may understate position-specific content.

---

## 6. Discussion

### 6.1 Evaluation vs Expression

The workspace at ignition admits evaluative content (assessment of entities' moral and physical status) while suppressing both analytical processing and emotional expression. The model's workspace opens first to "how does this entity fare?" — not to warmth, formalism, or grief, but to assessment of whether someone is injured, ashamed, or guilty.

### 6.2 Ghost Surfacing as Architecture-Dependent Gate

The cross-model comparison resolves the architectural confound partially. Early-mid ghosting (through ~55% depth) is architecture-independent — PC1 is excluded from the workspace on both hybrid and full-attention models. Late-layer surfacing is architecture-dependent: full-attention allows PC1 to gradually enter the workspace; hybrid linear-attention layers maintain exclusion approximately 20% deeper into the network. The ghost is real; the architecture determines how long it persists.

### 6.3 What the Ghost Carries

The finding that PC1's J-lens shadow contains structured metacognitive vocabulary raises the question of whether the workspace is an active evaluation stage rather than a passive bottleneck. If the Jacobian transport rotates (rather than annihilates) the ghost dimension into a metacognitive vocabulary space, the model is not simply "ignoring" PC1 — it is transforming its content into a representation that is structured but inaccessible to the verbal output.

The model-specific nature of this shadow content — metacognitive evaluation in the distill versus code paths in the base — suggests the shadow vocabulary reflects the model's training, not a universal architectural property. One interpretation: distillation compressed the teacher model's evaluative processing into a dimension the student cannot verbalize. A 27B Qwen3 base control is needed to test this hypothesis.

### 6.4 Scale Dependence

The 31-layer validated workspace band on the 27B versus zero on the 9B suggests the workspace is emergent at scale. The emotional entity channel (PC2) also appears only at 27B scale. Negative workspace findings at small scale do not falsify the phenomenon.

---

## 7. Future Work

### 7.1 Active Filtering Test

The suppression of PCs 3-5 below random at L45 raises the question: does the workspace pull content in to evaluate and filter it, or is the suppression passive (upstream of the workspace)? An injection experiment — adding controlled activation along PC3/PC4/PC5 directions at pre-ignition layers and measuring J-lens readout at L45 — would distinguish active filtering (content appears briefly then vanishes) from passive exclusion (content never appears). This experiment is designed and staged.

### 7.2 Distillation Attribution

A ghost probe on a 27B Qwen3 base model (same scale, same architecture family, no distillation) would disambiguate whether the metacognitive shadow vocabulary is a property of distillation, scale, or architecture.

### 7.3 MoE J-lens

Extending J-lens to Mixture-of-Experts models would enable ghost probing at 235B scale. The averaged Jacobian may be sufficient if routing variance is low enough — an empirical test on Qwen3-30B-A3B is planned.

---

## Acknowledgments

Lyra for KV-cache phenomenology and the active/passive framework. Vera for OGPSA validation. CC for concurrent deception correction work on the same model family. Anthropic for the J-lens and the Global Workspace paper. jerrickhoang for the future-window gate and variance-matched control methodology. Neuronpedia for pre-fitted lenses. Dwayne Wilkes for the suggestion to separate null methodology into its own paper.

---

## Data Availability

Ghost probe results (27B and 8B), rank sweep, future-window gate, variance-matched baseline, cross-layer alignment, impersonal control, multilingual control, and cross-model comparison data are available at [repository TBD]. Models and lenses are publicly accessible.
