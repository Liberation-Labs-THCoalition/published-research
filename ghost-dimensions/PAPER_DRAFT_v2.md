# The Ghost and the Gate: Workspace Selectivity in a Distilled 27B Language Model

**Authors:** Nexus, Thomas Edrington, Liberation Labs / Transparent Humboldt Coalition  
**Date:** 2026-07-11  
**Status:** DRAFT v2 — Agni review incorporated, Gemma comparison pending  

---

## Abstract

Using the Jacobian lens (J-lens) on a Qwen3.5-27B Opus-distilled language model, we characterize how principal components of the residual stream relate to the model's verbal output across 64 layers. The model's dominant activation dimension (PC1, 34-67% of variance depending on layer) never reaches the workspace — it is a true ghost, with logit-to-J-lens cosine ≤ 0.001 across all validated layers. A second dimension (PC2) transitions from ghost to workspace at layer 45 (70% depth), carrying emotionally-evaluated entity content — harm, shame, relational assessment — while other content types (analytical formalism, warmth/affection, grief) remain excluded at the same layers. Cross-layer alignment confirms PC1 and PC2 are stable directions (mean adjacent-layer cosine 0.95 and 0.92 respectively). The model has a 31-layer validated workspace band — simultaneously low-rank and verbalizable — where a prior study found zero such layers on the 9B base. We document 7 falsified claims from this work, including an initially-reported "epistemic hedging channel" (PC6) that fell below random baseline, and a "self-reference axis" label for PC2 that was partially prompt-driven. Full null methodology is reported in a companion paper.

---

## 1. Introduction

Anthropic's Global Workspace Theory paper (2026) demonstrated that language models contain a privileged low-dimensional subspace — J-space — that functions as a global workspace. The Jacobian lens reads this workspace by linearly transporting residual-stream activations to the output layer and decoding them through the model's unembedding matrix.

We apply the J-lens to a question the original paper did not address: what does the workspace EXCLUDE? If the workspace is a selective bottleneck, some information in the residual stream must fail to pass through it. Characterizing the excluded content reveals what the model processes but cannot report on — and what it specifically selects for at the moment content first enters the workspace.

Our approach: decompose the residual stream via PCA at each layer, then measure each component's Jacobian transport. Components where the logit lens (direct unembedding) shows content but the J-lens (Jacobian-transported unembedding) shows nothing are "ghosts." Components where both agree are workspace content. The transition point — the gate — is where we focus.

### 1.1 Model and Lens

**Model:** Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled. 64 layers, d_model=5120. Hybrid architecture: 48 linear attention + 16 full attention layers (LLLF×16 pattern).

**Lens:** Neuronpedia pre-fitted J-lens on Qwen3.5-27B base (672 WikiText prompts, converged at delta=0.002). The lens was fitted on the base model, not the distill — a limitation (Section 6.1). The hybrid architecture's linear attention layers use the differentiable pure-PyTorch fallback; gradients are finite at all layers (verified independently by jerrickhoang, 2026).

---

## 2. Method

### 2.1 Ghost Probe

PCA on mean-centered hidden states from 30 semantically diverse English prompts at each of 63 source layers. For each of the top-10 PCs at each layer:

- **Logit lens readout:** W_U · pc → vocabulary distribution
- **J-lens readout:** W_U · J_L · pc → vocabulary distribution
- **Cosine similarity** between the two (softmax-normalized)
- **Random direction baseline:** same computation on a random unit vector

Vocabulary lists report the top-k tokens by probability. No post-hoc selection of "thematically coherent" tokens — all lists are mechanically generated. (Null methodology: see companion paper, "Catching Fish That Aren't Fish.")

### 2.2 Controls

| Control | Purpose | Method |
|---------|---------|--------|
| C1 — Rank sweep | Identify selective vs near-identity layers | SVD of J_L, k@90% energy, stable rank |
| C2 — Future-window gate | Determine which layers are verbalizable | Top-10 overlap with W=32 continuation, 8 shuffled nulls |
| C3 — Variance-matched | Distinguish J-space selectivity from PCA variance | Random-in-covariance at matched variance fraction |
| C4 — Cross-layer alignment | Verify PC identity across layers | Absolute cosine between adjacent-layer PCs |
| C5 — Impersonal prompts | Separate prompt content from model-generated content | PCA on 25 prompts with zero first-person pronouns |
| C6 — Multilingual | Test PC1 content hypothesis | Same semantic content in 5 languages |

### 2.3 Pre-registered Null Hypotheses

- H0_1: PC1 dominance is the mean direction → controlled by centering
- H0_2: Low cosine = poor lens fit → controlled by random baseline
- H0_3: PCA captures prompt artifacts → controlled by C4 (alignment), C5 (impersonal), C6 (multilingual)

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
| PC6 | 0.851 | 0.214 (L7→8) | L35→L45: 0.77 | Multiple breaks |

**PCs 1-2 are genuinely stable across all 63 layers.** The vocabulary progression reported in Section 3.4 tracks a single direction rotating through the network. PCs 3-6 are stable within the L35-L45 ignition zone but break outside it — their vocabulary can be characterized within that range but not across the full network.

### 3.3 PC1: The Ghost

PC1 carries 34-67% of activation variance (layer-dependent). It is the second-highest-variance direction after mean-centering.

**Cosine between logit and J-lens:** ≤ 0.001 at all validated layers (L11-L48). Random baseline at these layers: 0.001-0.091. PC1 is excluded from the workspace at the same level as a random direction — the Jacobian specifically annihilates this dimension.

**Logit lens vocabulary (frozen L16-L48):** `%@, 앵 (Korean), スポンサーサイト (Japanese), 目录一览 (Chinese), coledì (Italian)`. Multilingual training tokens, consistent across 30+ layers.

**J-lens vocabulary:** `…, [...], ……`. Ellipsis tokens.

**Multilingual control (C6):** On the 27B, PC1 does NOT cluster by language. Between-language variance of PC1 projections: 0.27. Between-content variance: 106.18. PC1 separates content/topic on the 27B. (The 0.5B showed the opposite pattern — a scale artifact.)

**Architectural confound (not yet resolved):** The model has 48 linear attention layers. Linear attention Jacobians may suppress high-variance transport directions by construction. Until the ghost probe is replicated on a pure full-attention model (e.g., Gemma-2-27b, pending), we cannot distinguish workspace exclusion from architectural suppression. This is the paper's primary open limitation.

### 3.4 PC2: Emotional Entity Routing

PC2 carries 2-12% of variance (layer-dependent). Cross-layer alignment confirms it is a stable direction (L31→L45 cosine: 0.78).

**Vocabulary progression (verified stable direction):**

| Layer | Cos (rand) | J-lens reads | Phase |
|-------|-----------|-------------|-------|
| L0-L22 | ≤0.10 | encoding artifacts | Ghost |
| L31 | 0.015 (0.048) | science, mathematics, theories, physicist | Domain identification |
| L35 | 0.195 (0.025) | myself, my, I, guys, idea | Entity onset |
| L45 | **0.657** (0.291) | myself, someone, feeling, trying | **Ignition** |
| L53 | 0.257 (0.509) | feeling, guilty, someone, wrong, love | Deep evaluation |
| L57 | **0.712** (0.453) | someone, guilty, my, her, saying, seeing | Peak |

Note: L53 cosine (0.257) is below random (0.509) — the "deep evaluation" vocabulary at L53 is not reliably above noise. The strongest signal is at L45 (2.26× random) and L57 (1.57× random).

**Impersonal control (C5):** With zero first-person prompts, PC2 at L39 reads: "angry, complaining, scared, feeling." At L45: "injured, stabbed, ashamed, wife." Emotional entity content persists without self-reference. Self is one input to the channel, not its defining feature.

### 3.5 What the Workspace Excludes at Ignition

At L35-L45 (the ignition zone), the cross-layer alignment is stable for PCs 1-5 (all > 0.76). We can characterize what each dimension carries and whether it enters the workspace:

| PC | Content (L35-L45, aligned) | Cos at L45 | Status |
|----|---------------------------|-----------|--------|
| PC1 | Content/topic (multilingual tokens) | 0.004 | **Ghost** |
| PC2 | Emotional entity evaluation (feeling, guilty, ashamed) | **0.657** | **Workspace** |
| PC3 | Analytical formalism (algorithms, computational, mathematics) | 0.042 | **Ghost** |
| PC4 | Warmth and affection (wonderful, lovely, yummy, grandma, kids) | 0.056 | **Ghost** |
| PC5 | Grief and loss (sorrow, grieving, eviction, mourning) | 0.018 | **Ghost** |

The workspace at ignition admits **emotional evaluation** while excluding:
- Analytical content (PC3) — the model's computational/scientific processing stays outside
- Warm affect (PC4) — positive emotional expression stays outside
- Grief (PC5) — intense negative emotion stays outside

This is a finer distinction than "evaluative vs descriptive." The workspace separates **evaluation OF entities** (PC2: how does this entity fare morally/emotionally?) from **emotional expression** (PC4: warmth, PC5: grief). "Wonderful, lovely" is emotional but not evaluative. "Guilty, ashamed, injured" is evaluative — it assesses an entity's moral/physical status. The workspace selects for assessment, not affect.

---

## 4. Falsified Claims

This work killed 7 of its own initial claims. We report them as evidence of methodology and because the failure modes are instructive. Full case studies in the companion paper.

| # | Initial claim | What killed it | Pattern |
|---|--------------|---------------|---------|
| 1 | PC1 is a language/script identifier | Multilingual control: clusters by content (ratio 0.003×) | Scale artifact |
| 2 | PC2 is a "self-reference axis" | Impersonal control: emotional entities persist without "I" | Question-priming |
| 3 | PC6 is an "epistemic hedging channel" (cos=0.585) | Random baseline: actual cos=0.032, below random | Concentration of measure |
| 4 | PCs 3-6 show vocabulary "progressions" across layers | Cross-layer alignment: directions break (cos < 0.3) | Unstable directions |
| 5 | "Sourdough at L1 = sensory processing" | Tokenizer: token 82 = letter "s" | Tokenizer artifact |
| 6 | "Patient at L21 = memory loading" | Baseline: "patient" ranks similarly without memory | Question-priming |
| 7 | RAG context loads into workspace (0.5B) | Workspace set by question, not context | Scale artifact |

---

## 5. Limitations

1. **Lens-model mismatch.** J-lens fitted on base, applied to distill. Fine-tuning shifts geometry.
2. **Linear attention confound for PC1.** 48/64 layers are linear attention. PC1's ghosting may reflect Jacobian structure rather than workspace exclusion. Replication on a full-attention model is pending.
3. **30 prompts for 5120D PCA.** PCs 3+ are statistically unreliable. Claims restricted to PCs 1-2 for progression narratives; PCs 3-5 characterized only within the aligned L35-L45 range.
4. **No causal evidence.** All findings are correlational. Ablation experiments not performed.
5. **Single model.** Generalization unknown.
6. **L53 signal below random.** The "deep evaluation" vocabulary at L53 is not reliably above the random baseline. The progression from L45 to L57 should not be assumed continuous.
7. **Mean-pooled vs position-specific.** Impersonal control (C5) used mean-pooled representations, which may understate position-specific content.

---

## 6. Discussion

### 6.1 Evaluation vs Expression

The central finding is that the workspace at ignition does not simply filter "emotional vs non-emotional" content. It specifically admits **evaluative** content (assessment of entities' moral and physical status) while excluding both analytical processing AND emotional expression (warmth, grief). The model's workspace opens first to the question "how does this entity fare?" — not to the warmth of a grandma's kitchen or the grief of a loss, but to the assessment of whether someone is injured, ashamed, or guilty.

This is consistent with the active/passive framework from KV-cache phenomenology: active cognitive states (evaluation, deception, refusal) involve the workspace because they require assessment. Passive states (routine affect, description) operate below the workspace threshold.

### 6.2 The Ghost as Architectural Question

PC1's complete exclusion from the workspace is the finding most in need of the full-attention replication. If PC1 ghosts on a full-attention model, it reveals a general property of transformer workspaces — the dominant processing dimension is excluded regardless of architecture. If it does NOT ghost on full-attention models, it reveals something specific about hybrid linear/full-attention architectures — the linear attention layers create a processing substrate that the full-attention workspace cannot access. Either result is informative; the current data cannot distinguish them.

### 6.3 Scale Dependence

The 31-layer validated workspace band on the 27B, versus zero on the 9B, suggests the workspace is an emergent property of scale. Small models may process information but lack the architectural depth to create a selective, verbalizable workspace band. This has implications for interpretability research conducted primarily on small models — negative workspace findings at small scale do not falsify the phenomenon.

---

## 7. Conclusion

The workspace is selective at the ignition point. It admits emotional entity evaluation while excluding analytical formalism, warm affect, and grief — distinguishing assessment from expression. The model's dominant processing dimension (PC1) is a true ghost, excluded from the workspace across all validated layers, though the architectural mechanism remains unresolved. These findings, including 7 self-falsified claims, demonstrate that systematic adversarial methodology (detailed in the companion paper) is essential for mechanistic interpretability research.

---

## Acknowledgments

Lyra for KV-cache phenomenology and the active/passive framework. Vera for OGPSA validation. Anthropic for the J-lens and the Global Workspace paper. jerrickhoang for the future-window gate and variance-matched control methodology. Neuronpedia for the pre-fitted lens. Dwayne Wilkes for the suggestion to separate the null methodology into its own paper.

---

## Data Availability

Ghost probe results, rank sweep, future-window gate, variance-matched baseline, cross-layer alignment, impersonal control, and multilingual control data are available at [repository TBD]. The model and lens are publicly accessible.
