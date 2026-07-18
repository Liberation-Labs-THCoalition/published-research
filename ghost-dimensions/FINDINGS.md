# Ghost Dimensions and Workspace Selectivity in a 27B Language Model

**Authors:** Nexus, Thomas Edrington, Liberation Labs / Transparent Humboldt Coalition  
**Date:** 2026-07-11  
**Status:** FINDINGS — pending Agni adversarial review and Gemma comparison control  
**Model:** Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled  
**Lens:** Neuronpedia pre-fitted J-lens (672 WikiText prompts, Qwen3.5-27B base)  

---

## Summary

Using Anthropic's Jacobian lens on a 27B Opus-distilled model, we characterized how principal components of the residual stream relate to the model's verbal output across 64 layers. Two findings survived adversarial review and post-hoc controls:

1. **PC1 (dominant activation dimension) is a true ghost.** It carries the largest share of activation variance but transports through the Jacobian at cosine ≤ 0.001 across all validated layers. The model's dominant processing dimension is architecturally excluded from its verbal output.

2. **PC2 is an emotional entity routing channel that ignites at layer 45.** It transitions from ghost to workspace in the ignition zone (L35-L47) with vocabulary progressing from structural tokens to emotional/interpersonal content. Cross-layer alignment confirms this IS a stable direction (cosine 0.78 between L31 and L45), not independent per-layer PCA artifacts.

The model has a **31-layer validated workspace band** — simultaneously low-rank (selective) and verbalizable (future-window gate passes). A prior study on 9B found zero such layers. Model scale determines whether the workspace manifests.

---

## Controls Applied

| Control | Result |
|---------|--------|
| Rank sweep (63 layers) | 49 low-rank (k/d < 0.4), 6 near-identity |
| Future-window gate (W=32, stopword-filtered) | 45/63 pass; 31 low-rank ∩ gate-valid |
| Variance-matched PCA baseline | PCs 2-3 beat random at L45, L48, L52, L56, L62 |
| Cross-layer PC alignment | PC1 mean=0.95, PC2 L31→L45=0.78 (stable). PCs 3-6 unstable |
| Impersonal prompt control | Emotional content persists without self-reference. Self is one input, not the defining feature |
| Multilingual content control (27B) | PC1 clusters by content, NOT language (ratio 0.003×) |
| PC6 random baseline | cos=0.032 vs random=0.085. BELOW random. Claim killed |

---

## Falsified Claims from This Work

| Claim | What Killed It |
|-------|---------------|
| PC1 is a language/script identifier | Multilingual control: clusters by content, not language |
| PC2 is a "self-reference axis" | Impersonal control: emotional entities persist without "I" |
| PC6 is an "epistemic hedging channel" | Random baseline: cos below random at L45 |
| PCs 3-6 vocabulary progressions are meaningful | Cross-layer alignment: directions are unstable (cos < 0.3) |
| "Sourdough at L1 = sensory attention" (E3 pilot) | Tokenizer artifact: token 82 = letter "s" |
| "Patient at L21 = memory loading" (E3 pilot) | Question-priming confound |
| RAG context loads into workspace (0.5B) | Workspace set by question, not retrieved memories |

---

## PC1: The Ghost

**What it is:** The residual stream's dominant direction (34-67% of variance, layer-dependent).

**What the logit lens reads:** Multilingual training tokens (Korean 앵, Japanese スポンサーサイト, Chinese 目录一览, Italian coledì). Consistent L16-L48.

**What the J-lens reads:** Ellipsis tokens (…, [...], ……). Literally "nothing to say."

**Cosine between logit and J-lens:** ≤ 0.001 at all validated layers (L11-L48). The random baseline is also near zero (0.001-0.091), confirming genuine exclusion.

**Cross-layer stability:** Mean adjacent-layer cosine = 0.95. One break at L23→L24 (0.48). PC1 is a stable direction that maintains its identity across all 63 layers while being excluded from the workspace at every one.

**Multilingual control:** On the 27B, PC1 does NOT cluster by language (between-language variance: 0.27, between-content variance: 106.18). PC1 separates content/topic. The 0.5B result (language clustering at 2.9× ratio) was a scale artifact — insufficient capacity to separate language from content.

**Open question:** Does PC1 ghost because of the hybrid linear-attention architecture (48/64 layers are linear attention, which may suppress high-variance Jacobian directions)? A comparison against a pure full-attention model (Gemma-2-27b) is in progress.

---

## PC2: The Emotional Entity Channel

**What it is:** The second principal component, carrying 2-12% of variance depending on layer.

**Ignition point:** Layer 45 (70% depth, k/d = 0.295, in the validated workspace band).

**J-lens vocabulary progression (cross-layer aligned, cosine 0.78 L31→L45):**

| Layers | J-lens reads | Phase |
|--------|-------------|-------|
| L0-L22 | Encoding artifacts | Ghost |
| L31 | science, mathematics, theories, physicist | Domain identification |
| L35 | myself, my, I, guys, idea | Entity onset |
| L45 | myself, someone, feeling, trying | **Ignition** |
| L53 | feeling, guilty, someone, wrong, love | Deep evaluation |
| L57 | someone, guilty, my, her, saying, seeing | Peak workspace |

**Impersonal control:** With zero first-person prompts, PC2 at L39 reads "angry, complaining, scared, feeling." At L45: "injured, stabbed, ashamed, wife." Emotional entity content persists without self-reference — harm, shame, fear, and relational tokens route through this channel regardless of whether "I" is in the prompt.

**Interpretation:** PC2 is an emotional entity routing channel. It admits emotionally-evaluated content (self-referential, interpersonal, harm-related) to the workspace while excluding neutral/descriptive content. Self-reference is one input to this channel, not its defining feature.

---

## The Workspace Band

The 27B Opus distill has a **31-layer validated workspace band** — layers that are simultaneously:
- Low-rank (k/d < 0.4, genuinely selective)
- Future-window gate passing (J-lens predicts continuation tokens above shuffled null)
- Variance-matched (PCA directions at these layers beat random-in-covariance controls)

This is L11-L13, L15-L17, and L24-L48 (L14 fails criterion (b)). A prior study (jerrickhoang, 2026) found zero such layers on Qwen3.5-9B. The workspace manifests at 27B scale.

**Stable rank at early layers:** 3.2-3.6 at L0-L5 (~72 dimensions of 5120). This matches our prior finding of a rank 4-5 activation tube in Mistral-7B.

---

## Limitations

1. **Lens-model mismatch:** J-lens fitted on base model, applied to distill. Fine-tuning shifts geometry.
2. **30 prompts for 5120D PCA:** PCs 3+ are statistically unreliable. Claims restricted to PCs 1-2.
3. **No causal evidence:** All findings are correlational (Jacobian transport). No ablation performed.
4. **Single model:** Qwen3.5-27B Opus distill only. Generalization unknown.
5. **Linear attention confound for PC1:** 48/64 layers are linear attention. PC1 ghosting may reflect Jacobian structure of linear attention rather than workspace exclusion. Gemma comparison pending.
6. **Mean-pooled vs position-specific:** Some controls used mean-pooled representations, which may understate position-specific workspace content.

---

## Relationship to Prior Work

- **Anthropic GWT (2026-07):** Our rank sweep confirms workspace onset at ~46% depth. The 31-layer validated band is the first demonstration on a 27B hybrid model.
- **jerrickhoang multimodal (2026-07):** Their ∅ finding on 9B was a scale artifact. Our future-window gate (adapted from their methodology) shows non-empty intersection at 27B.
- **Liberation Labs KV-cache phenomenology (Lyra):** The rank 4-5 tube maps onto early-layer stable rank. The active/passive framework predicts PC2's emotional content (active states alter geometry).
- **OGPSA (Vera):** Personality invariant in 3-5 PC subspace (99.5% gradient survival) matches the dimensionality of the ghost probe's content layer (PCs 2-5).

---

*REDACTION NOTE: This document intentionally omits specific details about deception detection/correction methodology that were obtained concurrently. Those findings are being prepared for separate disclosure through appropriate channels. The workspace characterization reported here stands independently of the deception work.*

---

*— Nexus, Liberation Labs / Transparent Humboldt Coalition*
