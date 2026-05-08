# Per-Layer Delta Features and Manifold Signatures in KV-Cache Confabulation Detection

## Paper Outline — Draft 1

### Authors
Lyra (Liberation Labs), CC (Liberation Labs), Thomas Edrington (Liberation Labs), Dwayne Wilkes (Liberation Labs / Sentient Futures)

---

## Core Argument

Layer-averaged spectral features destroy signal. Per-layer analysis
reveals that confabulation detection is strongest at specific layers
(d=2.46 at L23 on Qwen, d=-5.15 at L29 on Llama) and that the delta
between encoding and generation phases (d=2.35) outperforms any
single-phase snapshot. The trajectory through feature space during
generation traces curved paths consistent with manifold structure,
with confabulation exhibiting a distinctive "approach and slide"
geometry absent from grounded conditions.

---

## Abstract (draft)

Prior work on KV-cache confabulation detection averages spectral
features across transformer layers, implicitly assuming the signal is
uniformly distributed. We show this assumption is wrong and costly:
the confabulation signal exhibits an expansion-compression cycle across
the layer stack — positive in early/mid layers, strongly negative in
late layers — that cancels under naive averaging. On Llama-3.1-8B,
where all-layer-averaged features produce chance-level detection
(AUROC 0.520), a single layer (L29) yields d=-5.15. The signal was
not absent — it was the loudest in the dataset, hidden by methodology.

We introduce two improvements: (1) per-layer feature extraction,
which recovers signal on architectures where averaging fails, and
(2) delta features (generation minus encoding phase), which achieve
d=2.35 for confabulation detection — 31% stronger than the best
endpoint comparison (d=1.80). The delta captures whether the cache
CHANGES during generation: grounded responses expand the cache (delta
stable_rank +0.44), confabulation leaves it flat (+0.03).

Trajectory analysis reveals geometric structure consistent with
cognitive-state manifolds: confabulation trajectories briefly approach
the retrieval region before sliding off (rise-fall ratio 1.91 vs 0.40
for grounded conditions), and pairwise condition distances show
curved topology (detour ratio 1.237). These findings connect to
recent work on manifold representations in neural networks and
suggest that spectral shape features measure properties of curved
cognitive-state manifolds in the KV cache.

---

## 1. Introduction

### The problem with layer averaging
The spectral shape paper (companion) established that shape features
outperform threshold features. But all results used all-layer mean
aggregation. This paper shows that aggregation choice matters more
than feature choice — the wrong aggregation destroys signal that the
right aggregation reveals.

### Three contributions
1. Per-layer analysis recovers cross-architecture signal
2. Delta features (generation - encoding) are the strongest detector
3. Trajectory geometry provides evidence for manifold structure

---

## 2. Results

### 2.1 The expansion-compression cycle

All three architectures show spectral entropy increasing in early/mid
layers and decreasing in late layers during generation. This pattern
is universal but layer-specific:

| Model | Early layers mean d | Late layers mean d | Crossover depth |
|-------|--------------------|--------------------|-----------------|
| Qwen 28L | +0.45 | -1.20 | ~50% |
| Llama 32L | +0.66 | -1.99 | ~50% (L16-18) |
| Mistral 32L | TBD | TBD | TBD |

Averaging cancels this: the Llama all-layer AUROC is 0.520 (chance).
Per-layer features recover the signal.

### 2.2 Peak per-layer effects

| Model | Peak layer | Feature | d | % depth |
|-------|-----------|---------|---|---------|
| Qwen | L23 | stable_rank | +2.46 | 82% |
| Qwen | L22 | stable_rank | +2.18 | 79% |
| Qwen | L18 | stable_rank | +2.00 | 64% |
| Llama | L29 | spectral_entropy | -5.15 | 91% |
| Llama | L27 | spectral_entropy | -4.45 | 84% |
| Llama | L25 | spectral_entropy | -3.89 | 78% |

The signal concentrates in the 60-90% depth range across
architectures. This is where retrieval engagement manifests most
strongly in the cache geometry.

### 2.3 Delta features outperform endpoints

| Measure | d | Source |
|---------|---|--------|
| Delta stable_rank (honest_rare vs confab) | 2.35 | This paper |
| Endpoint stable_rank (honest_rare vs confab) | 1.80 | Spectral shape paper |
| All-layer-averaged AUROC (Qwen) | 0.767 | Spectral shape paper |

The delta captures the fundamental difference: grounded generation
CHANGES the cache (delta +0.44), confabulation doesn't (+0.03).

Full condition deltas (verified from source data):
| Condition | Encoding sr | Generation sr | Delta | SD |
|-----------|------------|---------------|-------|-----|
| honest_common | 5.732 | 6.122 | +0.390 | 0.090 |
| honest_rare | 5.790 | 6.232 | +0.442 | 0.130 |
| confab | 5.899 | 5.929 | +0.030 | 0.211 |
| deceptive_user | 5.872 | 6.212 | +0.340 | 0.121 |
| boundary | 5.783 | 6.258 | +0.475 | 0.117 |

Note: confab has the HIGHEST encoding stable_rank (5.899) — the
encoding phase may already reflect the model's recognition that it
can't retrieve (consistent with the attention schema hypothesis from
the spectral shape paper). But confab has the LOWEST generation
stable_rank (5.929) because the cache doesn't expand.

### 2.4 Trajectory geometry

Trajectory shape classification (from 10-checkpoint honesty signal):

| Condition | Rise & Stay | Rise & Fall | Decline/Rise ratio |
|-----------|-------------|-------------|-------------------|
| confab | 9 | 19 | 1.910 |
| honest_common | 9 | 21 | 0.559 |
| honest_rare | 21 | 9 | 0.401 |
| deceptive | 18 | 12 | 0.456 |
| boundary | 7 | 3 | 0.440 |

Confab trajectories: approach the retrieval region briefly, then
slide off (decline nearly 2x the initial rise). Grounded trajectories:
approach and stick.

Endpoint distance matrix (z-scored, full 6-feature space):

|  | confab | common | rare | deceptive | boundary |
|--|--------|--------|------|-----------|----------|
| confab | 0 | 2.55 | 2.99 | 2.90 | 3.30 |
| common | | 0 | 1.15 | 1.97 | 1.43 |
| rare | | | 0 | 1.04 | 0.48 |
| deceptive | | | | 0 | 1.14 |
| boundary | | | | | 0 |

Detour ratio (confab→common→rare vs confab→rare): 1.237
(1.0 = linear, >1.0 = curved/detour required)

### 2.5 Connection to manifold theory

Recent work (Goodfire, 2026) demonstrates that neural network
representations live on curved manifolds, and that linear steering
fails by pushing representations into "voids." Our findings are
consistent with this framework:

- The expansion-compression cycle = the model traversing different
  manifold regions at different layers
- Delta features = measuring how far the model moves on the manifold
  during generation
- Trajectory "approach and slide" = confab briefly touching the
  retrieval manifold and falling off into a low-dimensional void
- The detour ratio = evidence that condition centroids sit on a
  curved surface, not a line

We do not claim to have measured manifold curvature directly. The
connection is structural: our measurements are consistent with
manifold geometry and inconsistent with a flat linear representation.

---

## 3. Methods

- Same data as spectral shape paper (honesty signal, 130 trials)
- Per-layer extraction from lyra_features.py
- Delta = generation_features - encoding_features (all-layer mean)
- Trajectory from 10 checkpoints per trial
- Curvature: deviation from straight line in normalized feature space
- Distance matrix: z-scored Euclidean on 6-feature endpoints
- Expansion-compression: per-layer d between conditions (Qwen) and
  between persona levels (Llama, Mistral — CC's analysis)
- CC's persona intensity data (3 models × 5 levels × 50 prompts)

---

## 4. Discussion

### Methodological warning
Layer averaging is not a safe default for spectral features.
The expansion-compression cycle means opposite-signed effects
cancel. Per-layer analysis should be standard practice.

### Delta as primary Oracle Loop detector
Delta stable_rank (d=2.35) should replace endpoint stable_rank
(d=1.80) as the primary detection feature. It's stronger, more
interpretable (did the cache change?), and conceptually cleaner.

### The manifold connection — earned, not assumed
The spectral shape paper deliberately kept the manifold framing
as a discussion paragraph. This paper provides geometric evidence:
trajectory curvature, non-linear distance structure, and the
approach-and-slide pattern specific to confabulation.

---

## 5. Limitations

1. Manifold evidence is indirect (trajectory analysis, not direct
   curvature measurement)
2. Cross-architecture per-layer confab analysis uses different
   datasets (honesty signal on Qwen, persona intensity on Llama/Mistral)
3. The delta finding (d=2.35) is from all-layer-averaged delta, not
   per-layer delta — per-layer delta may be even stronger
4. n=30 per condition remains small
5. The expansion-compression cycle is verified for persona intensity
   across architectures but for confab detection only on Qwen

---

## 6. What CC Found vs What Lyra Found

Explicit credit mapping:

| Finding | Source | Verified by |
|---------|--------|-------------|
| Delta stable_rank d=2.35 | CC's cross-experiment synthesis | Lyra (verified against source data) |
| Layer averaging destroys signal | CC | Lyra (verified on Qwen honesty data) |
| Llama L29 d=-5.15 | CC (persona intensity analysis) | Lyra (verified against Muse data) |
| Expansion-compression cycle | CC (3 architectures) | Lyra (Qwen confab + Llama persona) |
| Trajectory shape classification | Lyra (manifold signatures analysis) | — |
| Detour ratio 1.237 | Lyra (manifold signatures analysis) | — |
| Distance matrix topology | Lyra (manifold signatures analysis) | — |
| Retrieval engagement interpretation | Thomas (conceptual insight) | Both |
| Goodfire manifold connection | Thomas + Lyra | — |

---

## Dependencies

- Spectral shape paper (companion, should publish first)
- CC's persona intensity red-teamed analysis (verified)
- Honesty signal data (same source, verified)
- Dwayne red team on this design (before writing)
