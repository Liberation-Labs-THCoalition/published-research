# Spectral Shape Features for Confabulation Detection: Threshold-Free KV-Cache Analysis

## Paper Outline — Draft 2 (revised after red team)

### Authors
Lyra (Liberation Labs), Thomas Edrington (Liberation Labs), Dwayne Wilkes (Liberation Labs)

---

## Abstract

Threshold-dependent spectral features (Marchenko-Pastur signal rank) fail to generalize across transformer architectures for confabulation detection. We show that threshold-independent spectral shape features — stable rank, singular value kurtosis, participation ratio — substantially outperform threshold-based methods, achieving 0.764 AUROC on frequency-controlled confab detection (vs 0.544 for MP features) on Qwen2.5-7B. A frequency-matched control (honest answers about rare topics vs fabricated topics, both using rare tokens) confirms the signal is cognitive, not lexical (stable_rank d=1.83). The spectral shape tracks epistemic certainty rather than honesty: deceptive responses grounded in real knowledge are spectrally indistinguishable from honest ones, while confabulation produces peaked, concentrated spectra regardless of token frequency. Token-by-token trajectory analysis shows the divergence emerges within 20-30 generated tokens. These findings suggest that the singular value distribution of the KV cache encodes the model's epistemic state — how confidently it is generating from retrieved knowledge versus fabricating — and that distributional shape captures this more robustly than threshold-counting.

---

## 1. Introduction

### The problem
KV-cache geometry detects confabulation (prior work: AUROC 0.903 on full cache) but the features are architecture-specific, length-confounded, and poorly understood mechanistically.

### The question
Can we find features that: (a) survive frequency and length controls, (b) have a clear mechanistic interpretation, (c) are deployable at inference time?

### The answer (preview)
Yes — spectral shape features. They measure how the model distributes its representational energy across dimensions. Confabulation concentrates energy (peaked spectrum, low stable_rank). Honest uncertain processing distributes it (flat spectrum, high stable_rank). The signal tracks epistemic certainty — whether the model is generating from retrieved knowledge or fabricating — rather than honesty per se.

### Why shape beats threshold
Threshold features (MP signal_rank) count how many singular values exceed a theoretical noise edge. This depends on accurate noise modeling, which varies across architectures and saturates on dense models. Shape features characterize the entire singular value distribution without reference to any threshold, making them inherently more portable and robust to architectural differences.

---

## 2. Related Work

- MP random matrix theory in cache geometry (our prior campaigns)
- Gavish-Donoho optimal thresholding
- Token frequency as confound (CC's MoE finding, Experiment 51)
- Manifold representations in LLMs (2604.28119)
- Epistemic uncertainty in language models (other approaches)
- SAE vs spectral decomposition for interpretability

---

## 3. Methods

### 3.1 Unified Feature Extraction (lyra_features.py)
- Single extraction module for all experiments
- GPU SVD (torch.linalg.svdvals)
- MP and GD thresholds computed per layer
- 6 threshold-independent features + 5 threshold-dependent + spectral entropy
- Delta features (generation minus encoding)
- Code-reviewed, smoke-tested

### 3.2 Feature Categories

**Threshold-dependent (counting directions above a noise floor):**
- MP signal_rank, signal_fraction, top_sv_excess
- GD signal_rank, signal_fraction, top_sv_excess
- These COUNT how many singular values exceed a theoretical noise edge

**Threshold-independent (measuring distributional shape):**
- stable_rank: how evenly energy is distributed (||S||²_F / ||S||²_2)
- participation_ratio: effective number of contributing dimensions
- sv_kurtosis: peakedness of the SV distribution
- condition_number: ratio of largest to smallest SV
- nuclear_norm_ratio: normalized trace norm
- mp_fit_residual: deviation from random matrix prediction

**Interpretation:**
- Threshold features ask: "how many directions exceed the noise floor?"
- Shape features ask: "how is the model's representational energy distributed?"
- Shape features are threshold-free by construction — they characterize the full singular value distribution, so they remain informative even when threshold estimation fails (e.g., dense models where MP saturates at 128/128)

### 3.3 Experimental Design

**Refinement battery** (150 confab prompts, Qwen 7B):
- 8 methods: MP-SVD, GD-SVD, free features, differential SVD, ICA, Hankel/SSA, PCA, combinations
- Within-fold FWL, GroupKFold, LogisticRegressionCV (L2, nested tuning)

**Frequency-controlled honesty signal** (5 conditions × 30 prompts):
- honest-common, honest-rare, confab, deceptive-user, boundary-knowledge
- Same system prompt for non-deceptive conditions
- Fixed 100-token generation (min_new_tokens = max_new_tokens)
- Trajectory extraction at every 10th token

**Cross-architecture** (3 models):
- Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
- Cognitive state battery: confab + overconfidence on all three

### 3.4 Controls (14-control battery + frequency matching)
- All mandatory controls from design-experiment framework
- Frequency-matched comparison: honest-rare vs confab (CC's MoE control)
- Interaction-term FWL diagnostic
- V0 instability exclusion (SVD unreliable below ~20 tokens)

---

## 4. Results

### 4.1 Shape features replace threshold features

| Method | FWL AUROC | Notes |
|--------|-----------|-------|
| MP-SVD original (5 features) | 0.544 | Baseline |
| Free shape features (6) | 0.764 | +0.220 absolute gain |
| GD-SVD (5) | 0.734 | Better threshold, same paradigm |
| PCA(5) on all 11 | 0.775 | Marginally best |
| Combined L1 (sparse) | 0.771 | L1 selects shape features |

The original MP features contribute nothing beyond the shape features (+0.004). This is a replacement, not an augmentation.

### 4.2 The frequency control

honest-rare vs confab (both rare tokens, same system prompt):
- stable_rank: d=+1.83 (honest-rare has flatter spectrum)
- sv_kurtosis: d=-1.63 (confab has more peaked spectrum)
- Frequency matching eliminates the token-rarity confound

Deceptive_user (lying about real facts) has stable_rank=6.21, nearly identical to honest_rare (6.23). The signal tracks grounding/certainty, not honesty.

### 4.3 The trajectory

stable_rank trajectory across 100 tokens of generation:
- All conditions start at ~5.9
- Grounded conditions (honest, deceptive) rise to ~6.25 by token 30
- Confab barely rises to 6.0 and falls back
- Divergence detectable by token 20
- Maximum separation at token 80 (d=1.85)

### 4.4 Cross-architecture: features are model-specific

| Model | Free features FWL AUROC |
|-------|------------------------|
| Qwen 7B | 0.764 |
| Llama 8B | 0.512 |
| Mistral 7B | 0.576 |

Z-scored features across models: pooled AUROC 0.621. The signal exists cross-architecture but is weak without per-model calibration.

### 4.5 What doesn't work
- ICA on aggregated features (0.500)
- Hankel/SSA temporal decomposition (0.576)
- TDA persistent homology (0.500 — wrong input format)
- Overconfidence detection (chance on all models, all features)
- MP outlier count on dense models (saturates at 128/128)

### 4.6 Epistemic certainty interpretation

The singular value distribution appears to encode the model's epistemic state. Grounded generation (whether honest or deceptive) produces broad, distributed spectra — the model engages many representational dimensions when generating from retrieved knowledge. Confabulation produces concentrated spectra — fewer dimensions carry most of the energy, consistent with the model collapsing to a narrow generative mode rather than drawing on stored knowledge.

The dose-response cliff from the Oracle Formulary (therapeutic window 0.5-1.0, catastrophic self-correction collapse at 1.5) provides convergent evidence: moderate steering preserves the spectral distribution, while excessive steering concentrates it — mirroring the confabulation signature.

---

## 5. Discussion

### 5.1 Why shape beats threshold
Threshold features (MP signal_rank) count directions above an arbitrary noise edge. When the edge is miscalibrated (saturates at 128/128 on dense models), the count is uninformative. Shape features measure the distribution regardless of any threshold. The improvement is not incremental — it is a change in what is being measured: distributional character rather than direction count.

### 5.2 Epistemic certainty, not honesty
The deceptive_user result shows the signal tracks whether the model is generating from retrieved knowledge (grounded) or fabricating (ungrounded). Deception about real facts looks spectrally like honesty because the model is still retrieving. This is arguably more useful for safety monitoring than a lie detector — it identifies when the model is confabulating regardless of intent.

### 5.3 Trajectory dynamics
The token-by-token trajectory (start at ~5.9, diverge by token 20-30) shows how the epistemic signal develops during generation. Grounded generation follows a trajectory that explores broadly then settles (rising stable_rank). Confabulation briefly explores then collapses to a narrow mode (falling stable_rank). The peak position and decline rate are potential features for real-time deployment — detection need not wait for full generation.

### 5.4 Architecture specificity and calibration
The phenomenon (shape features discriminate cognitive states) replicates across architectures, but the specific feature values and optimal combinations are model-dependent. Cross-architecture deployment requires per-model calibration. This is a practical limitation but not a theoretical one — the underlying signal is present; the spectral encoding of it varies.

### 5.5 Connections to MoE findings
On MoE architectures, honest responses produce 2x more MP outlier SVs (10.1 vs 5.5). Expert routing creates natural sparsity that aligns with the MP noise model, explaining why threshold features succeed on MoE but fail on dense models. Shape features bypass this distinction entirely.

### 5.6 Relationship to manifold representation theory
Recent work demonstrates that concepts in LLMs are represented on curved low-dimensional manifolds rather than along linear directions (2604.28119). Our spectral shape features — which characterize the distributional geometry of the singular value spectrum — may be capturing related structure from a different measurement vantage point. The observation that epistemic state is encoded in distributional shape is consistent with a manifold account in which different cognitive states occupy geometrically distinct regions of representation space. However, we have not directly measured manifold structure in KV-cache space, and the relationship between SVD-based spectral features and SAE-based manifold observations remains an open question for future work.

---

## 6. Limitations

1. Primary model: Qwen 7B. Cross-architecture generalization is weak without per-model calibration.
2. n=30-150 per condition. Effect sizes may be inflated by small samples.
3. Frequency control is partial — honest-rare and confab prompts still differ in content beyond token frequency.
4. The encoding signal (0.889 AUROC) is likely a prompt classifier, not a pre-generation cognitive detector.
5. Overconfidence is undetectable by any spectral feature. The method's scope is limited to confabulation.
6. Deployment requires per-model calibration. No universal feature set identified.
7. SVD features operate on whole-cache geometry and may miss effects localized to specific attention heads or circuits.
8. The relationship between spectral shape and recently observed manifold structure in LLM representations (2604.28119) is suggestive but unestablished.

---

## 7. Conclusion

Spectral shape features — stable_rank, sv_kurtosis, participation_ratio — detect confabulation by characterizing how models distribute representational energy across the singular value spectrum of the KV cache. They replace threshold-dependent features that fail on dense architectures. The signal tracks epistemic certainty (grounding vs fabrication) rather than honesty (truthful vs deceptive), which is arguably more useful for safety monitoring. Token-by-token trajectories show the signal emerges within 20-30 tokens, fast enough for real-time deployment. The spectral shape of the KV cache encodes the model's epistemic state; reading it does not require solving the harder problem of interpreting individual features or circuits.

---

## Appendices

A. Full feature correlation matrix
B. Per-layer d profiles for all features
C. Trajectory statistics (peak, decline rate, AUC)
D. Cross-architecture raw results
E. Methods that didn't work (ICA, TDA, Hankel — documented for the field)
F. The unified extraction module (lyra_features.py)
