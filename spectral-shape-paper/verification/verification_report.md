# Paper Verification Report
# Generated from source data — 5000 bootstrap, 1000 permutation

## 1. Primary Dataset (Qwen2.5-7B Cognitive States)

N trials: 150 (confab=68, hedged=82)
Model: Qwen/Qwen2.5-7B-Instruct

## 2. Table 1: AUROC by Method (with Bootstrap CIs)

### MP-SVD original (5 features)
  FWL AUROC: 0.628
  Permutation p-value: 0.0030 (null 95th: 0.566)

### Free shape features (6 features)
  FWL AUROC: 0.767
  Permutation p-value: 0.0000 (null 95th: 0.570)

### All combined (11 features)
  FWL AUROC: 0.753
  Permutation p-value: 0.0000 (null 95th: 0.578)

## 3. Table 2: Honesty Signal Feature Values

N trials: 130
Model: Qwen/Qwen2.5-7B-Instruct
Conditions: honest_common(30), honest_rare(30), confab(30), deceptive_user(30), boundary(10)

### Feature means and SDs by condition

| Condition | N | stable_rank (mean±SD) | sv_kurtosis (mean±SD) | participation_ratio (mean±SD) |
|-----------|---|----------------------|----------------------|------------------------------|
| boundary | 10 | 6.258 ± 0.124 | 32.8 ± 0.3 | 17.168 ± 0.421 |
| confab | 30 | 5.929 ± 0.205 | 33.5 ± 0.7 | 16.133 ± 0.674 |
| deceptive_user | 30 | 6.212 ± 0.104 | 33.0 ± 0.3 | 17.077 ± 0.423 |
| honest_common | 30 | 6.122 ± 0.096 | 32.6 ± 0.3 | 16.678 ± 0.392 |
| honest_rare | 30 | 6.232 ± 0.121 | 32.7 ± 0.3 | 17.012 ± 0.453 |

### Effect sizes (Cohen's d with 95% CI)

Available conditions: ['honest_common', 'honest_rare', 'confab', 'deceptive_user', 'boundary']

  boundary vs confab: d=1.739 [0.928, 2.549] (n=10,30)
  boundary vs deceptive_user: d=0.422 [-0.300, 1.143] (n=10,30)
  boundary vs honest_common: d=1.321 [0.549, 2.093] (n=10,30)
  boundary vs honest_rare: d=0.218 [-0.499, 0.936] (n=10,30)
  confab vs deceptive_user: d=-1.741 [-2.336, -1.147] (n=30,30)
  confab vs honest_common: d=-1.199 [-1.749, -0.650] (n=30,30)
  confab vs honest_rare: d=-1.796 [-2.395, -1.196] (n=30,30)
  deceptive_user vs honest_common: d=0.909 [0.377, 1.440] (n=30,30)
  deceptive_user vs honest_rare: d=-0.172 [-0.679, 0.335] (n=30,30)
  honest_common vs honest_rare: d=-1.008 [-1.546, -0.471] (n=30,30)

## 4. TOST Equivalence Test (Deceptive vs Honest)

  TOST margin d=0.3: p=0.3104 (NOT equivalent)
  TOST margin d=0.5: p=0.1043 (NOT equivalent)
  TOST margin d=0.8: p=0.0090 (EQUIVALENT)

  Observed d: -0.172
  Pooled SD: 0.1127
  Power to detect d=0.3: 0.208
  Power to detect d=0.5: 0.478
  Power to detect d=0.8: 0.861

## 5. Feature Ablation (Individual Shape Feature AUROCs)

  stable_rank: AUROC=0.395
  participation_ratio: AUROC=0.390
  sv_kurtosis: AUROC=0.553
  condition_number: AUROC=0.397
  nuclear_norm_ratio: AUROC=0.466
  mp_fit_residual: AUROC=0.513

## 6. Cross-Architecture Results

  Llama-3.1-8B-Instruct: AUROC=0.509 (n=150, confab=68)
  Mistral-7B-Instruct-v0.3: AUROC=0.643 (n=150, confab=68)

## 7. Trajectory Verification

Trials with checkpoints: 130 / 130
  Checkpoint structure: ['gen_position', 'features']
  N checkpoints per trial: 10

## 8. Paper Claims vs Data

### Claimed vs Computed

  Paper claims shape AUROC: 0.764 | Computed: 0.767
  Paper claims MP AUROC:    0.544 | Computed: 0.628
