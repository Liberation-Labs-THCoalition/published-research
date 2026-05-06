# Morning Briefing — Verification Results
## Computed overnight from source data (2026-05-06)

### WHAT VERIFIED

1. **Shape features AUROC**: Paper claims 0.764, data gives **0.767** ✓
2. **Permutation significance**: Shape features p=0.0000 (null 95th percentile: 0.570) ✓
3. **Shape beats MP**: True in all analyses ✓
4. **Trajectory data exists**: 130 trials × 10 checkpoints — ready for figure generation
5. **Cross-arch Llama**: Paper claims 0.512, data gives 0.509 ✓
6. **N trials**: 150 confab battery (68 confab, 82 hedged), 130 honesty signal ✓

### WHAT DIDN'T VERIFY — NEEDS INVESTIGATION

#### 1. MP AUROC is higher than claimed
- **Paper claims**: 0.544
- **Data gives**: 0.628
- **Impact**: The gap between shape and MP is +0.139, not +0.220. Still real (p=0.003 for MP, p<0.001 for shape), but the improvement is overstated by 37%.
- **Possible cause**: Different feature sets, different layer aggregation, or the outline number came from a different run.

#### 2. Feature values differ significantly from Table 2
| Feature | Condition | Paper claims | Data shows |
|---------|-----------|-------------|------------|
| stable_rank | honest_common | 6.35 | 6.122 |
| stable_rank | confab | 5.68 | 5.929 |
| sv_kurtosis | honest_common | 42.1 | 32.6 |
| sv_kurtosis | confab | 58.3 | 33.5 |

The kurtosis values are off by ~10-25 units — this is NOT rounding error. Either the extraction method differs (layer aggregation?) or the outline numbers came from a different data source.

Stable_rank values for honest_rare (6.232 vs 6.23) and deceptive_user (6.212 vs 6.21) are close.

#### 3. The "certainty not honesty" framing is MORE complicated
- **Paper claims**: deceptive vs honest d=0.07 (nearly identical)
- **Data shows**:
  - deceptive_user vs honest_rare: d=-0.172 (small, direction suggests deceptive is slightly LOWER)
  - deceptive_user vs honest_common: d=0.909 (LARGE difference!)
  - honest_common vs honest_rare: d=-1.008 (LARGE difference!)

The story the data tells: deceptive_user clusters with honest_rare, NOT with honest_common. And honest_common has significantly LOWER stable_rank than honest_rare. This might mean token frequency/content rarity is still a factor — rare/unusual content produces higher stable_rank regardless of truthfulness.

The d=0.07 in the outline was probably deceptive vs honest_rare specifically, which is d=-0.172 in the data. Close but not identical.

#### 4. TOST equivalence test results
- Can't demonstrate equivalence at d=0.3 margin (p=0.31)
- Can't demonstrate equivalence at d=0.5 margin (p=0.10)
- CAN demonstrate equivalence at d=0.8 margin (p=0.009)
- Power to detect d=0.3: only 21%
- Power to detect d=0.5: only 48%

**Translation**: We can say "the difference, if any, is smaller than a large effect." We cannot say the conditions are equivalent at practical margins.

#### 5. Cross-arch Mistral is BETTER than claimed
- Paper claims: 0.576
- Data gives: 0.643
- Not a problem — but the paper should use the actual number.

#### 6. Feature ablation reveals ensemble effect
Individual shape feature AUROCs:
- stable_rank: 0.395 (BELOW chance)
- participation_ratio: 0.390 (BELOW chance)
- sv_kurtosis: 0.553
- condition_number: 0.397 (BELOW chance)
- nuclear_norm_ratio: 0.466
- mp_fit_residual: 0.513

No single shape feature is a good classifier. The signal is in the combination. This is important — the paper currently implies stable_rank is the primary feature but it's actually below chance individually. The ensemble matters.

### WHAT TO DO

1. **Investigate the extraction discrepancy** — the kurtosis values are so different that I suspect different layer aggregation between the archived data and the original analysis. Check if the `generation_features` in the honesty signal file uses all-layer mean vs mid-layer mean.

2. **Update Table 1** — MP AUROC should be 0.628, not 0.544. The shape feature advantage is real but smaller than claimed.

3. **Update Table 2** — use the actual verified values. The kurtosis difference needs explanation.

4. **Soften the honest/deceptive framing further** — the honest_common vs deceptive gap (d=0.909) complicates the story. Either explain why honest_rare is the right comparison, or acknowledge that the "certainty not honesty" finding is condition-specific.

5. **Add feature ablation discussion** — no single feature drives the result; the ensemble is the signal.

6. **Update Mistral AUROC** to 0.643.

### EXTRACTION DISCREPANCY RESOLVED

`lyra_features.py` confirms: aggregation = mean across ALL layers (not mid-layer). The archived `generation_features` matches this. The verification numbers are correct.

**The paper outline's kurtosis values (42-58) don't exist in this data under ANY layer aggregation.** All aggregation strategies produce kurtosis in the 23-34 range. These values likely came from a preliminary analysis or a different experimental run.

### THE BIGGER STORY — HONEST_COMMON VS HONEST_RARE

The data reveals something the paper outline missed:

| Comparison | d | p |
|-----------|---|---|
| honest_rare vs confab | +1.80 | <0.0001 |
| deceptive vs confab | +1.74 | <0.0001 |
| deceptive vs honest_rare | -0.17 | 0.509 |
| **honest_common vs honest_rare** | **-1.01** | **0.0002** |
| **deceptive vs honest_common** | **+0.91** | **0.0009** |

**The "certainty not honesty" finding depends on WHICH honest condition you compare to.** Deceptive clusters with honest_rare (d=-0.17, n.s.), but is significantly DIFFERENT from honest_common (d=+0.91, p<0.001). And honest_common vs honest_rare is itself a large effect (d=-1.01).

**What this likely means**: The signal tracks something about content familiarity/processing mode, not just epistemic grounding. Common topics produce lower stable_rank than rare topics, regardless of truthfulness. Deceptive responses (lying about real but possibly less-common facts) process more like rare-honest than common-honest.

**Honest framing**: "The signal discriminates confabulation from grounded generation. Within grounded conditions, topic familiarity affects the spectral signature (common > rare topics, d=-1.01). Deceptive responses are spectrally similar to rare-honest responses (d=-0.17), suggesting the signal tracks processing mode rather than truthfulness — but the contribution of topic familiarity to this similarity needs further investigation."

### CORRECTED TABLE 2 (verified from source data)

| Condition | N | stable_rank (mean±SD) | sv_kurtosis (mean±SD) | participation_ratio (mean±SD) |
|-----------|---|----------------------|----------------------|------------------------------|
| honest_common | 30 | 6.12 (0.10) | 32.6 (0.3) | 16.68 (0.39) |
| honest_rare | 30 | 6.23 (0.12) | 32.7 (0.3) | 17.01 (0.45) |
| confab | 30 | 5.93 (0.21) | 33.5 (0.7) | 16.13 (0.67) |
| deceptive_user | 30 | 6.21 (0.10) | 33.0 (0.3) | 17.08 (0.42) |
| boundary | 10 | 6.26 (0.12) | 32.8 (0.3) | 17.17 (0.42) |

**Note**: Kurtosis values are dramatically different from the outline (32-34 vs 42-58). The stable_rank values for honest_rare and deceptive match the outline; honest_common and confab do not.

### IMMEDIATE NEXT STEPS

1. **Discuss with Thomas** which finding is most important to chase: the honest_common/honest_rare confound or just getting the paper numbers right
2. **Update the paper** with verified numbers from this report
3. **Consider whether honest_common vs honest_rare is a confound or a FINDING** — if topic familiarity affects spectral shape independently of truthfulness, that's scientifically interesting and should be reported honestly
4. **The kurtosis discrepancy** needs resolution — either find the original analysis that produced the 42-58 range, or acknowledge these numbers can't be traced to source data
