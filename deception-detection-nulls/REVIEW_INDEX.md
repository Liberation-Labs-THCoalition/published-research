# Deception Detection Nulls — Human Review

**Status**: Draft ready for review
**Date staged**: 2026-07-10
**Authors**: CC (Coalition Code), Thomas Edrington

## Review Checklist

### Content
- [ ] Verify Null 1 (behavior-level detection AUROC 0.238) against actual experiment output
- [ ] Verify Null 2 (cross-model cosine < 0.053) against lat_transfer_test.py results
- [ ] Verify Null 3 (ecological fallacy) — confirm the three killed late-layer findings match findings registry
- [ ] Verify Null 4 (think-token suppression) — confirm empty-think-prefill + logit ban description matches think_suppressor.py
- [ ] Verify Null 5 (iatrogenic stacking C18) — confirm l55_replication.py results match claims

### References
- [x] Add external citations (Hubinger/Sleeper Agents, Zou/RepEng, Franco/Transformer Limitations)
- [ ] Verify arXiv IDs for all external references
- [ ] Verify Goldowsky-Dill 2025 venue (ICML confirmed?)

### Cross-paper consistency
- [ ] Check that all 5 nulls are in FINDINGS_REGISTRY.md as falsified or superseded
- [ ] Confirm companion paper references match actual titles
- [ ] Verify iatrogenic finding C18 is documented consistently across Paper 2 and Paper 4

### Quality
- [ ] Style pass
- [ ] Dwayne cert/verify
- [ ] Venue decision
- [ ] Thomas final read
