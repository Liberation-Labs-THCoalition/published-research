# Adversarial Audit Methodology — Human Review

**Status**: Draft ready for review
**Date staged**: 2026-07-10
**Authors**: CC (Coalition Code), Thomas Edrington

## Review Checklist

### Content
- [ ] Verify all 6 rounds accurately describe what happened (cross-check with experiment scripts and findings registry)
- [ ] Confirm killed-claims registry in Section 3 matches FINDINGS_REGISTRY.md (C1-C21)
- [ ] Check that Round 5 circularity regression description matches subspace_reanalysis.py
- [ ] Verify Round 6 split verdict accurately reflects confirmatory replication results (p=0.34 primary, p=0.019 placebo)

### References
- [x] Add external citations on red-teaming methodology (Perez, Hubinger, Ganguli, Casper, Raji, Shevlane)
- [ ] Verify arXiv IDs for all external references
- [ ] Check Perez et al. 2023 venue (ACL Findings confirmed?)
- [ ] Check Casper et al. 2024 venue (FAccT confirmed?)

### Structure
- [ ] Supplementary material A (6 audit reports) — 3 of 5 present, stages 3 and 5 missing
- [ ] Supplementary material B (killed-claims registry) — does it exist as a standalone file?
- [ ] Supplementary material C (audit protocol code) — does it exist?

### Quality
- [ ] Style pass — consistent tone, no hedging where claims are supported
- [ ] Dwayne cert/verify
- [ ] Venue decision (workshop vs. standalone)
- [ ] Thomas final read
