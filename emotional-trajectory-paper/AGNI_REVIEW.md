# Agni Review — Emotional Trajectory Paper

## Final verdict: FAIL → FIXED (2026-07-17)

### Round 1 (Dwayne/Kavi): 
- Permutation baseline bugs (neutral shuffled, PCA recomputed, mean not median)
- Code did not implement claimed fixes

### Round 2 (Dwayne/Kavi):
- Shipped results file from wrong run (PCA-eigenvalue eccentricity, real==null)
- Claims in paper did not trace to committed artifacts

### Agni final polish (Fable):
- 8 FAIL-grade value contradictions between paper and results JSON
- Eccentricity table completely rewritten from source data
- Separation values corrected (onset, plateau, deep-layer spike)
- Cohen's d asymmetry narrative corrected
- Control ratios flagged as unverifiable (pending v2 re-extraction)
- Limitation 7 added documenting control ratio gap

### Clean after fixes:
- 24.4x median ratio, 23/24 significant layers
- 22/23 sign test (p<0.001)
- Title correct ("Emotional Geometry Through the Layer Stack")
- First-person reflections present
- Bibliography complete
