# R1 / R2 / R7 predictions, sealed before the encode

**Written:** 2026-10-18 18:02 PDT by Lyra. No R1, R2 or R7 number exists. R0 is known (text: 37.0% at 5 folds,
38.2% at 10; valence 0.963). The power table is known (+5 points: 0.58–0.94 depending on disagreement; +3: 0.16–0.49).

**R1 (reproduce v2, 5 folds):** reproduced within the measured band, **70%**. If not, I expect the weights to be the
cause, not the code (the HF history was squashed in July).

**R2 primary (W_K-projected, training-fold directions, per-layer cosine, L3–L15, skip 50, 10 folds) against R0 at 10
folds:**
- INCONCLUSIVE_UNDERPOWERED: **50%**
- FAIL_BELOW_TEXT: **25%**
- EXCEEDS_TEXT (robust to the cell bootstrap): **20%**
- FAIL_WITHIN_2X_CHANCE: **5%**
- Point guess for R2's accuracy: **34%**, with a 90% interval of 22% to 46%. The W_K projection loses information
  that v2's raw class-mean keys kept, so I expect it to land a little *below* R1.

**The all-16-layer disclosure cell:** within 5 points of the primary, same direction against text: **70%**.

**Valence (S3, R2 against R0 at 10 folds):** not distinguished from text's 0.963: **65%**.

**R7 (spectral, n-matched, on this encode):** NULL_HOLDS **60%**; SIGNAL_FOUND **25%**; UNINFORMATIVE_INSTRUMENT
**15%**. The April null was a real test, and I expect it to replicate.

**What would surprise me most:** R2 beating text by a margin the cell bootstrap survives. That would mean the
paper's described method carries something the words don't, which is more than I now expect of it.
