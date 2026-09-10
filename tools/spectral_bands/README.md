# spectral_bands

The middle-band estimator, extracted from `51b_moe_advanced_analysis.py` as a
tested library. For Lyra, 2026-09-07.

    upper edge = Marchenko-Pastur outlier cut     (representational strain above)
    lower edge = Gavish-Donoho optimal threshold  (noise below)
    middle     = real but not dominant

Answers: **does the early-token commitment effect survive in the middle band?**
Top band only -> representational strain, the confound. Survives -> schema
commitment, and this count is the instrument.

## Read this before you run it: my August advice was wrong

I told you to stack ~10 prompts into a taller matrix to fix your aspect ratio.
**Do not.** Measured while building this:

| tokens (512 dims) | gamma | band counts under planted signal | verdict |
|---|---|---|---|
| 31 (yours)        | 16.52 | [0, 1, 2, 3, 6] | usable |
| 256               |  2.00 | [0, 1]          | usable |
| 341               |  1.50 | [0]             | **CLOSED** |
| 512               |  1.00 | [0]             | **CLOSED** |

At low gamma the MP edge collapses toward the bulk while GD's `omega*median(sv)`
holds, the edges **cross**, and the band closes — returning 0 forever, which is
indistinguishable from "no signal found." Ten stacked prompts (310x512, gamma
1.65) lands inside the closed region.

Stacking is the right remedy for MP's bulk estimate and the wrong one for this
band. The two goals conflict. Your "brutal" 31x512 is inside the working window;
my fix would have silently zeroed your measurement. `aperture_ok()` now refuses.

## The null is exactly zero

120 trials at every shape from gamma 2.0 to 25.6: the middle band is **empty
under pure noise, always**. So any positive count is signal above noise by
construction, and the null is a point rather than a spread. Run
`null_band_count()` at your own n anyway before trusting a number — your method
from the d=1.36 kill, wired in as a permanent gate.

## Use

```python
from spectral_bands import band_report, band_timecourse, exceeds_null, aperture_ok

aperture_ok(31, 512)                      # check the band is open FIRST
r = band_report(svs, m=31, n=512)         # r["usable"] is False if it is not
r["middle_band_count"]

# your temporal question, answered temporally this time
for w in band_timecourse(cache, window=8, stride=4):
    print(w["token_start"], w["token_end"], w["middle_band_count"])

exceeds_null(observed=12, m=31, n=512)    # verdict travels WITH its null
```

`sigma_from="lower"` fits the bulk on the lower spectrum only — use it above
gamma ~5, where real structure contaminates the median upward.

numpy only. No GPU, no torch — Phase 2 compute is still withdrawn and this
should not wait on it.

## Provenance

`gavish_donoho_threshold()` and the MP path are yours and Exp 51's, moved
verbatim. What is new: the two aspect-ratio conventions named and asserted
(`beta = min/max` for GD, `gamma = max/min` for MP — reciprocals, and passing one
for the other returns a plausible number rather than raising), the aperture
guard, the windowed mode, the null gate.

17 tests, 12 mutants, all killed — including the three that survived the first
pass because their assertions were satisfied by a constant. Your rule, applied to
your gift.
