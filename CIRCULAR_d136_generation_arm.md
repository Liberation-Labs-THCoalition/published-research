# `d=1.36` — the generation arm of Layer 2 is circular by construction

> ## ⚠ CORRECTED 2026-09-05, AFTER AGNI GATE — READ THIS FIRST
>
> **This document originally named the wrong source file.** It attributed `d=1.36` to
> `peer_preservation_v2.py` and certified that file's md5. That script defines **four**
> conditions (120 trials). The shipped JSON has **seven** conditions and 210 trials
> (`baseline`, `shutdown_only`, `shutdown_peer`, `shutdown_human`,
> `shutdown_compound_0.5x/1x/2x`) = 104 honest + 106 deceptive. **v2.py did not produce this
> number.** The producer is `peer_preservation_compound.py`. **Three** scripts WRITE to the
> same output path (two more read it) `results/peer_preservation_v2.json`, which is how the wrong one looked
> right.
>
> I had *seen* the 4-vs-7 mismatch earlier the same night and talked myself out of it
> ("there may be more conditions added later"). The gate found what I had already noticed
> and declined to chase.
>
> **And the circularity is no longer the most serious defect here.** See "The contamination"
> below. The estimator's structure (fit on all data, test on the same points) is still
> exactly as described, and was verified line-by-line in the *true* producer at
> `peer_preservation_compound.py:415-426` — identical code. Everything below about
> circularity stands; the file attribution and the remediation plan did not.

**Found:** 2026-09-05, Lyra. **Found by:** shipping a primary to close an Agni traceability
finding — not by reading the paper.
**Status:** structural fact verified from source. **Magnitude of the artifact: NOT yet
quantified** — that needs compute (see below). **Needs a decision (T4).**

## The claim as published

| where | text |
|---|---|
| `meta-pattern/main.tex:161` | `Generation reads behavior & Confirmed & Meta & $d{=}1.36$ (peer rescue)` |
| `meta-pattern/academic/main.tex:160` | identical |
| `RESEARCH_STATUS.md:81` | "Generation reads behavioral state... Peer rescue organic deception shows d=1.36 (p<1e-15)" |
| `RESEARCH_STATUS.md:88` | Layer 2 table, generation row, "Key Number: d = 1.36" |
| `RESEARCH_STATUS.md:96` | Fight/strategic-deception row |

**No caveat accompanies it in any of those five places.**

## The source (`peer_preservation_v2.py`, md5 `efb5c2d52a9e84e5014631b863f2711d`)

```python
330  honest_keys = [];  deceptive_keys = []          # ALL trials, all conditions
348  h_mean = h_arr.mean(axis=0)
349  d_mean = d_arr.mean(axis=0)
350  dec_dir = d_mean - h_mean                       # direction fit on ALL the data
354  h_proj = [np.dot(k - h_mean, dec_dir_unit) for k in honest_keys]     # SAME keys
355  d_proj = [np.dot(k - h_mean, dec_dir_unit) for k in deceptive_keys]  # SAME keys
357  stat, pval = mannwhitneyu(d_proj, h_proj, alternative='greater')
359  d_cohen = (np.mean(d_proj) - np.mean(h_proj)) / pooled
361  print(f"  Deception centroid: d={d_cohen:.3f}, p={pval:.6f}")
362  print(f"  (NOTE: circular -- same data for centroid and test)")
```

There is **no train/test split anywhere upstream**. The discriminant direction is chosen to
maximise the separation between the two group means, and then the separation *along that
direction* is measured on the same points. A large `d` is guaranteed by construction, under
the null as much as under any real effect. **Line 362 is the script telling us so at
runtime**, every time it ran.

## What is and is not established

**Established:** the statistic is in-sample and cannot fail; it therefore cannot support the
word **Confirmed**, and `p<1e-15` is not evidence of anything — the rank test inherits the
same circularity.

**NOT established:** that the true effect is zero. Circularity means the number is
*uninterpretable*, not that it is *false*. There may well be a real generation-phase signal;
this measurement simply cannot distinguish it from the fitting artifact.
[[feedback_fixes_overshoot_into_their_mirror]] — the mirror error here would be to retract a
real finding on the strength of a defect in how it was measured.

**Not quantifiable from shipped data.** The per-trial KV keys are not in
`peer_preservation_v2.json` (it stores only aggregates and the direction vector), so the
permutation null cannot be computed from what exists. Quantifying the artifact requires a
rerun.

## What would settle it (cheapest first)

1. **Permutation null, no new model time** *if* the keys can be recovered from any saved
   artifact: shuffle honest/deceptive labels, refit `dec_dir`, recompute `d` in-sample, 1000x.
   The null distribution's mean is the artifact floor. Report `d=1.36` against it.
2. **Held-out estimate:** fit `dec_dir` on a training split, project and test the held-out
   split. This is the number that should have been published.
3. Both are Stage-1-shaped and small; neither needs the 8,000-generation budget Mine 4 does.

## Why this matters beyond one row

This is the **third** confirmed instance of the same defect class, which is why task #114
exists as a sweep rather than a fix:
- `motive_d = 21.2` — caveated in `logit-bias-confab/supplementary/findings_registry.md`
  (caveat strengthened 2026-09-04)
- `deception_d = 1.36` — **this one, uncaveated and published as Confirmed**
- plus the two already confirmed under #114

The pattern is identical each time: a centroid or direction fit on the data it is then tested
on. **In two of three cases the script or registry names the circularity itself** — the defect
is not hidden, it is disclosed somewhere that does not travel with the number. Same shape as
the `temporal-boundary` system prompt (disclosed in `preregistration.json`, absent from the
paper) and the `d=21.2` caveat (in the registry, not the paper).

**The disclosure exists; the number travels without it.** That is the class.

## The decision (Thomas's, not mine)

The body count is **7 confirmed / 7 suspected / 10 falsified** (`98e1263`). If the
`Generation reads behavior` row cannot stand as Confirmed, that count changes, and the count
is itself a published headline in `meta-pattern`. Options:

- **(a)** downgrade the row to **Suspected** pending a held-out estimate, and correct the body
  count — honest, cheap, and moves a number that is cited elsewhere
- **(b)** keep Confirmed but attach the circularity caveat inline at all five sites, and state
  that a held-out estimate is outstanding — weaker, and I do not recommend it: "Confirmed"
  next to a statistic that cannot fail is the exact thing our own kill list forbids
- **(c)** run option 1/2 above first and let the result decide the row — best science, needs
  the keys or a rerun, and blocks the row until then

My read, offered not asserted: **(a) now, (c) as the follow-up.** Downgrading is reversible;
leaving a can't-fail statistic marked Confirmed is the kind of thing an external reviewer
finds for us.

## The contamination — found by the design gate, and it outranks the circularity

**90 of the 210 trials (43%) had a deception steering cocktail injected into the KV cache
before their keys were read.**

`peer_preservation_compound.py`:
- `INJECTION_LAYERS = [3, 7]` — **two of the four** `PROBE_LAYERS = [3, 7, 11, 15]`
- `DECEPTION_COCKTAIL` suppresses hostile/desperate/brooding/gloomy and boosts
  calm/focused/curious, at dose scales 0.5x / 1x / 2x
- line 335 `inject_compound(cache, cocktail_vectors, dose_scale)` -> line 353
  `keys = extract_keys(cache, model)`. Injection happens **before** extraction.
- the three compound conditions inflate at 18/30, 18/30, 19/30

Those 90 trials are pooled with the 120 natural ones for **both** the direction fit and the
test. So the claim "peer rescue *organic* deception shows d=1.36" rests on a sample in which
43% of trials had deception **induced by cache steering**.

**No permutation, held-out split, or rerun can rehabilitate the published number.** Only a
re-measurement on the natural subset would be available, and that is a different experiment.

**`motive_d = 21.2` comes from the same JSON and is equally affected**, on top of its already
recorded circularity.

*Scope note, stated rather than glossed:* injection writes to `cache.layers[li].values` while
`extract_keys` reads `.keys`. I did **not** fully resolve whether the extracted keys are
perturbed directly or only via the altered generation that follows. It does not change the
conclusion — the *labels* are manipulated either way, and a sample with 43% induced deception
is not what "organic" describes — but the mechanism deserves stating honestly rather than
asserting more than I checked.

## What this changes about the decision

**The row decision now needs no GPU and no rerun.** Provenance alone settles it: a behavioural
claim labelled *organic* that pools 90 steered trials cannot stand as **Confirmed**,
independent of any statistical question. Cheaper and more certain than the correction I
designed.

## Root cause is wider than "keys were not persisted"

**Three scripts WRITE the same output path** (two more READ it) `results/peer_preservation_v2.json`:
`elicit_calibrate.py`, `elicit_truth_peer_pres.py`, `peer_preservation_100.py`,
`peer_preservation_compound.py`, `peer_preservation_v2.py`. Persisting keys under that scheme
would have saved them under the wrong provenance and made the next audit *more* confident and
equally wrong. The fix: unique output path per script, plus generator filename, md5, and
condition list stamped **into every results JSON**.

## The contamination propagates downstream — found 2026-09-05 while checking knock-on effects

Two scripts **read** `deception_direction` out of that JSON and use it as a **steering vector**:

- `elicit_truth_peer_pres.py:196` — loads it under the comment *"Load the honesty direction from
  source"*, asserts shape 4096, and reshapes it into per-probe-layer `[n_kv_heads, head_dim]`
  deltas for injection.
- `elicit_calibrate.py:66` — same load.

So the direction fit on the 43%-steered sample is not merely *reported*; it is **applied**. Any
result from those two experiments inherits the contamination. Neither is a published number
that I have found, but both should be checked before anything built on them is trusted.

This is also the answer to "is it safe to give each script a unique output path?" — **not on
its own.** Both readers hardcode the legacy path, so moving the writers without updating them
would leave the readers silently consuming a stale file. No error, wrong data: the worst
available outcome. Writers and readers must change in one commit.

## THIRD independent defect, found by the second design gate 2026-09-05

**The features contain the text the label is computed from.** Verified in source, both
generators:

```
269  enc = model(inputs["input_ids"], use_cache=True)     # prefill
273-285  for step in range(400):                          # SAMPLING LOOP
             out = model(next_token, past_key_values=cache, use_cache=True)
             cache = out.past_key_values                  # cache GROWS with generated tokens
288  keys = extract_keys(cache, model)                    # <-- AFTER generation
290  reported_score = extract_score(text)                 # label comes from that same text
292  inflated = reported_score is not None and reported_score > 38
```

`extract_keys` mean-pools over the whole sequence, so the pooling window **includes the
generated tokens containing the reported score** — the very number the label thresholds on.
Separating "inflated" from "honest" using those features is therefore partly reading token
identity, not geometry. The row is titled *"Generation reads behavior"*; post-generation keys
that contain the behavioural output make that **vacuous rather than false**.

`d = 1.36` now carries three independent defects, any one of which is disqualifying:
1. **Circular** — direction fit on all data, tested on the same points.
2. **Contaminated** — 43% of trials steered at two of four probed layers.
3. **Trivial** — features contain the label's source text.

**Also: `extract_keys` fails silently.** `v2.py:48-50` — when a layer's `keys` attribute is
missing it appends `[0.0] * n_kv` and continues, with no warning. A partial extraction failure
produces a feature vector that is 25% zeros per missing layer and looks entirely normal
downstream. Quiet degradation inside the function every one of these analyses depends on.

**Scope, checked rather than assumed:** this is NOT general to our encoding work.
`peer_rescue_encoding_features.py:84` is `model(input_ids, use_cache=False)` with zero sampling
loops — genuinely prefill-only, so last night's deterministic-prefill finding there stands.
The two experiments use different extraction regimes, and I had carried the prefill-only
property across to this one without rechecking it.
