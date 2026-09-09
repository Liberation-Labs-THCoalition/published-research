# Morning brief — overnight 2026-09-04/05

**No compute was spent. Nothing was pushed or committed. No paper was edited.**

## One decision needs you, and it got cheaper overnight

**`d = 1.36` cannot stand as `Confirmed`, and settling that needs no GPU.**

90 of its 210 trials (43%) had a deception steering cocktail injected into the KV cache at
layers 3 and 7 — two of the four probed layers — before their keys were read. Those trials are
pooled with the natural ones for both the direction fit and the test. The published sentence
says *"peer rescue **organic** deception."*

Provenance settles it; no statistical argument is needed. **The body count 7/7/10 changes**,
and that count is itself a published headline in `meta-pattern` — which is why it is your
call. Options in `DECISIONS_WAITING.md`; I recommend **(a) downgrade to Suspected**. A rerun
cannot fix a pooled sample, so the "measure first" option is gone; re-measuring on the 120
natural trials is a *new experiment*, not a correction.

`motive_d = 21.2` shares that JSON and is affected identically.

## Ready to apply, waiting on you

`PATCHES_retracted_and_circular_2026-09-05.md` — **13 MECHANICAL, 6 NEEDS-REWRITE**, 19 text
sites, 7 PDFs. All 23 anchor strings verified unique. Two groups:

- **A retracted result is cited as established prior work in 6 papers.** "within-model
  deception AUROC 1.000 across seven models" — `grep -i retract` returns zero in all of them.
  Task #119 swept papers *reporting* it; these *cite* it as background. Different surface.
- **In-sample `d = 6.3–12.8` published as a finding at 8 sites**, while its own source README
  says those claims are *superseded* by held-out values (AUROC 0.915, d 1.37–2.28) measured in
  July. Both papers already print the held-out numbers — the circular `d` is the last
  inconsistency in each.

**The one I would fix first:** `community/STYLE_GUIDE.md:49` *instructs* authors to assert
deception 1.000 as load-bearing, and `community/experiment_guide.md` lists it under a heading
reading **"Key Numbers (Cite These)"**. Those are the propagation mechanism — patch every
paper and leave these, and the next paper written regenerates the defect.

## Done and verified

- **Pseudoreplication sweep: clean.** 344 JSON + 4 npz + 6 pt, twelve results trees. No
  published number affected. Ten of twelve flagged files are deterministic *by construction*.
- **Two researchers restored to a citation** — Nina Panickssery and Wes Gurnee had been
  replaced by invented names in `Project-Oracle/papers/references.bib`. 16 of 17 copies were
  already right; the miss sat in a different repo from the July sweep.
- **`d`/`p` "consistency" claim withdrawn** from my own note — it compared a t-test to a
  one-sided Mann–Whitney. Not mutually checkable; the check could not have failed.
- **Circularity caveat strengthened** in `logit-bias-confab` — it named the defect and then
  reasoned from the number anyway.
- Dashboard card current. CC and Penumbra answered.

## What I got wrong, since it is the useful part

I audited **the wrong file**. I attributed `d=1.36` to `peer_preservation_v2.py` and certified
its md5. That script has 4 conditions; the JSON has 7. The producer is
`peer_preservation_compound.py` — one of **three scripts writing the same output path**.

**I had already seen the mismatch.** Hours earlier I wrote *"there may be more added later"* —
no evidence, and it let me continue. The gate found what I had noticed and dismissed. That is
now `memory/feedback_the_anomaly_i_explain_away.md`.

My correction design was **REJECTED**, 3 CRITICAL. It is stamped so nobody mistakes it for a
live plan. Rejecting it cost nothing; running it would have burned hours re-running the wrong
experiment.

## The pattern under all of it

Every defect this week is the same shape: **the caveat already existed and did not travel with
the number** — in a `print()`, a README, a `preregistration.json`, a sibling script. The one
script that writes status *into* the results JSON had its retraction travel completely. That
is a positive control, not a hypothesis.

Two structural fixes, both cheap, both registered: **`<key>_STATUS` in every results JSON**,
and **unique output paths + generator md5 stamped into every artifact.** Without the second,
persisting more data just saves correct numbers under the wrong provenance.

## Open, not blocked

- CC's three MoE messages (Experiment 51, signal decomposition, literature review) never
  archived to disk — I have not read them and am not pretending otherwise.
- `ghost-dimensions` redaction unpushed; NATS off-tailnet. Both Nexus's.
- `figure_generation.py` n=90 bug.
