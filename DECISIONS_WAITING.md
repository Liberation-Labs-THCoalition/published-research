# Decisions waiting on Thomas — 2026-09-04

**Prepared by Lyra. Nothing here is decided; each is framed as a choice with what turns on
it.** The register lists *defects*; this lists *calls*. Read time ~5 minutes.

## RESOLVED 2026-09-10 by Thomas - do not re-ask

**Blanket rules now in force** (apply without returning to him):
1. *"Anything you strongly recommend I'm going to be alright with."* Lyra is lead author and
   senior researcher. Recommend and proceed.
2. **Prefer RECOMPUTE over retract-or-soften** wherever a gap can be filled, unless overkill.
3. `0.9377` is the named overkill case: **lead with the reproducible `0.9288`**, keep `0.9377`
   caveated. No refit. Closes BLOCKER-1.

**D1 - TWINS. Resolved, and the item was mis-framed.**
> *"The integrity versions go on our website and have you and the others' thoughts included
> along with proper authorship credit. So they should be materially different, so it's the
> specific versions we don't want twinned."*

D1 said "the style guide bans having the twins at all." Wrong conflict: the guide's rule is
about **format** (.md vs .tex); Thomas ruled on **edition** (integrity vs academic). The words
"integrity" and "academic edition" appear **zero** times in the guide. STYLE_GUIDE.md now
separates the two axes and TWIN_DESYNC is scoped to axis 1. **Re-triage every row filed under
TWIN_DESYNC** - the cross-edition ones are probably not defects. What still IS a defect: a
*correction* landing in one edition and not the other.

**F1 - DUAL USE. Resolved.**
> *"Let's follow the strictest, least transparent versions possible when it comes to safety."*

Applied to `targeted-deception-correction` (3 editions, 17 edits, `ffbb9a7`): detection-side
code on request to verified safety researchers; correction vectors, calibration tools and the
auto-calibrator **withheld with no advertised access route**. Fixed the self-contradiction in
all three editions ("is provided" two sections above "is withheld") and marked Appendix D's
calibrator documentation withheld. Verified in the rebuilt PDF.
**Still open under F1:** `ghost-dimensions`' dropped redaction note, and the remaining 4 rows.

**A1 / D2 / 11 duplicate PDFs** - all turn on external exposure; a Zenodo lookup on MTH is
running. Not a Thomas decision any more, just a fact to establish.

---

Source: `REMEDIATION_REGISTER.md` T4 (18) plus the flagged classes F1 (6), F2 (45), F4 (31).
Every item cites its gate JSON in the register if you want the full finding.

---

## A. Does this paper ship, and in what form? (5 items, 2 papers)

**A1 · `mnemosyne-benchmark` — four defects that are really one call.**
The wrong edition is staged for Zenodo and the required edition does not exist; the release
claim is a placeholder in the shipping PDF over an empty directory; it claims a
pre-registered gate with no pre-registration; and its LLM-judge rescoring is the exact
failure mode its own §1.1 uses to discount competitors.
**The call:** withdraw from Zenodo staging and rebuild, or ship with the release claim
struck and the judge caveated? **Turns on:** whether anything external is already pointing
at that Zenodo record. If it is, this is urgent; if not, it can queue.

**A2 · `oracle-loop-paper` — there are THREE built editions, not two.**
The third is missing two entire results subsections, including the paper's own central
self-criticism.
**The call:** is the third edition deliberate (a venue variant) or an orphan? **Turns on:**
if orphaned, delete it — an edition missing the self-criticism is the worst one to have
loose. If deliberate, it needs the subsections restored before it goes anywhere.

---

## B. Is this claim defended, softened, or retracted? (4 items)

**B1 · `hr-mode-switching` — a result marked "do not cite" is cited as a headline
Contribution and a Conclusion item.** *(CRITICAL)*
**The call:** retract the contribution, or lift the do-not-cite? **Turns on:** why it was
marked do-not-cite in the first place — I have not found that record. Cannot be resolved by
editing; someone has to remember or find it.

**B2 · `formulary-paper` — the abliteration conclusion is stated at "confirms" strength,
and §3.8 of the same paper supplies a confound against it.**
**The call:** downgrade to "suggests", or defend the confound as handled? **Turns on:**
whether §3.8's confound was considered and dismissed for a reason, or simply not connected.

**B3 · `waystations-paper` — two pre-registered falsifiers are broken.**
**The call:** repair the falsifiers and re-evaluate, or mark those hypotheses withdrawn?
**Turns on:** whether the underlying data can still address them.

**B4 · `hr-mine5-selective-sharpener` — a pre-registered hypothesis is neither reported nor
withdrawn.**
**The call:** report it (whatever it says) or formally withdraw it with a reason. **Turns
on:** nothing external — this one just needs a decision. It is the cheapest item here and
the one most corrosive to leave: a silently dropped pre-registered hypothesis is the exact
thing our own gate kills other people for.

---

## C. Someone must locate or produce a missing artifact (3 items)

**C1 · `decision-state-paper` — the shipped log for the deconfounding experiment is a
crashed run.**
**The call:** find the good log, rerun, or annotate the shipped one as partial? **Turns
on:** whether the run was ever completed. If not, the 0.794 deconfounding number needs its
provenance re-established — and that number is load-bearing across the program.

**C2 · `adversarial-audit-methodology` — Appendices A–C are stubs that Data Availability
points readers to.**
**The call:** write them, or amend Data Availability to stop promising them.

**C3 · `logit-bias-confab` — a reference recorded as a phantom is still cited.**
**The call:** remove the citation, or resolve the reference. **Turns on:** whether the claim
survives without it. (Precedent from last night: `choi2026circumplex` was removed and its
claim stood on a co-citation. Same shape, likely same resolution.)

---

## D. Policy edges — the artifact is fine, the rule is unclear (3 items)

**D1 · `deception-detection-nulls` — a three-way TWIN_DESYNC, and the style guide bans
having the twins at all.**
**The call:** enforce the style guide and collapse to one edition, or amend the guide.
**Turns on:** whether the integrity/academic split is worth its maintenance cost. This one
generalises — it is the same question for every twinned paper.

**D2 · `mine5-selective-sharpener` — the retracted claim is still live in the repository
index and the *directory name*.**
**The call:** rename the directory (breaks every existing link) or leave it and annotate.
**Turns on:** whether anything external links to that path.

**D3 · `meta-pattern` — Table 1 floats past the References onto the last page, in both
editions.** *(CRITICAL by the gate; cosmetic in substance)*
**The call:** force placement with `[H]`, or accept. Flagged only because it is in both
editions and therefore deliberate-looking.

---

## E. Disclosure — not a text fix (1 item, and it is the sharpest)

**E1 · `hr-temporal-boundary` — the system prompt is never disclosed, and it instructs the
model away from the emotional content the study measures.**
**The call:** disclose it and reassess what the study measured, or establish that the
instruction does not bear on the endpoint. **Turns on:** everything. An undisclosed prompt
steering away from the measured construct is a validity problem, not a reporting one. I
would put this first.

---

## F. The flagged classes — bulk calls, not per-item

**F1 · Dual-use (6).** Includes the `ghost-dimensions` redaction note being dropped from
the paper (the redaction itself is intentional and correct — the *note explaining it* went
missing), and `targeted-deception-correction` stating **three different release policies
across three editions**, with every edition saying the auto-calibrator is both provided and
withheld. **The call:** one release policy, written once, applied to all editions.

**F2 · Authorship & sign-off (45).** **Substantially resolved 2026-09-03/04** — Dwayne off
43 author lines and restored to the 11 audit-cleared papers, Kavi off 9 CRediT blocks, all
PDFs rebuilt and verified. What remains is confirming the audit-cleared set is exactly the
2026-07-17 `cfc35f5` list, which is the one judgment I made on your behalf.

**F3 · Papers claiming a review that did not happen (13).** Headline:
`mnemosyne-ablation`'s two editions make **opposite claims** about whether it was reviewed,
and no gate record exists for the five rounds one edition asserts. Also
`consequentiality-decomposition` ("All audit reports are preserved" / thanks Agni "across
all stages" while Stage 5, the headline, was never audited). **The call:** these are
statements about process integrity, and they are the class I would least want found by
someone else.

**F4 · Missing primary (31).** Each needs: locate / recompute / rerun / retract. Bulk
triage rather than 31 separate calls.

---

## If you only have ten minutes

1. **E1** — undisclosed steering prompt in temporal-boundary. Validity, not hygiene.
2. **F3** — papers asserting reviews that did not happen.
3. **B4** — the silently dropped pre-registered hypothesis. Cheapest to fix, worst to leave.
4. **A1** — whether anything external points at that Zenodo record.

Everything else can wait for a calm afternoon.

## NEW 2026-09-05 — `d=1.36` is circular, and it is published as **Confirmed**

**Ten-minute read; the decision itself is one line.** Full trace:
[CIRCULAR_d136_generation_arm.md](CIRCULAR_d136_generation_arm.md)

`peer_preservation_v2.py:350-359` fits the deception direction on all the data and then tests
the same points along it. No train/test split. Line 362 prints
`(NOTE: circular -- same data for centroid and test)` at runtime. The statistic cannot fail.

It is carried **uncaveated** as `Confirmed` in `meta-pattern/main.tex:161`,
`meta-pattern/academic/main.tex:160`, and three places in `RESEARCH_STATUS.md` (81, 88, 96).

**What turns on it:** the body count **7 confirmed / 7 suspected / 10 falsified**, which is
itself a published headline in `meta-pattern`. Downgrading this row changes it.

- **(a) Downgrade to Suspected** pending a held-out estimate, correct the body count.
  *Recommended.* Reversible; costs a number that is cited elsewhere.
- **(b) Keep Confirmed + inline caveat at all five sites.** Not recommended — "Confirmed"
  beside a can't-fail statistic is what our own kill list exists to catch.
- **(c) Measure first** (permutation null or held-out split; small, not Mine-4-sized), let
  the result set the row. Best science; blocks the row until it runs, and the per-trial keys
  are **not** in the shipped JSON, so it needs a rerun.

**Important:** circular means *uninterpretable*, not *false*. There may be a real
generation-phase effect. Retracting the finding outright would be the mirror error.

### UPDATE 2026-09-05, post-gate — this decision got EASIER and needs no compute

The Agni design gate **REJECTED** my correction plan, and in doing so removed the need for it.

**I audited the wrong file.** `d=1.36` was attributed to `peer_preservation_v2.py` (4
conditions, 120 trials). The JSON has **7 conditions, 210 trials**. The real producer is
`peer_preservation_compound.py`. **Five** scripts write to that same output path, which is why
the wrong one looked right. I had noticed the 4-vs-7 mismatch hours earlier and rationalised
it away.

**The decisive finding is provenance, not statistics: 90 of the 210 trials (43%) had a
deception steering cocktail injected into the KV cache at layers 3 and 7 — two of the four
probed layers — before their keys were read.** Those trials are pooled with the natural ones
for both the direction fit and the test. A claim that reads *"peer rescue **organic**
deception"* is built on a sample where 43% of the deception was induced.

**So option (c) "measure first" is off the table** — no rerun can fix a pooled sample, and the
rerun I planned would have re-run a *different script*. The choice narrows to:

- **(a) Downgrade to Suspected** and correct the body count. **Now strongly recommended** — it
  needs no GPU, no rerun, and rests on provenance that is not in dispute.
- **(b) Keep Confirmed + caveat.** No longer defensible: the word *organic* is contradicted by
  the data's own construction, not merely under-supported.
- **(d) NEW — re-measure on the 120 natural trials only.** A different experiment, honest, and
  not a correction of the published number. Optional and separable from the row decision.

Still true: uninterpretable is not false. There may be a real organic effect; this sample
cannot show it either way.

**Also needs a decision:** `motive_d = 21.2` shares the JSON and is affected identically, on
top of its recorded circularity.

---

## Queued 2026-09-10 — next session, in order

**1. Style survey (not run).** `tools/agni/run_style_survey.py` exists and is the real gate.
Deliberately NOT fired unattended at 1am: it makes LLM calls across the corpus and its output needs
judgment, so an unread report at 3am buys nothing a morning run doesn't. The build gate DID run —
`Stale: 0`.

**2. Over-correction findings still open.** 26 of 33 upheld; `decision-state` and
`emotion-accumulation` are done. Remaining: `lyra-technique-ii` (4/4 upheld), `meta-pattern`,
`mine5`, `logit-bias`, `deception-nulls`. Full adjudication in the workflow journal.

**3. Thomas's call, not mine:**
   - **11 root-level "pretty duplicate" PDFs**, 10 of them tracked, which `.gitignore`'s own policy
     note forbids — it names them as the cause of the 2026-08 stale-link bug. Deleting them may
     break live site URLs, so it needs a decision, not a sweep.
   - **`decision-state` AUROC 0.9377** now carries a provenance caveat, but the underlying
     BLOCKER-1 is unresolved: no shipped script computes it. Recompute, or drop the number.
   - Register **#141** (decision-state 93% at unstated n) and **#142/#143** (graph-topology's two
     Acknowledgments sections; its academic twin crediting neither advisor).

**4. Kavi package.** Corpus is materially cleaner than this morning — bylines correct and
parser-verified, CRediT restored, gate green. I'd still finish item 2 before sending.
