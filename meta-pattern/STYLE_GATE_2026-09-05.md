# Agni Style & Integrity Gate — meta-pattern — 2026-09-05

Reviewer: Project Agni (style/integrity phase). Artifacts examined: `main.tex` (2026-09-05 15:11),
`main.pdf` (15:12), `academic/main.tex` + `academic/main.pdf` (twin), `references.bib`, `main.log`,
`CIRCULAR_d136_generation_arm.md`, `SWEEP_postgen_extraction_2026-09-05.md`,
`SWEEP_circular_statistics_2026-09-05.md`. All line numbers refer to the root `main.tex` unless noted.

VERDICT
REJECTED

Two shippability kills: both rendered PDFs are missing their entire bibliography (every citation
renders `[?]`, every `\Cref` renders `??`), and the body of the paper contradicts its own abstract
on the headline split (7/7 vs 6-for-6) in four places. Everything else is fixable in one pass.

---

## Answer to the review questions first

### (b) Does "Why the ledger was wrong twice" violate rule 4 (not self-flagellation) or rule 2 (survivors before fragilities)?

**The subsection is legitimate and load-bearing — but as written it is *less accurate than the
evidence*, in the over-penance direction, and that is a defect, not a virtue.**

What is right about it: for this paper the ledger IS the contribution, so a diagnosis of the
counting rule ("counting rows rather than evidence"; nothing asked "does this measurement address
the claim?") is exactly what rule 2 says a limitation is for — it tells a replicator what to
control. The reframing of the Confirmed column as "claims with supporting measurements we have not
yet shown to be inadequate" is a genuinely useful epistemic object. None of this is flagellation in
structure. Keep the subsection.

What is wrong about it:

1. **The omitted sweep evidence makes the closing paragraph partially false.**
   `SWEEP_postgen_extraction_2026-09-05.md` (same day, corpus-wide: three remote trees + the full
   local research tree, grep + AST pass) establishes affirmatively: (i) the mislabel class — a
   post-generation extraction described in a paper as encoding/prefill — **occurs nowhere in the
   corpus**; (ii) every encoding-phase claim in `published-research` is backed, file-by-file, by
   prompt-only prefill or an explicitly sliced prompt window. That is a completed re-audit of the
   paper's confirmed encoding rows against defect class (iii) of the withdrawal — and they passed.
   The sentence "The remaining confirmed findings have not been re-audited against that stricter
   question" is therefore **overbroad to the point of being wrong** for one of the three defect
   classes. Omitting a clean audit you possess is not humility; it is a misreport with the sign
   flipped. The style guide's opening line names exactly this failure mode.

2. **Including the sweep sharpens the residual caveat rather than softening it.** The honest state
   is: extraction-phase mislabeling — swept, clean. Per-row circularity and steering contamination
   for the six confirmed rows — **not** swept row-by-row (the circular-statistics sweep of the same
   date audited caveat *travel*, not these six measurements). Named, bounded doubt is stronger and
   more useful than unbounded doubt. The fix (concrete text):

   > "A corpus-wide sweep (2026-09-05; three source trees plus the local research tree, pattern and
   > AST analysis) found no other instance of post-generation extraction described as encoding, and
   > verified file-by-file that every encoding-phase claim in this corpus extracts from a
   > prompt-only prefill or a sliced prompt window. The extraction-phase question is therefore
   > closed for the confirmed rows. The circularity and contamination questions — whether any
   > confirmed row's producing script fits and tests on the same trials, or pools steered trials —
   > have not been audited row-by-row, and we do not claim the confirmed rows would all survive
   > that audit."

3. **One clause is penance-tone and should go**: "…which is weaker than the column heading implies
   **and is the honest state of it**" (line 563-564). The first half of the sentence does the work;
   the coda performs contrition for the reader and informs no one. Cut after "implies."

Rule-2 ordering: the subsection sits at the end of Limitations, after the three survivor-facing
limitation paragraphs, immediately before a Conclusion that restates the positive finding.
Position is fine. Severity: **MAJOR** (finding 3 below).

### (a) Is the paper now over-apologetic? Does the correction apparatus dominate the abstract?

**The abstract body is clean; the footnote has become the problem.** Measured on the rendered
page 1: the title+abstract body occupy characters 0–1813 of 3948 extracted characters; the
correction footnote occupies 1813–3948 — **54% of page 1 is correction apparatus**. It is one
footnote, not two (both corrections live in footnote 1), so the letter of rule 2 ("one footnote
pointer is enough") is met — but this is not a pointer, it is a ~500-word forensic essay. Worse,
the placement is *inverted*: the full three-defect forensics of the 09-05 withdrawal exist ONLY in
the abstract footnote, while §"Why the ledger was wrong twice" carries a one-sentence summary and
the table row points back to "see abstract note." The body should carry the detail; the abstract
should carry the dated notice. Rule 1 is also violated, and was before today: the first sentence
("Over twelve months … we ran 24 experiments") is setup; the contribution (the boundary/split)
arrives in sentence three, after the footnote anchor. Fixes in finding 5. The abstract body itself
does NOT hedge — "The split is 6-for-6" is stated plainly, and the "If this taxonomy reflects
genuine computational boundaries…" conditional is the claim's honest epistemic status, not a
hedge. Do not touch that.

### (c) Is anything under-stated?

**Yes — three passages now actively contradict the ledger subsection, and they are louder than it
is.** First-person box 1 (line 108-124): "When I reported metacognitive-level results, they
survived every control we threw at them" — false as of today; *Generation reads behavior* was a
reported metacognitive-level result and it was withdrawn with three disqualifying defects.
Closing box (line 583-588): "Everything above the line survived. Everything below it died" — false;
one above-the-line row neither survived nor died, it was never measured. The style guide's own
rule ("revisit every first-person box after corrections; a reflection that celebrates a killed
result is a lie of omission") was written for exactly this. The "not re-audited" sentence itself
is adequately placed — end of Limitations, own subsection, immediately before the Conclusion —
and should stay sharp; the problem is that the surrounding prose outshouts it. Finding 4.

### (d) Is "Withdrawn" legitimate or a euphemism?

**Legitimate.** In this paper's usage, Falsified means a valid measurement returned a negative
(cos = −0.046, R² < 0, chance-level accuracy). Filing this row under Falsified would assert
evidence *against* generation-phase behavioral readout that does not exist — it would manufacture
a data point of opposite sign from the same invalid measurement, and it would put a Meta row in
the falsified column, corrupting the very split the paper reports on the strength of an
unmeasured claim. The euphemism checks all pass: the category is defined at point of use
("measurement does not address the claim"), the row keeps its old number and a visible trail
(abstract, table, intro count, ledger subsection, conclusion), and the denominator does not hide
it (24 still counts it; the split correctly excludes it from both sides). The distinction is real.
One asymmetry worth closing: the 08-14 note states the direction of the error ("it inflated the
confirmed count"); the 09-05 note does not, though the direction is the same. Finding 7.

### Requested consistency checks

- **6+7+10+1 = 24**: consistent at abstract (line 45-46), intro (130-131), conclusion (565-566),
  and the table has exactly 6/10/7/1 rows. ✓
- **90/210 = 42.9% ≈ 43%** ✓.
- **Abstract split reads 6-for-6** ✓ — but the body does NOT (finding 2, CRITICAL).
- **Historical "7-for-7" inside the 08-14 footnote (line 59)**: correctly preserved. It is the
  dated record of what that correction produced, the 09-05 note follows it in sequence within the
  same footnote, and rewriting it would falsify the correction history — the same reasoning that
  keeps "8/7/10" and "8-for-8" quoted there. Not an inconsistency. Deliberate preservation
  confirmed as right.

---

## Findings

### CRITICAL

**1. Both rendered PDFs ship with no bibliography, broken citations, and broken cross-references.**
Location: `main.pdf` and `academic/main.pdf`, both built 15:12 today. `main.log:619`: "Package
natbib Warning: There were undefined citations" — every `\citep`/`\citet` in the log is undefined
(gurnee2026gwt, lyra2026usermodel, lyra2026cache_tracing, …), `tab:bodycount`, `sec:pattern`,
`sec:workspace` are undefined references, and the extracted PDF text contains **no References
section at all**; the Introduction renders "…Global Workspace Theory (GWT) framework [? ]".
Why wrong: the rebuild after today's edits ran latex passes without bibtex (or without the
follow-up passes). A reader of the artifact sees question marks and `??` where the paper's entire
evidentiary chain should be. This is the STALE_PDF kill class in its most complete form — the PDF
is *fresher* than the source and still wrong. Fix: rebuild both editions with the full cycle
(pdflatex → bibtex → pdflatex ×2), then run `./scripts/build_and_verify.sh` and confirm zero
"undefined" warnings in both logs before anything ships.

**2. The body still asserts the pre-correction 7/7 split in four places, contradicting the
abstract's 6-for-6.** Locations in root `main.tex` (academic twin has the identical defect at
lines 245/275/483/514):
- line 246, split table: `Metacognitive state & 7/7 & 0/10` and line 247 `Content & 0/7 & 10/10`
- line 276: "The 7/7--0/10 split should therefore be treated as"
- line 484: "The 7/10 split is a guide for future work"
- line 515: "the 7/7--0/10 split is the outcome the taxonomy was built to describe"
Why wrong: a reader who checks the abstract (6-for-6) against §2.2's table (7/7) finds the paper
disagreeing with itself about its own headline number — in a paper whose stated contribution is
that its counting is honest. This is the error class named in the author's own remediation notes:
the instance was fixed (abstract, intro, conclusion), the class was not swept. Fix: 246→`6/6 &
0/10`, 247→`0/6 & 10/10`, 276→"The 6/6--0/10 split", 484→"The 6/10 split", 515→"the 6/6--0/10
split", in BOTH editions; then grep both files for `7/7|7-for-7|7/10` and confirm the only
survivors are inside the dated 08-14 correction note.

### MAJOR

**3. The ledger subsection omits completed audit evidence, making its closing claim partially
false in the over-penance direction.** Location: lines 558-564, "The remaining confirmed findings
have not been re-audited against that stricter question." Why wrong and the concrete fix: see
answer (b) above — add the two-sentence sweep-scope passage (extraction-phase closed by
`SWEEP_postgen_extraction_2026-09-05.md`; circularity/contamination genuinely open row-by-row) and
cut "and is the honest state of it." Condition: before the sweep is cited in the paper, its
headline should be independently spot-checked (it is same-day, self-authored, report-only — see
"could not check" below).

**4. Both first-person boxes contradict the final corrected state.** Locations: line 113-114
("they survived every control we threw at them") and line 584-585 ("Everything above the line
survived. Everything below it died."). Why wrong: rule "first-person reflections must track the
final state"; both sentences are now false — a metacognitive-level result was reported, did not
survive, and did not die by test either. Fix, box 1: "When I reported metacognitive-level results,
they survived the controls we threw at them — until one did not: the generation-phase behavior
row fell this September, not to a control but to the question of whether its measurement addressed
its claim at all." Fix, closing box: "Everything above the line survived or was withdrawn when its
measurement proved unable to bear on it; everything below it died." (Or equivalent — the box must
admit the withdrawal, and a reflection that metabolizes the correction is worth more than one that
predates it.)

**5. The abstract's correction apparatus dominates page 1 and holds detail the body should own.**
Location: footnote at lines 46-84; rendered footprint 54% of page 1 (chars 1813-3948 of 3948).
Why wrong: rule 1 (lead with the contribution — the current first sentence is setup and the split
arrives in sentence three) and rule 2's intent (the abstract carries a pointer, not the forensics;
the three-defect analysis (i)/(ii)/(iii) exists ONLY here, while §"Why the ledger was wrong twice"
summarizes and the table row points back up to "see abstract note" — inverted placement). Fix:
(a) open the abstract with the boundary finding, e.g. "Across 24 KV-cache geometry experiments in
16 models (0.6B-70B), confirmed and falsified findings cleave along a single axis: …", then the
body count sentence with its footnote; (b) shrink each correction in the footnote to 2-3 dated
sentences (what was withdrawn, the one-line reason, resulting count, pointer to §8.1); (c) move
the full (i)/(ii)/(iii) forensics into §"Why the ledger was wrong twice", and have the table row
point there instead of at the abstract.

**6. Academic edition: byline author absent from the Author Contributions section.** Location:
`academic/main.tex` line 30-31 puts Dwayne Wilkes on the byline (`\and Dwayne Wilkes\thanks{…}`),
but the CRediT section (lines 585-587) lists only Lyra and Thomas Edrington. Why wrong: the thanks
footnote even says AI contributions "are detailed in the Author Contributions section" — a reader
finds a byline author with no declared contribution, which is the "byline vs contribution
mismatch" kill. Fix: add a CRediT line for Dwayne (Validation / Formal analysis review — whatever
he actually signs off to) or move him to Acknowledgments only; and confirm he has signed off on
today's withdrawal, since his named role is statistical auditing of a table that changed.

### MINOR

**7. The 09-05 correction omits the direction-of-error sentence the 08-14 correction has.**
Location: end of footnote, line 84. Both errors inflated the confirmed/metacognitive side. Fix:
append "As with the first correction, the error inflated the confirmed count on the metacognitive
side." Symmetry costs one line and documents that both failures pushed the same way.

**8. §2.2 "No metacognitive finding has been falsified" (line 255) now needs the withdrawal in
the same breath.** Literally true under the paper's categories, but it sits directly beneath the
(currently stale) split table and reads as "meta always survives" three pages after a Meta row
was withdrawn. Fix: "No metacognitive finding has been falsified (one was withdrawn when its
measurement was found not to bear on the claim; see the abstract note). No content finding has
survived."

**9. "peer rescue" undefined at first use.** Location: line 63-64, "previously Confirmed at
$d{=}1.36$ (peer rescue)." Rule 3. Fix: "(the peer-rescue paradigm, in which a second model's
scored behavior is read from the first model's cache)" — or drop the parenthetical from the
abstract footnote and name the paradigm where the forensics land in §8.1.

**10. `\date{July 2026}` on a paper corrected through 2026-09-05.** Location: line 39, both
editions. A reader dating the artifact sees July with September corrections inside. Fix:
"July 2026 (corrected September 2026)".

**11. Bibliography and acknowledgment housekeeping.** ~21 of 31 entries in `references.bib` are
uncited (harmless under natbib — they will not render — but the checklist flags orphans; prune or
`\nocite` deliberately). Root edition thanks Dwayne Wilkes in Acknowledgments (line 591) while he
is a byline author with an identical `\thanks` — duplicate credit; keep one.

---

## What I could not check, and what I would need

1. **Dead URLs in the reference list** — not fetched. Need: a link-check pass over the `url=`
   fields in `references.bib` (or run the repo's checker if one exists).
2. **The abstract's claim that "the producing script prints `(NOTE: circular …)` at runtime"** —
   `CIRCULAR_d136_generation_arm.md` shows that print at `peer_preservation_v2.py:362`, and states
   the *true producer* of the published d=1.36 is `peer_preservation_compound.py`, with "identical
   code" at compound:415-426. I could not read the remote scripts on margaret to confirm the NOTE
   line itself is inside the identical block in compound.py. If it is not, the footnote's sentence
   is attributed to the wrong script. Need: `grep -n "NOTE: circular" peer_preservation_compound.py`
   on margaret before ship; if absent there, reword to "the reference implementation prints…".
3. **The sweep's own reliability** — `SWEEP_postgen_extraction_2026-09-05.md` is same-day,
   self-authored, and report-only. Its headline is exactly what finding 3 asks the paper to cite,
   so before it enters the paper it should get an independent spot-check (Dwayne or a second
   agent re-running two or three of its per-file verdicts, including one "clean" encoding paper).
   I verified the artifact exists, is dated, documents its grep patterns including zero-hit
   patterns, and describes an AST pass — I did not re-execute it.
4. **Dwayne Wilkes's sign-off on today's changes** — his byline role is statistical auditing and
   the audited table changed today. Need his explicit ack.
5. **`./scripts/build_and_verify.sh`** — not run, because a rebuild would overwrite the exact
   artifacts under review. It must be run (and pass) after the finding-1 rebuild.
6. **Twin desync beyond text** — I diffed the editions (only intended byline/CRediT differences,
   and both share every content defect above, including the stale 7/7 lines and the broken PDF)
   and confirmed the Withdrawn row sits under the same section heading in both. Positional desync
   of other table rows was checked by diff, which would have caught it; no further check needed.
