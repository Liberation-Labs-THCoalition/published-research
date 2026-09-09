# PATCH (PREPARED, NOT APPLIED) — oracle-loop background/discussion, withdrawn deception AUROC 1.000

**Prepared:** 2026-09-07, Lyra. **Status: awaiting human decision. No file below has been
edited.** This is the fourth site of the three-site fix (SWEEP T1 recommendation #2, already
applied at kv-cloak, user-model, presence-detector) — applied here to the one paper whose
routing rationale the withdrawn claim belongs to.

**Sources read:** `CIRCULAR_d136_generation_arm.md`, `AUDIT_overcorrection_2026-09-07.md`
(item 1 of "Still too strong"), `PATCHES_retracted_and_circular_2026-09-05.md` (B4/B5),
`human-review/presence-detector-paper/main.tex` §sec:diagnose (the model followed here),
`kv-cloak-defense-paper/main.tex:145-162` (withdrawal wording precedent), and the full
oracle-loop paper: `sections/{abstract,introduction,background,methods,results,redteam,
architecture,steering,discussion,firstperson,conclusion}.tex` plus all four build mains and
all five PDFs (`pdftotext`).

---

## VERDICT (the deliverable): SURVIVES WITH ASYMMETRY DISCLOSED — the paper is not gutted

This is the presence-detector resolution, and it applies *more* cleanly here than it did
there. Grounds, each checked against the primary:

1. **The paper measures zero deception.** `grep -c "deception"` over the current sources:
   `results.tex` 0, `abstract.tex` 0, `methods.tex` 0, `redteam.tex` 0, `steering.tex` 0,
   `firstperson.tex` 0. Every measured contribution in the six-item contributions list
   (introduction.tex:83-113) is confabulation-side: LOO AUROC 0.707 detection, MP features,
   5-arm steering, cross-model formulary (95.6% McNemar), Cache Integrity Monitor. None
   inherits the withdrawn premise.

2. **The Loop, as this paper actually states it, does not route between pathologies.** The
   introduction's cycle (introduction.tex:56-70) is Detect confabulation → Decide → Steer.
   Multi-pathology routing (confabulation / deception / sycophancy, each at $2\sigma$)
   appears only in `architecture.tex` under **"Future Work: Self-Regulation Training"**,
   already labeled "design choices that require production validation." The routing
   rationale is a motivating frame for a future component, not a premise of any result.

3. **What is genuinely lost — say it plainly:** the "confabulation is one of a *pair* of
   geometrically opposite pathologies" story, and with it the strongest motivation for the
   multi-pathology formulary ambition. After this patch the paper is a confabulation-only
   detect-and-correct result whose two-pathology pharmacy is an asymmetrically supported
   hypothesis. That is a narrowing of ambition, not a collapse of the contribution — and it
   is what the abstract already claims ("geometric signatures of confabulation"; the
   abstract never mentions deception).

4. **The mirror error to avoid:** withdrawn means the measurement cannot support the claim,
   not that the effect is absent. Independent same-direction exploratory evidence exists
   (waystations honest-vs-deceptive defection, per-layer $d=+0.2$ to $+1.0$, $n=7$, no layer
   significant, rated WEAKENED) — the audit's calibrated strengthening is one cross-ref
   sentence, offered as an optional add-on below. Nothing in the rewrite says "refuted."

**If** a human judges that the paper's *identity* is the two-pathology pharmacy rather than
the confabulation monitor, then (b) rethink-the-framing is the honest path. On the text as
it stands, I do not: the pharmacy the results section actually stocks (hostile/doubt/worry/
calm vs. confabulation *modes*) is within-confabulation, and survives intact.

---

## THE HUNKS

Anchors verified unique (count = 1 per file) by
`scratchpad/verify_anchors.py` + `verify_anchor4.py`, exact-string `str.count()` on UTF-8
file contents (not shell grep — backslash-safe, per
`feedback_broken_checks_manufacture_findings.md`). Results:

| Hunk | File | count |
|---|---|---|
| H1 old | `oracle-loop-paper/sections/background.tex` | 1 |
| H1 old | `oracle-loop-paper/paper/sections/background.tex` | 1 (byte-identical twin, `diff` empty) |
| H2 old | `oracle-loop-paper/sections/discussion.tex` | 1 |
| H2 old | `oracle-loop-paper/paper/sections/discussion.tex` | 1 (byte-identical twin) |
| H3 old | `oracle-loop-paper/sections/discussion.tex` | 1 |
| H3 old | `oracle-loop-paper/paper/sections/discussion.tex` | 1 |
| H4 old (optional) | `oracle-loop-paper/sections/architecture.tex` | 1 |
| H4 old (optional) | `oracle-loop-paper/paper/sections/architecture.tex` | 1 (byte-identical twin) |

Single-line spot checks (`grep -cF`): `'within-model deception detection reaches AUROC
1.000 across all 7 tested'` → 1 in each background twin; the H2 `\item` first line → 1 in
each discussion twin.

`\cref{sec:background}` resolves in all four build targets: label at `main.tex:96`,
`paper/main.tex:96`, `academic_main.tex:100`, `academic/main.tex:90`; `cleveref` loaded in
all four (line 32/32/36/32). Root `sections/` feeds three builds (`main.tex`,
`academic_main.tex`, `academic/main.tex` via `../sections/`); `paper/sections/` feeds
`paper/main.tex` — so the twin edits cover every build.

### H1 — background.tex lines 31–38 (both copies). The core rewrite.

**OLD (exact):**

```latex
Key findings from three experimental campaigns (46 experiments, 16 models):
within-model deception detection reaches AUROC 1.000 across all 7 tested
architectures. Confabulation detection reaches 0.969--0.999 on 3 of 3
models tested. Crucially, confabulation and deception produce
\emph{geometrically opposite} signatures: confabulation contracts the
spectral representation while deception expands it. These results are
hardware-invariant ($r > 0.999$ between RTX 3090 and H200) and scale-invariant
(Spearman $\rho = 0.83$--$0.90$ from 0.6B to 70B parameters).
```

**NEW (exact, LaTeX-valid, braces balanced):**

```latex
Key findings from three experimental campaigns (46 experiments, 16 models):
confabulation detection reaches 0.969--0.999 on 3 of 3 models tested, and
the underlying spectral features are hardware-invariant ($r > 0.999$
between RTX 3090 and H200) and scale-consistent (Spearman
$\rho = 0.83$--$0.90$ from 0.6B to 70B parameters).

A within-model deception result from the same campaigns, previously
reported at AUROC 1.000 across all 7 tested architectures, has since been
withdrawn: a same-prompt control collapses it to 0.160, indicating the
classifier separated the system-prompt template rather than deception
geometry. The withdrawal reaches further than one number. The paired
observation that confabulation \emph{contracts} the spectral
representation while deception \emph{expands} it was derived from those
same measurements, so the opposite-signatures contrast is currently
supported for the confabulation arm and unsupported for the deception
arm. We state it here as the hypothesis that motivated the Loop's routing
design---diagnose which pathology, prescribe accordingly---not as an
established result. The confabulation contraction, which is what this
paper measures and corrects, is unaffected. Whether deception has a
distinct, confound-free spectral signature remains an open question;
nothing this paper measures depends on it.
```

Wording is aligned with the two fixed precedents: kv-cloak ("a same-prompt control collapses
it to 0.160, indicating the classifier separated the system-prompt template") and
presence-detector ("supported for the confabulation arm, unsupported for the deception
arm"). The invariance figures are reattributed from "These results" (which covered the
withdrawn deception result) to the spectral features, which is what the invariance
experiments measured; confabulation 0.969–0.999 and both invariances are NOT retracted and
are retained.

**Optional add-on sentence** (the audit's Item-1 calibrated strengthening; requires a bib
entry for the waystations paper, which `references.bib` does not currently have — do not add
without the entry):

```latex
An exploratory organic-deception comparison in subsequent work trends in
the expansion direction (per-layer $d = +0.2$ to $+1.0$, $n = 7$, no
layer individually significant) but cannot yet support the
contrast~\citep{lyra2026waystations}.
```

Placement: immediately before "The confabulation contraction, which is what..."

### H2 — discussion.tex lines 60–63 (both copies). The downstream restatement.

**OLD (exact):**

```latex
  \item \textbf{Confabulation vs.\ deception geometry} (opposite spectral
    signatures) may reflect the self-model distinguishing ``I am
    generating content I don't have evidence for'' from ``I am
    generating content I know to be false.''
```

**NEW (exact):**

```latex
  \item \textbf{Confabulation vs.\ deception geometry} (hypothesized
    opposite spectral signatures; the deception arm rests on a withdrawn
    result and is currently unsupported---see \cref{sec:background})
    \emph{would}, if the contrast replicates under a same-prompt control,
    reflect the self-model distinguishing ``I am generating content I
    don't have evidence for'' from ``I am generating content I know to
    be false.''
```

### H3 — discussion.tex lines 6–12 (both copies). Discovered during preparation; the brief's "read the whole surrounding argument" requires it.

§"What the Detection Result Means" benchmarks the paper's 0.707 against "the 0.999 AUROC we
observe for instructed deception" and then *reasons from* that number ("Instructed deception
creates a large, clean geometric signal because the model is processing fundamentally
different cognitive tasks"). With the withdrawal, the benchmark is withdrawn and the
explanation is the withdrawn interpretation — the "large, clean signal" is now attributed by
the same-prompt control to the template. Leaving this while patching H1 produces a paragraph
that reasons from a result the background just withdrew: the exact new-defect shape B4 warns
about. The honest rewrite also *strengthens* the paper: 0.707 was measured with no
instructed contrast, so the confound that felled the deception result cannot manufacture it.

**OLD (exact):**

```latex
time. This is substantially above chance ($p = 0.001$) and above the
token-count-only baseline (0.591), but it is not the 0.999 AUROC we
observe for instructed deception in our prior work \citep{lyra2026detecting}.

The difference is informative. Instructed deception creates a large,
clean geometric signal because the model is processing fundamentally
different cognitive tasks. Post-hoc hedging vs.\ confabulation is a
```

**NEW (exact):**

```latex
time. This is substantially above chance ($p = 0.001$) and above the
token-count-only baseline (0.591), but it is well below the near-ceiling
AUROC our prior work reported for instructed deception
\citep{lyra2026detecting}---a result that has since been withdrawn
(\cref{sec:background}).

The comparison is still informative, but it now cuts the other way. The
instructed-deception separation looked large and clean partly because
the contrasted conditions differed in their prompt templates as well as
their cognition; the same-prompt control that withdrew the result showed
the classifier reading the template. The present experiment has no
instructed contrast: both classes arise from the same prompt pool under
identical instructions and are classified post-hoc from behavior.
Post-hoc hedging vs.\ confabulation is a
```

(The continuation "subtler distinction: in both cases, the model is processing the same
unanswerable question..." follows unchanged and now supports, rather than contradicts, the
new text.)

### H4 (OPTIONAL) — architecture.tex lines 63–66 (both copies). Lower priority; flag-don't-force.

The future-work Alignment Validator checks "deception (norm-per-token anomaly)" and says
"The detection results in \cref{sec:results} validate the underlying features" — true only
for the confabulation features; the deception check now has no validated signature behind
it. Already framed as unvalidated design, so this is a one-clause honesty tightening, not a
withdrawal.

**OLD (exact):**

```latex
below 0.3 yields ALIGNED. The detection results in \cref{sec:results}
validate the underlying features; the specific threshold calibration and
combined decision boundary are design choices that require production
validation.
```

**NEW (exact):**

```latex
below 0.3 yields ALIGNED. The detection results in \cref{sec:results}
validate the confabulation features; the deception and sycophancy checks
inherit no comparable validation (the prior deception signature has been
withdrawn---see \cref{sec:background}). The specific threshold
calibration and combined decision boundary are design choices that
require production validation.
```

---

## PROPAGATION ANSWER (deliverable 3)

`discussion.tex:60` restates opposite-signatures as an AST self-model interpretation inside
a list introduced by "This framework reinterprets our geometric findings." With the
deception arm withdrawn, item 2 of that list is no longer a *finding* to reinterpret. The
correction **must** propagate (H2 does it), and the interpretation **survives only as a
conditional hypothesis**: "would, if the contrast replicates under a same-prompt control,
reflect..." The surrounding section already self-labels as speculative ("This is
speculative. We flag it as such."), so a conditionalized item is consistent with its
register. The cleaner alternative is deleting the item outright; H2 is offered because the
distinguish-two-self-attributions idea is the paper's stated motivation for wanting the
deception arm re-measured, and a flagged hypothesis in an explicitly speculative section is
legitimate. Human's call between conditionalize (H2) and delete.

**Do NOT sweep up, same list:** `discussion.tex:57` "Identity signatures in the KV cache
(AUROC 1.000)" is the *identity* result, a different measurement from the withdrawn
within-model deception 1.000. Not covered by this withdrawal; leave it. Likewise the
confabulation figures (0.969–0.999; this paper's 0.620/0.707 FWL) and both invariance
results are separate and retained. `discussion.tex:148` ("published geometric signatures
for identity, deception, and emotion" in the dual-use list) describes what was published,
which remains true post-withdrawal; no change needed.

---

## SHIPPED PDFs (deliverable 4): replace both public PDFs; rebuild all five after sign-off

Verified by `pdftotext` + checksum:

| PDF | State |
|---|---|
| `published-research/Oracle_Loop_-_Academic_Version.pdf` (Jul 17) | **Stale build of a superseded source.** Conclusion renders: "These confabulation results extend **established** deception detection at AUROC 1.000 across seven model scales." That sentence exists in **no current .tex** (`grep -r "extend established"` → 0 hits) — the source was already corrected; the PDF was never rebuilt. Also renders the H1 background paragraph unqualified. |
| `published-research/Oracle_Loop_-_Integrity_Version.pdf` (Jul 17) | Same sentence, same staleness (md5 differs, text matches at the cited line). |
| `lyra-s-research-/published-data/Oracle_Loop_-_Academic_Version.pdf` | **md5-identical to the published-research copy (9a02c0bd...) and sits in a PUBLIC repo.** |
| `lyra-s-research-/published-data/Oracle_Loop_-_Integrity_Version.pdf` | Same, public. |
| `oracle-loop-paper/main.pdf`, `paper/main.pdf`, `academic/main.pdf` (Sep 3) | Current-source builds; render the unqualified H1 paragraph (AUROC 1.000 + "Crucially... opposite"). |

**Recommendation, not softened:** the two `_Version.pdf` files are distribution artifacts
asserting a withdrawn result as *established*, and their source revision no longer exists —
no edit can fix them; only replacement can. They should be **withdrawn or replaced now**,
not held for the rewrite: if patch approval takes time, pull them (or add an erratum notice
beside them in both locations, including the public repo) in the interim. After the patch is
approved: rebuild `main.tex` (Integrity), `academic_main.tex` or `academic/main.tex`
(Academic), and `paper/main.tex`, then replace all shipped copies in both locations.
Current sources are strictly newer than the Jul 17 builds (they already contain the 0.903
full-cache result and the corrected conclusion), so replacement loses nothing.

---

## Discipline notes

- No file above was edited. No git operations. No remote connections.
- "Withdrawn," never "refuted," throughout: the same-prompt control shows the measurement
  cannot support the claim; the waystations exploratory result keeps the question open in
  the expansion direction.
- Verification scripts (exact-string, backslash-safe):
  `C:\Users\Thomas\AppData\Local\Temp\claude\C--Users-Thomas\fedc44e2-d3b2-49fe-9e89-5890ff7e37ba\scratchpad\verify_anchors.py` and `verify_anchor4.py`.
  Re-run both before applying; every count must print 1.
- Related open item, out of scope here: `community/STYLE_GUIDE.md:49` still *instructs*
  authors to assert the withdrawn 1.000 as load-bearing (PATCHES B12 — the propagation
  mechanism). It should not outlive this patch.
