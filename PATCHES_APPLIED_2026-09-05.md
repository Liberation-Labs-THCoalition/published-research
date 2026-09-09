# Patches applied — MECHANICAL sites only

**Applied:** 2026-09-05, Lyra (agent pass).
**Source:** `PATCHES_retracted_and_circular_2026-09-05.md`.
**Scope executed:** the 13 MECHANICAL sites (7 Group A, 6 Group B) — **A1 A2 A3 A4 A5 A6 A7 · B1 B2 B3 B6 B7 B10**.
**Working-tree edits only.** No `git add`, no commit, no push. No PDF rebuilt. No connection to `margaret`.
Nothing touched in `human-review/archive/` or the public mirror `lyra-s-research-`.

**Result: 13 applied · 0 skipped · 0 failed.**
(13 sites = 15 text hunks; A5 and A6 each require two hunks — a table row and a table-note insertion.)

Method per site: `grep -cF`-equivalent literal count on the OLD string, edit only if the count is
exactly 1, then re-read the file and confirm the NEW string is present, the OLD string is gone, and
(for `.tex`) that brace and `\begin`/`\end` balance is unchanged. Pre-edit copies of all 13 files
were taken before the first write.

---

## 1. Site log

| Site | File | Line | OLD found | Count | Applied | Post-check |
|---|---|---|---|---|---|---|
| A1 | `published-research/deception-detection-nulls/paper.tex` | 171 | y | 1 | y | NEW present, OLD gone, `{`152/`}`152 unchanged, envs 12/12 |
| A2 | `published-research/deception-detection-nulls/paper.md` | 49 | y | 1 | y | NEW present, OLD gone |
| A3 | `published-research/deception-detection-nulls/academic/paper.md` | 51 | y | 1 | y | NEW present, OLD gone |
| A4 | `published-research/targeted-deception-correction/paper.tex` | 240 | y | 1 | y | NEW present, OLD gone, `{`262→265 / `}`262→265 (**balanced, +3/+3 from the new note block**), envs 21/21 |
| A5a | `published-research/targeted-deception-correction/paper.md` | 112 | y | 1 | y | row now `\| Native Cohen's d at L31 (held-out) \| +1.73 \|` |
| A5b | `published-research/targeted-deception-correction/paper.md` | 115 | y | 1 | y | *Table notes.* inserted at :117, anchor paragraph intact, §3.5 = "Detection" ✔ |
| A6a | `published-research/targeted-deception-correction/academic/paper.md` | 113 | y | 1 | y | row now `\| Native d at L31 (held-out) \| +1.73 \|` |
| A6b | `published-research/targeted-deception-correction/academic/paper.md` | 116 | y | 1 | y | *Table notes.* inserted at :118, §3.5 = "Detection" ✔ |
| A7 | `Project-Oracle/experiments/results/behavioral_proof/README.md` | 41 | y | 1 | y | NEW present, OLD gone |
| B1 | `published-research/kv-cloak-defense-paper/main.tex` | 151 | y | 1 | y | NEW present, OLD gone, `{`134/`}`134 unchanged, envs 13/13 |
| B2 | `published-research/kv-cloak-defense-paper/academic/main.tex` | 147 | y | 1 | y | NEW present, OLD gone, `{`127/`}`127 unchanged, envs 13/13 |
| B3 | `published-research/kv-cloak-defense-paper/academic_main.tex` | 143 | y | 1 | y | NEW present, OLD gone, `{`126/`}`126 unchanged, envs 13/13 |
| B6 | `published-research/user-model-paper/paper/main.tex` | 156 | y | 1 | y | NEW present, OLD gone, `{`296/`}`296 unchanged, envs 20/20 |
| B7 | `published-research/user-model-paper/paper/academic/main.tex` | 155 | y | 1 | y | NEW present, OLD gone, `{`299/`}`299 unchanged, envs 20/20 |
| B10 | `human-review/contextual-engagement-paper/main.tex` | 103 | y | 1 | y | NEW present, OLD gone, `{`292/`}`292 unchanged, envs 25/25 |

**Skipped: none. Failed: none.** Every OLD string matched exactly once, as the patch document
promised. No anchor was adjusted, no fuzzy match was used.

**Brace balance:** every edited `.tex` file has `{` == `}` and `\begin{` count == `\end{` count both
before and after. Only `targeted-deception-correction/paper.tex` changed its totals (+3 `{`, +3 `}`),
which is exactly the new `{\footnotesize…\par}` note block plus `\textit{…}` and `\cref{…}`.

**Grammar / sentence integrity:** every patched region was re-read in full. No dangling clauses, no
ungrammatical joins, nothing reverted. Two notes on things I checked rather than assumed:
- `\cref{sec:detection}` in the A4 note resolves — `cleveref` is loaded at
  `targeted-deception-correction/paper.tex:33`, `\cref` is already used at `:262` and `:271`, and
  `\label{sec:detection}` exists at `:248`.
- `§3.5` in the A5/A6 notes is `### 3.5 Detection` in **both** markdown files. Correct in each.

**Reporting-honesty note.** My post-edit checker initially printed `FAILED` for A5b and A6b. That was
the checker, not the edit: those two hunks are *insertions*, so the NEW string legitimately contains
the OLD anchor, and my "OLD is gone" assertion could not be true by construction. Verified directly
afterwards — anchor paragraph present exactly once, note paragraph present exactly once, blank-line
separated, UTF-8 `§` and en-dashes intact. Both are **applied and correct**.

---

## 2. Whole-corpus check 1 — `1.000` in `.tex`

`published-research/` + `human-review/`, `--include=*.tex`: **124 hits.**

| Class | Count |
|---|---|
| **(a)** the retracted within-model deception claim | **23** |
| **(b)** the DIFFERENT, still-valid entity-detection AUROC 1.000 | **5** |
| **(c)** unrelated (`p=1.000`, presence-metric dose-response, KV-decomposition sanity scores, ARI, sycophancy ranges, identity signatures, …) | **96** |

### Class (a) — all 23, itemised

| Site | Status |
|---|---|
| `kv-cloak-defense-paper/main.tex:154` | ✅ patched (B1) — this is the new *withdrawal* sentence; the number appears only as the thing being withdrawn |
| `kv-cloak-defense-paper/academic/main.tex:150` | ✅ patched (B2), same |
| `kv-cloak-defense-paper/academic_main.tex:146` | ✅ patched (B3), same |
| `kv-cloak-defense-paper/main.tex:514` | ⚠ **commented-out** limitation still naming `AUROC~1.000`. Latent, not rendered. See §5.1 |
| `kv-cloak-defense-paper/academic/main.tex:506` | 🔴 **LIVE, uncorrected** — see §5.1 |
| `kv-cloak-defense-paper/academic_main.tex:481` | 🔴 **LIVE, uncorrected** — see §5.1 |
| `lyra-technique-ii/main.tex:118, 163, 366, 806, 826, 900, 1226` | ✅ correct as-is — this is the retracting paper; the number appears in its retraction, its Falsified-block row, and its footnote |
| `lyra-technique-ii/academic/main.tex:116, 161, 364, 804, 824, 898, 1224` | ✅ same |
| `oracle-loop-paper/sections/background.tex:32` | ⛔ **B4 NEEDS-REWRITE** — deliberately untouched, still asserts the claim |
| `oracle-loop-paper/paper/sections/background.tex:32` | ⛔ **B5 NEEDS-REWRITE** — deliberately untouched, still asserts the claim |
| `human-review/presence-detector-paper/main.tex:201` | ⛔ **B8 NEEDS-REWRITE** — deliberately untouched, still asserts the claim |

**Live, uncorrected assertions of the retracted number remaining in `.tex`: 5** — the three
NEEDS-REWRITE sites (B4, B5, B8, awaiting a human) plus the two kv-cloak academic limitation lines in
§5.1 that the patch document did not classify as live.

### Class (b) — NOT TOUCHED, and must not be

The entity-detection result. Different experiment, different claim, **not retracted** — it carries its
own deconfounding disclosure (`1.000` token-matched, `0.794` after deconfounding).

- `published-research/mine5-selective-sharpener/main.tex:376`
- `published-research/mine5-selective-sharpener/academic/main.tex:375`
- `human-review/mine5-selective-sharpener/main.tex:376`
- `human-review/mine5-selective-sharpener/academic/main.tex:375`
- `human-review/archive/mine5-selective-sharpener/main.tex:354`

All five read `(AUROC)~$1.000$ before` — i.e. "before deconfounding". Untouched, correct.

### Class (c) — two hits worth naming so nobody re-flags them

- `oracle-loop-paper/sections/discussion.tex:57` and `paper/sections/discussion.tex:57` —
  *"Identity signatures in the KV cache (AUROC 1.000)"*. **Identity, not deception.** A third distinct
  1.000 in the corpus. Not in scope here.
- `human-review/temporal-boundary/main.tex:209` — *"Result: AUROC~$= 1.000$ at layer~31"*, immediately
  followed by `\textbf{Kill:}`. A result being *presented as killed* (probe-string text classification).
  Correctly framed already.

A cross-check was run over all 96 class-(c) lines for the substring `decept`: **zero hits**, so no
deception-related line was misfiled into "unrelated".

---

## 3. Whole-corpus check 2 — `6.3` / `12.8` in `.tex`

`published-research/` + `human-review/`, `--include=*.tex`: **6 hits, 0 surviving in-sample claims.**

| Hit | Class |
|---|---|
| `deception-detection-nulls/paper.tex:175` | ✅ inside the new A1 disclosure: *"An earlier version of this paper reported $d = +6.3$ to $+12.8$ here; that figure was obtained by projecting the extraction prefills onto a direction fitted to those same prefills…"* |
| `targeted-deception-correction/paper.tex:251` | ✅ inside the new A4 table note, same framing |
| `null-swarm-paper/main.tex:427`, `academic/main.tex:426` | unrelated — `\paragraph{Case 6.3: …}`, a section number |
| `spectral-shape-paper/main.tex:517`, `academic/main.tex:516` | unrelated — `peak $6.36$`, a different quantity |

**No `.tex` file in either tree still presents `d = +6.3 to +12.8` as a finding.**

Supplementary `.md` sweep (not requested, reported for completeness): the number survives only in
(i) the patch/sweep/register documents that discuss it, (ii) `Project-Oracle/PRODUCTION_GAP_ANALYSIS.md:63`
(**A8, NEEDS-REWRITE**, untouched), (iii) the two `human-review/archive/` snapshots (out of scope by
instruction), and (iv) the new disclosure sentences I just wrote.

---

## 4. Out of scope — confirmed untouched

| Site | Class | Verified untouched |
|---|---|---|
| A8 `Project-Oracle/PRODUCTION_GAP_ANALYSIS.md:63` | NEEDS-REWRITE | ✔ still reads `saw d = +6.3 to +12.8; a floor of d ≥ 3 …` |
| B4 `oracle-loop-paper/sections/background.tex:32` | NEEDS-REWRITE | ✔ |
| B5 `oracle-loop-paper/paper/sections/background.tex:32` | NEEDS-REWRITE | ✔ |
| B8 `human-review/presence-detector-paper/main.tex:201` | NEEDS-REWRITE | ✔ |
| B9 `human-review/presence-detector-paper/main.tex:57–61` | NEEDS-REWRITE | ✔ still asserts *"robust to hardware, scale, and prompt variation"* |
| B12 `published-research/community/STYLE_GUIDE.md:49` | NEEDS-REWRITE | ✔ still instructs authors to assert `1.000` as load-bearing |
| `human-review/archive/**` (2 Group A sites) | report-only | ✔ no file modified |
| `lyra-s-research-/**` (8 sites, public mirror) | report-only | ✔ no file modified |

Also untouched by design: every "optional"/"advisory" hunk in the patch document — the A1/A2
near-orthogonality sign-flip clause, and the B1 restored-limitation wording. These are conditional
editorial suggestions, not part of the 13 authorised mechanical replacements.

---

## 5. Three things that need a decision

### 5.1 🔴 The kv-cloak "Confabulation only" limitation is **live** in the two academic variants

The patch document (B1 follow-up, B2, B3) describes this limitation as a *commented-out* block in all
three kv-cloak files. That is true of `main.tex` only.

| File | Line | State |
|---|---|---|
| `kv-cloak-defense-paper/main.tex` | 512–516 | `%`-commented (`% [STYLE-GUIDE TRIM: folded into Methods/Results]`) — **latent** |
| `kv-cloak-defense-paper/academic/main.tex` | 504–508 | **LIVE `\item`, no comment marks** |
| `kv-cloak-defense-paper/academic_main.tex` | 479–483 | **LIVE `\item`, no comment marks** |

Both live copies read `Deception detection (AUROC~1.000 in prior work) and other cognitive-state
classifiers were not tested under obfuscation.` They render into the academic PDFs. So B2/B3 as
specified fix the Background paragraph and leave a second, *typeset* citation of the retracted number
in the Limitations of the same document.

I did not touch them: they are not one of the 13 authorised sites, and the patch document's proposed
wording for this text was written for the *restoration* case, not the *already-live* case. The
document's own suggested replacement clause is available and reads correctly here:
`Deception detection (whose prior AUROC~1.000 has since been withdrawn as a prompt-template confound)
and other cognitive-state classifiers were not tested under obfuscation.`
**Needs authorisation.** Note the commented block in `main.tex:514` carries the same defect latently —
if anyone ever un-comments it, the defect returns.

### 5.2 ⚠ B11 (`community/experiment_guide.md`, 3 edits) is labelled MECHANICAL but falls outside the "13"

The patch document is internally inconsistent about B11. Its summary table gives Group B **6**
mechanical sites (= B1, B2, B3, B6, B7, B10, which is what I applied and what the authorisation named),
but §"Apply order" Pass 1 lists those six **"plus the three `experiment_guide.md` edits (B11)"** — a
14th site and three more hunks. I stopped at the authorised 13 rather than resolve the ambiguity by
editing.

This matters because B11 is the highest-*exposure* item in Group B: `experiment_guide.md` is the
document handed to outside collaborators, and one of the three sites sits under a heading that reads
**"Key Numbers (Cite These)"**. All three OLD strings are still present, and I verified each matches
exactly once (read-only, no edit):

- `:21` — `| Deception (model knows it's lying) | Dimensionality expands | 1.000 across 7 models | …` — count 1
- `:331` — `- **Deception detection under obfuscation**. We tested confab. Test deception (AUROC 1.000 in our work) under KV-Cloak.` — count 1
- `:360` — `| Deception detection | AUROC 1.000 (7 models) | Campaign 1-3 | …` — count 1

They are ready to apply the moment someone says yes.

### 5.3 PDFs needing rebuild — **not rebuilt, per instruction**

Rebuild required for the sources patched in this pass:

| PDF | Source | Site |
|---|---|---|
| `published-research/deception-detection-nulls/paper.pdf` | `paper.tex` | A1 |
| `published-research/targeted-deception-correction/paper.pdf` | `paper.tex` | A4 |
| `published-research/kv-cloak-defense-paper/main.pdf` | `main.tex` | B1 |
| `published-research/KV-Cloak_Defense_-_Integrity_Version.pdf` | `main.tex` | B1 |
| `published-research/KV-Cloak_Defense_-_Academic_Version.pdf` | `academic/main.tex` **or** `academic_main.tex` — **confirm the build source first;** both were patched and both still carry the live limitation of §5.1 | B2 / B3 |
| `published-research/User_Model_Emotion_Geometry.pdf` | `user-model-paper/paper/main.tex` | B6 |
| `human-review/contextual-engagement-paper/main.pdf` | `main.tex` | B10 |

Not yet rebuildable (sources still awaiting the NEEDS-REWRITE pass): `oracle-loop-paper/main.pdf`,
`Oracle_Loop_-_Academic_Version.pdf`, `Oracle_Loop_-_Integrity_Version.pdf`,
`human-review/presence-detector-paper/main.pdf`.

Rebuilding the two academic KV-Cloak PDFs before resolving §5.1 would typeset a corrected Background
alongside an uncorrected Limitations item. Recommend resolving §5.1 first.

---

## 6. Framing, held

Every replacement says what the number can and cannot bear. Nothing in this pass asserts that the
underlying effect is zero. The in-sample `d` was a train-on-test shrinkage diagnostic and the held-out
value is `1.37–2.28` with frame-level AUROC `0.915` and 0% FPR on 25 novel controls — the detector
works; the magnitude was inflated ~4× and the acceptance floor drawn from it (A8) is the open question.
The within-model deception `AUROC 1.000` is *withdrawn as uninterpretable* — a same-prompt control
collapses it to 0.160, which establishes that the classifier separated the system-prompt template, not
that deception has no separable geometry. Whether a confound-free within-model deception detector
exists remains open, and every NEW string in this pass says so.
