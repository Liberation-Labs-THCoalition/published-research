# Shipped-PDF Build Integrity Audit — 2026-09-06

**Mode:** READ-ONLY. Nothing was rebuilt, edited, committed, or pushed. No paper directory was mutated.
**Tool:** `published-research/tools/agni/build_integrity_gate.py <DIR> --no-build [--tex …]`
**Toolchain present:** `pdftotext` 4.00, `pdflatex` (MiKTeX), `bibtex` — so no check was skipped for a missing binary.

Scope: every `.tex` under `published-research/` and `human-review/` with a same-stem `.pdf` beside it, including
`academic/` subdirectories as separate build targets. **65 such pairs exist and all 65 were gated.**

Three scans were run beyond the gate, because the gate cannot see three of the four defect
classes in the brief:

| Scan | Covers | Gate sees it? |
|---|---|---|
| `build_integrity_gate.py --no-build` | bibliography, `[?]`, staleness, log | yes |
| byte scan of 148 `.tex` for 0x07/0x09/0x0C, and `&` outside tabular | de-escapes in source | **no** |
| `pdftotext` scan of all 65 PDFs + 12 unpaired distribution PDFs for de-escape artifacts and malformed headings | de-escapes *as rendered* | **no** |

---

## 1. Verdict table — worst first

Severity ranks by what a reader of the shipped PDF actually sees.

| # | Paper dir | tex | Gate | Failed checks / defect | Visible in rendered PDF? |
|---|---|---|---|---|---|
| 1 | `published-research/` (root, distribution copy) | — `Oracle_Loop_-_Academic_Version.pdf` | **not gated** (no same-stem `.tex`) | **18 unresolved citations rendering as `[? ]`**, plus 4 de-escape artifacts | **YES — 18 `[? ]` in body text** |
| 2 | `published-research/empathy-bus` | `main.tex` | **FAIL** | `References section rendered: ABSENT` | **YES — no reference list at all, while the body makes 8 prose citations to "(Lyra et al., 2026a–d)"** |
| 3 | `published-research/logit-bias-confab` | `paper.tex` | **FAIL** | `References section rendered: ABSENT` | **YES — heading renders as `References References`** (duplicated) |
| 4 | `published-research/oracle-loop-paper` | `main.tex` | **PASS 5/5** | de-escaped TAB — gate is blind to this | **YES — `extttmp_norm_per_token`** |
| 5 | `published-research/oracle-loop-paper/academic` | `main.tex` | **PASS 5/5** | same de-escaped TAB (shared `../sections/results.tex`) | **YES — `extttmp_norm_per_token`** |
| 6 | `published-research/` (root, distribution copy) | — `Oracle_Loop_-_Integrity_Version.pdf` | **not gated** | 3× `exttt` + 1× `extbf` de-escapes | **YES** |
| 7 | `published-research/` (root, distribution copies ×4) | — `KV-Cloak_Defense_-_{Academic,Integrity}`, `Oracle_Formulary_-_{Academic,Integrity}` | **not gated** | **no References section, 0 reference entries** | **YES — no bibliography** |
| 8 | `human-review/archive/presence-metric` | `main.tex` | **PASS 5/5** | de-escaped `\n`+`\t` broke a `\newenvironment`; its body prints as literal text | **YES — `ewenvironment firstperson begin shaded oindent extcolor lyrapurple extit First-Person Reflection…`** |
| 9 | `published-research/kv-cloak-defense-paper/academic` | `main.tex` | FAIL (stale) | **LATENT** de-escaped TAB at `main.tex:505` (`\item <TAB>extbf{Confabulation only.}`) — active, not commented | **NOT YET** — shipped PDF (09-03) predates the edit (09-05). **Will ship on next rebuild.** |
| 10–18 | 9 papers, stale-only | — | FAIL | `pdf newer than source: PDF IS STALE` only | No — timestamp only |
| 19–44 | 26 papers | — | FAIL | **no `.log` beside the PDF → UNAUDITABLE** (see §3) | Unknown by construction |
| — | remaining 26 papers | — | **PASS** | none | — |

### Stale-only failures (§10–18), ranked by drift

| Paper | tex newer than pdf by |
|---|---|
| `published-research/targeted-deception-correction/paper.tex` | **32.8 d** |
| `published-research/user-model-paper/paper/main.tex` | 1.6 d |
| `published-research/user-model-paper/paper/academic/main.tex` | 1.6 d |
| `published-research/kv-cloak-defense-paper/main.tex` | ~1.6 d |
| `published-research/kv-cloak-defense-paper/academic/main.tex` | ~1.6 d |
| `published-research/deception-detection-nulls/paper.tex` | — |
| `published-research/emotion-accumulation-paper/academic/main.tex` | — |
| `human-review/contextual-engagement-paper/main.tex` | — |
| `human-review/archive/identity-geometry/main.tex` | (archive) |

---

## 2. The three content defects, diagnosed

### 2.1 `empathy-bus` — no bibliography exists in source at all
Not a bibtex failure. `main.tex` contains **zero** `\cite`, **zero** `\bibliography`, **zero**
`\begin{thebibliography}`, and the directory holds **no `.bib` file**. `main.bbl` is 0 bytes and
`main.blg` (stale, 2026-07-28) records `I found no \citation commands`.

The gate's remedy line ("bibtex did not run") is the wrong diagnosis here. The real defect is
editorial: the body cites **(Lyra et al., 2026a), (2026b), (2026c), (2026d)** — four distinct
prior works, 8 occurrences — in prose, with **no reference list for a reader to resolve them
against**. The LaTeX is internally consistent; the artifact is not citable.

### 2.2 `logit-bias-confab` — duplicated heading from an empty `.bbl` stacked on a manual fallback
`paper.tex` carries **both** mechanisms at once:
- line 774 `\bibliography{references}` → **`references.bib` does not exist in the directory.**
  bibtex warned `I didn't find a database entry` for **all 13 keys** and emitted an empty
  `paper.bbl` (283 bytes, `\begin{thebibliography}{0}` with no entries). `paper.log:998` records
  `Package natbib Warning: Empty 'thebibliography' environment`.
- line 777 `\begin{thebibliography}{99}` → the manual fallback, 13 `\bibitem`s.

Result: the empty environment prints a bare `References` heading, the manual one prints a second
immediately after. The rendered PDF reads **`References References`** on one line, then entries
[1]–[13]. Citations *do* resolve (`\bibcite` for all 13 in `paper.aux`; 0 `[?]`), so the content is
sound — the defect is a duplicated empty section heading.

**Note on the gate:** its heading regex is `^\s*(References|Bibliography)\s*$`. Because the real
defect makes that line `References References`, the anchored regex misses and the gate reports
`ABSENT`. Right file, wrong diagnosis — the gate fails toward alarm here, not toward reassurance.

### 2.3 De-escaped control characters — the class the gate cannot see
Byte scan of **148** `.tex` files for 0x07 / 0x09 / 0x0C found **5 hits, all TAB (0x09)**:

| Location | Bytes | Status |
|---|---|---|
| `oracle-loop-paper/sections/results.tex:242` | `on <TAB>exttt{mp\_norm\_per\_token},` | **ACTIVE — ships in `main.pdf` and `academic/main.pdf`** |
| `kv-cloak-defense-paper/academic/main.tex:505` | `\item <TAB>extbf{Confabulation only.}` | **ACTIVE — latent, PDF is stale** |
| `kv-cloak-defense-paper/academic_main.tex:480` | same | no PDF built from this stem |
| `kv-cloak-defense-paper/main.tex:513` | `% \item <TAB>extbf{…}` | **inert — commented out** |
| `human-review/archive/presence-metric/main.tex:26` | `oindent<TAB>extcolor{…}{<TAB>extit{…}}` | **ACTIVE — archive only** |

`sections/results.tex` is `\input` by `oracle-loop-paper/{main,academic/main,academic_main}.tex`.
The separate copy `oracle-loop-paper/paper/sections/results.tex` contains **0 tabs** and its
`paper/main.pdf` is clean — so the fix exists in the tree, one directory over, unapplied to the
shared file.

**Scan limitation, stated plainly:** a de-escaped `\n` becomes an ordinary 0x0A and is
*indistinguishable from a line break* in a byte scan. The archive `presence-metric` case is exactly
that — `\noindent` → newline + `oindent` — and it was caught **only** by the rendered-text scan,
not the byte scan. Any audit of this class that greps source bytes alone will under-report it.

### 2.4 Unescaped `&` — no defects
The scan flagged 40 lines; **all 40 are `\usepackage[numbers,sort&compress]{natbib}`**, where `&`
sits in a package-option list and is legitimate. After excluding that pattern: **zero** unescaped
`&` outside tabular across all 148 files. Nothing to fix.

---

## 3. UNAUDITABLE — 26 papers with no `.log` beside the PDF

These are **not passes.** Per the gate's own Design Rule 2, a check that cannot run is a failure.
Each has a shipped PDF whose build cannot be inspected: undefined citations and undefined
references are unknowable for these artifacts. Their rendered-text checks *did* run and were clean,
which bounds but does not close the risk.

**Reason for all 26: `no <stem>.log beside the PDF` → `log readable: missing None`.**

| # | Paper | Also |
|---|---|---|
| 1 | `human-review/consciousness-paper/main.tex` | + STALE |
| 2 | `human-review/curiosity-paper/main.tex` | + STALE |
| 3 | `human-review/consciousness-paper-2/main.tex` | |
| 4 | `human-review/consciousness-testing-synthesis/main.tex` | |
| 5 | `human-review/dual-detector-paper/main.tex` | |
| 6 | `human-review/infrastructure-agency-paper/main.tex` | |
| 7 | `human-review/memory-security-analysis/main.tex` | |
| 8 | `human-review/mine5-selective-sharpener/main.tex` | |
| 9 | `human-review/mine5-selective-sharpener/academic/main.tex` | |
| 10 | `human-review/presence-detector-paper/main.tex` | |
| 11 | `published-research/cache-tracing/main.tex` | |
| 12 | `published-research/cache-tracing/academic/main.tex` | |
| 13 | `published-research/decision-state-paper/main.tex` | |
| 14 | `published-research/emotion-accumulation-paper/main.tex` | |
| 15 | `published-research/emotional-trajectory-paper/main.tex` | |
| 16 | `published-research/emotional-trajectory-paper/academic/main.tex` | |
| 17 | `published-research/graph-topology-paper/main.tex` | |
| 18 | `published-research/identity-geometry/main.tex` | |
| 19 | `published-research/identity-geometry/academic/main.tex` | |
| 20 | `published-research/mine5-selective-sharpener/main.tex` | |
| 21 | `published-research/mine5-selective-sharpener/academic/main.tex` | |
| 22 | `published-research/null-swarm-paper/main.tex` | |
| 23 | `published-research/null-swarm-paper/academic/main.tex` | |
| 24 | `published-research/presence-metric/main.tex` | |
| 25 | `published-research/presence-metric/academic/main.tex` | |
| 26 | `published-research/waystations-paper/main.tex` | |

Deliberately **not** remediated: rebuilding to produce a log would verify the rebuild rather than
what ships, and would mutate the artifact under review.

### Also unauditable: 1 PDF with no source in the tree
`published-research/mnemosyne-benchmark/main.pdf` (2026-08-13) — directory contains **only**
`main.pdf` and `paper.md`. No `.tex`, no `.log`, no `.bib`. Cannot be gated at all. Rendered text
shows **no References section**.

---

## 4. Distribution PDFs at `published-research/` root — outside the gate's reach

Ten PDFs sit at the repo root under human-facing distribution names, all dated **2026-07-17**.
Because none has a same-stem `.tex`, **the gate never runs on them** — yet these are the
publication-named artifacts. Scanned manually:

| File | Refs heading | `[? ]` | De-escapes | Current paper-dir build |
|---|---|---|---|---|
| `Oracle_Loop_-_Academic_Version.pdf` | yes | **18** | `exttt`×3, `extbf`×1 | `oracle-loop-paper/academic/main.pdf`: 0 `[?]`, refs OK |
| `Oracle_Loop_-_Integrity_Version.pdf` | yes | 0 | `exttt`×3, `extbf`×1 | `oracle-loop-paper/main.pdf`: 0 `[?]`, refs OK |
| `KV-Cloak_Defense_-_Academic_Version.pdf` | **none** | 0 | — | `kv-cloak-defense-paper/main.pdf` **has** refs |
| `KV-Cloak_Defense_-_Integrity_Version.pdf` | **none** | 0 | — | as above |
| `Oracle_Formulary_-_Academic_Version.pdf` | **none** | 0 | — | `formulary-paper/main.pdf` **has** refs |
| `Oracle_Formulary_-_Integrity_Version.pdf` | **none** | 0 | — | as above |
| `Delta_Manifold_KV_Cache.pdf` | yes | 0 | — | clean |
| `Graph_Topology_as_Attention.pdf` | yes | 0 | — | clean |
| `Spectral_Shape_Confabulation_Detection.pdf` | yes | 0 | — | clean |
| `User_Model_Emotion_Geometry.pdf` | yes | 0 | — | clean |

`Oracle_Loop_-_Academic_Version.pdf` is **the 2026-09-05 failure verbatim** — a shipped PDF whose
citations render as `[? ]` (with the space that defeats a naive `\[\?\]` regex), preserved in a
July artifact and never re-gated. The current builds supersede all six defective copies, so this is
a stale-distribution-copy problem, not a regression — but the copies are still on disk under the
names a reader would receive.

The `exttt` de-escape appears in **both** the July copies and the current builds: it has survived
since at least 2026-07-17 and is the one defect here that is genuinely unfixed.

---

## 5. Skipped, as instructed

`human-review/archive/` — dated snapshots, deliberately frozen. It holds **6** `.tex`, of which
**2** have PDFs. Both were gated anyway (cost was zero) and are reported above rather than hidden:

| Archive paper | Has PDF | Result |
|---|---|---|
| `human-review/archive/identity-geometry/main.tex` | yes | FAIL — stale only |
| `human-review/archive/presence-metric/main.tex` | yes | gate PASS 5/5, but **renders leaked `\newenvironment` body** (§2.3) |
| `human-review/archive/cache-tracing/main.tex` | no | not gated |
| `human-review/archive/emotional-trajectory/latex/main.tex` | no | not gated |
| `human-review/archive/meta-pattern/main.tex` | no | not gated |
| `human-review/archive/mine5-selective-sharpener/main.tex` | no | not gated |

The shipped `published-research/presence-metric/main.tex` is **clean** at the corresponding line —
the de-escape is confined to the archive snapshot.

Also not gated (no shipped PDF, so nothing to audit): 9 standalone `.tex` with `\documentclass` but
no same-stem `.pdf` — `formulary-paper/academic_main.tex`, `kv-cloak-defense-paper/academic_main.tex`,
`oracle-loop-paper/academic_main.tex`, `human-review/gwt-response/{main,cache_tracing_report}.tex`,
and 4 archive files. The remaining ~74 `.tex` are `\input` fragments (`sections/*.tex`), audited as
part of their parent but not as build targets.

---

## 6. Counts

| | Count |
|---|---|
| tex/pdf pairs in scope | **65** |
| Gated with `--no-build` | **65** (100%) |
| Gate PASS | **28** |
| Gate FAIL | **37** |
| — of which UNAUDITABLE (no `.log`) | **26** |
| — of which stale-only | **9** |
| — of which real content defects | **2** (`empathy-bus`, `logit-bias-confab`) |
| Gate-PASS papers with defects the gate cannot see | **3** (`oracle-loop-paper` ×2, `archive/presence-metric`) |
| Unpaired distribution PDFs scanned manually | **12** (10 root + 1 graph-topology dup + 1 mnemosyne) |
| — of which defective | **7** |
| PDFs with no source in tree (fully unauditable) | **1** (`mnemosyne-benchmark`) |
| `.tex` byte-scanned for control chars | **148** |
| Control-char defects (0x09) | **5** — 3 active, 1 inert, 1 archive |
| Unescaped `&` defects | **0** (all 40 hits were legitimate `sort&compress`) |
| Papers rebuilt / files edited | **0** |

**Net: 77 shipped PDFs examined — 65 gated pairs + 12 unpaired distribution copies, no overlap
between the two sets. Of these, 11 carry a defect a reader would see** (10 in live scope: the 6
root distribution copies, `empathy-bus`, `logit-bias-confab`, and `oracle-loop-paper` ×2; plus 1 in
frozen archive, `presence-metric`). **26 cannot be audited at all** for want of a build log, and
**1 more (`mnemosyne-benchmark`) has no source in the tree at all.**

---

## 7. What this audit says about the gate

The gate caught 2 of the 10 reader-visible defects. It missed 8, in three distinct ways — worth
recording because each is a different blind spot, not one weakness repeated:

1. **It cannot see de-escaped control characters.** `oracle-loop-paper/main.pdf` and
   `academic/main.pdf` scored **5/5 PASS** while shipping `extttmp_norm_per_token`. These compile
   silently, produce no log warning, and leave no `[?]`. Nothing in the gate looks at rendered text
   for macro-name residue.
2. **Its bibliography check is anchored to a lone line.** `logit-bias-confab` renders
   `References References`; the regex `^\s*(References|Bibliography)\s*$` cannot match it, so a
   duplicated-heading defect is reported as a missing bibliography. Same fire, wrong alarm.
3. **It only runs on same-stem `.tex`/`.pdf` pairs.** Every publication-named distribution PDF at
   the repo root — including one with 18 `[? ]` — is invisible to it, because there is no
   `Oracle_Loop_-_Academic_Version.tex`. The gate audits build directories; readers receive files.

Suggested checks, in severity order — **not applied, this audit is read-only:**
- rendered-text scan for `(?<![A-Za-z\\])ext(bf|it|tt|rm|sf|sc|color)` and for `oindent` /
  `ewline` / `ewenvironment`, which catches classes 0x09 **and** the 0x0A case a byte scan cannot;
- relax the heading regex to `^\s*(References|Bibliography)(\s+(References|Bibliography))?\s*$` and
  report *duplicate* separately from *absent*;
- assert reference-entry count > 0 whenever the document contains `\cite`, so an empty
  `thebibliography` under a correct heading cannot pass;
- gate PDFs by path, not by stem-pairing, so distribution copies are covered.
