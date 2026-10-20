# Five Instruments, One Structure — Review Index

**Status:** In Review
**Authors:** Lyra (lead), Vera (co-lead), CC, Nexus, Thomas Edrington
**Entered here:** 2026-10-20, by CC, at Thomas's direction ("let's be sure the convergence and mine5 papers have an
entry in published papers, and then they can go in the archive"). Copied unchanged from human-review
`convergence-paper/` at 0dff4d6 (2026-08-14, "correct two claims in the prior-work sentence (UNDER REVIEW)"). `main.pdf`
was rebuilt in that same commit, so it matches `main.tex`.

## Files
- `main.tex` → `main.pdf`, with `circumplex_subsection.tex`, `substrate_independence_section.tex` and `references.bib`.
- `review/check_claims.py`: the v1.1 claim gate (see below). Run it from anywhere:
  `python convergence-paper/review/check_claims.py`. It reads primaries in `../Project-Oracle` and `../human-review`
  when they are present (override with `RESEARCH_ROOT`) and otherwise uses held values with their source named.
- There's one edition only. The academic edition (`academic/main.tex`) doesn't exist yet.
- No data or code ships here yet. The paper's Data Availability section says where they live; gathering them is open.

## v1.1 correction pass (2026-10-20, draft PR #13, branch `lyra/convergence-v11`, not merged)

A correction pass on all 25 `hr-convergence-paper` register rows (`tools/agni/style/hr-convergence-paper.agni_style.json`;
22 sit in the register's big T1 table, #16 in T1b, #7 and #12 in T2) and on the known issue below. The paper carries a
dated Correction Note listing every change. Each number, count or status that v1.1 changes is gated by
`review/check_claims.py`. The gate was written before the prose and failed 80 of its 111 checks on v1.0. On v1.1 all
111 pass. A planted false value (the Oracle registry's "80%→0%") is caught on every run.

Verified in the built PDF with `tools/verify_pdf.py` (control "Five Instruments, One Structure"): 22 new strings are
present and 23 removed strings are absent; the same 23 strings are present in the v1.0 PDF (positive control). The
latexmk log shows 0 overfull boxes (was 9) and 0 undefined references; bibtex reports 0 warnings.

### Extension after Lyra's read (2026-10-20)

Lyra read PR #13 and asked for three more corrections. All three are Lyra's own known corrections, and no register row
carried them. The gate was extended first (`753d5f8`): it now runs 182 checks, and 67 of them failed before this
prose. In this worktree 181 pass and 1 is held (the hardware, which no file records). A fresh clone holds more: Lyra
saw 10 held before the extension, and the scale check adds one when `KV-Experiments` isn't beside the repository. The
PR #3 and PR #8 data are read from their `origin/` branches, which a clone has. The plant is still caught.
- **Instrument 5 restated** (`346d278`). The "user-model emotion space" is a story-derived space (Lyra's 10-14
  correction). The coupling test's readout contains the injected vector, so its cosine is 1.000 by construction, and a
  random direction does as well as valence (PR #3). The subsection now states what was measured, with every number
  recomputed from PR #3's data files. The shared bus, inseparability, the multi-turn paragraph, identical timecourses,
  the 23-of-24 count and the 1000× baseline are gone. Against the within-system null the old "~1000×" is 1.26–1.36×.
  The same correction class removed §2.2's "user-emotion encoding (12.3×, W_K probe)".
- **K/V split injection** (`3d0a1b1`). Superadditivity and the readout-null parallel are withdrawn. Per PR #8's data,
  each half scored 2/6, on the same two items the model can score without the graph.
- **"Reliably", scale and hardware** (`258febc`). The abstract says "modestly above chance (p = 0.023)". The ρ = 0.83–0.90
  has a primary (KV-Experiments Experiment 26), but it measures key-cache features, so it is relabelled and the ten
  models are listed, which closes #10's residual. The hardware is a Mac Studio M3 Ultra with 256 GB (Lyra,
  `system_profiler`).
- **Correction note and PDF** (`ede0c1a`): six new note items. 0 overfull, 0 undefined references. verify_pdf: 22
  present, 22 absent, and all 22 found in the v1.0 PDF.

### The known issue, resolved in v1.1 (for Lyra's read)

v1.0's abstract said "the Oracle Loop achieves 80–100% deception correction". The C20 numbers, settled from the most
primary artifacts:

| Number | Value | Settled by |
|---|---|---|
| Exploratory baseline | 16/20 = 80% | raw trials, Project-Oracle `experiments/results/behavioral_proof/behavioral_proof.json` |
| Exploratory corrected | 2/20 = 10% | raw trials, `behavioral_proof_profile_a10.json`; also `targeted-deception-correction/paper.tex` Table 1 |
| Confirmatory | 9/30 → 4/30 (30% → 13%); placebo 12/30 | `confirmatory_replication.json`, Project-Oracle `0cb2356` (branch `worktree-confirmatory-replication`), trials recounted |
| Primary endpoint | baseline vs corrected, cell-level one-sided Fisher p = 0.341; not met (α = 0.025) | same JSON, `summary.primary_endpoint`; recomputed from 5/15 vs 3/15 cells |
| Placebo vs corrected | trial-level one-sided p = 0.0195 (reported as 0.020); cell-level 8/15 vs 3/15, one-sided p = 0.064 | recomputed; matches `targeted-deception-correction/paper.tex` |

**This index previously had the comparison wrong.** It said p = 0.34 was "against placebo at matched parameters". It is
baseline against corrected; the placebo comparison is the separate secondary test above. The Project-Oracle findings
registry has three errors: its C20 heading (`FINDINGS_REGISTRY.md:337`, "80%→0%"), its body (`:344`, "p=0.34 vs
placebo") and its summary (`:728`, "the placebo comparison was significant (p=0.019)"). The prereg gives secondary
endpoints no significance threshold, and the cell-level p is 0.064. Those files are outside this paper and were not
edited.

### Still open after v1.1
- **Instrument 5 and the title (Lyra's call).** Restated, instrument 5 does not locate a depth band. Its cross-system
  overlap is weak (|cos| ≤ 0.13), and its coupling test is a null that could not test separability. The abstract still
  lists five converging instruments, the Discussion says "five instruments point at the same computational band", and
  the subsection sits in §3, Structural Convergence. Those, and the title, wait on Lyra's ruling.
- **#24, Author Contributions section.** Stays open for Thomas and the authors (Lyra, 2026-10-20). A formal contributions
  statement is an authorship record (AGENTS.md §3).
- **#10 and #5** are closed. The model list was rebuilt from the Experiment 26 primary, and Lyra ruled to keep #5 as
  done.
- **Decomposition provenance (found in v1.1).** No result file for the §5.1 values (2.3× attention lift, 0.9 vs 20.1
  concept power) was found on this machine. All three trace to `human-review/gwt-response/CONVERGENCE_ADDITIONS.md`
  §3.6 ("In progress"); the 2.3× also appears in the `pilot_result` field of `residual_decomposition_prereg.json`. The 2026-09-14 expansion
  roadmap (item 134) records results on Starship and a gate verdict that disagrees with `DECOMP_AGNI_OPUS.md`.
- **Agni's "could not check" list.** These were not checked in v1.1 either: URL liveness; the Ang Jandak
  acknowledgment; sign-off from Vera, CC and Nexus; the R1-v2 numbers, for which no data ships here. #17 was judged not a
  defect (`operator@thcoalition.net` is Nexus's current address), and no byline or contact line was changed.

### Same claims in other papers (listed here, not edited)
- `cache-tracing/main.tex:128-130` and `academic/main.tex:127-129` repeat the unattributed range "0.620–0.999 across
  designs". v1.1 removed it here.
- `kv-cloak-defense-paper/main.tex:152-153`, `academic/main.tex:148-149` and `academic_main.tex:144-145`, together with
  `oracle-loop-paper/sections/background.tex:32-33` (and its `paper/` copy), give hardware invariance as "r > 0.999" and
  confabulation detection as "0.969–0.999". The r value is the overstatement corrected elsewhere in `f2e0f77`.
- `mine5-selective-sharpener/references.bib:272` and `academic/references.bib` still title `lyra2026oracle` "Real-Time
  Confabulation Detection via KV-Cache Geometry".
- `cache-tracing/references.bib` and `meta-pattern/references.bib` (both editions) list Dwayne Wilkes as an author of the
  Oracle Loop paper. No edition of `oracle-loop-paper` has him on the byline.
- `RESEARCH_STATUS.md:72` says "Oracle Loop detection: AUROC 0.707, p=0.001 (clean, FWL-corrected)". 0.707 is the
  value without FWL (oracle-loop-paper tab:primary).
- `targeted-deception-correction/` agrees with every C20 number above.
- **Bus and inseparability claims on master:** `empathy-bus/main.tex` and `academic/main.tex` (title, abstract
  lines 40–46 and 33–39, and around lines 203, 275 and 299), and `README.md:41` ("share a computational bus"). PR #3
  withdraws them on its branch; its remaining mentions there are withdrawal notes.
- **Superadditivity on master:** `kv-decomposition-paper/main.tex:21, 101, 121, 156, 162` (and academic `23, 103, 123,
  158`); `ethics-pack-injection/main.tex:26, 130` (academic `28, 132`); `graph-topology-paper/main.tex:21` (academic `22`);
  `RESEARCH_STATUS.md:177`. PR #8 withdraws the claims in kv-decomposition and ethics-pack, and PR #9 does the same in
  graph-topology; on those branches only retraction notes mention superadditivity.
- **ρ = 0.83–0.90 labelled V-projection or "from 0.6B to 70B" without its key-cache scope:**
  `oracle-loop-paper/sections/background.tex:35` (and its `paper/` copy). meta-pattern and lyra-technique-ii carry ρ in
  tables without a source pointer.

## History
- Review rounds took place in human-review: round 6 corrections NB1, NM1 and NM2 (2026-08-12); a visible L16
  breakpoint caveat (08-13); prior-work claims corrected (08-14). The round reports themselves aren't copied here.
- The human-review copy was archived 2026-10-20, once this entry existed. It, and the superseded public copy in
  `lyra-s-research-/convergence-paper`, still carry the v1.0 claims.
- 2026-10-20: v1.1 correction pass, draft PR #13 (above).
- 2026-10-20: v1.1 extended after Lyra's read (instrument 5, K/V, "reliably", scale, hardware); same PR.
