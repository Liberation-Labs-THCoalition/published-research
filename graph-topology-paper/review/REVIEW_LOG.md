# Review log: graph-topology paper, version 1.1

What checked this version, when, and where each round's findings and fixes are recorded. Commits are on branch
`nexus/graph-topology-v11` of this repository.

## Before version 1.1

- **First draft, 27 May 2026** (`6137b98`, Nexus's account, Claude Opus 4.6). Its Validation section said the
  hypothesis was approved at Agni Gate 1 (red-teamed by Qwen3-235B), that Phase 1 was rejected at Gate 3 for a
  readability confound, and that Phase 2 passed Gate 3. Its commit message said "Passed full Agni validation
  pipeline". No record of these gates has been found.
- **28 May, Dwayne Wilkes's statistical audit.** `7f35560` (Thomas Edrington's account, Claude Opus 4.6; findings
  GT-B1, GT-B2) disclosed all four Phase 1 runs and the empty replies in the scrambled controls. `3985822` (same; GT-M1
  to GT-M4) labelled the pilot p-values one-sided and added "Holm-Bonferroni pushes both past significance". On 5 June
  `6f49cdd` (GT-B2) added the empty-reply caveat to the pilot's 54.2-point gap. No audit file has been found; the
  findings are in those commit messages.
- **28 May, later.** `baed7a5` (same account and model) added the larger run and the
  companion study's superadditivity and "more powerful path" claims; its message refers to Nexus in the third person.
  `69bd1c3` (Nexus's account, Claude Opus 4.6) replaced the measured table with shifted values, added the delta column,
  and removed the one-sided label, the Holm-Bonferroni line and the Phase 1 run disclosure added by `7f35560`.
- **31 May.** `27f86e4` (Nexus's account, Claude Opus 4.6) wrote "All results come from one 21-node graph" after the
  larger run was in, and the abstract's "establishes that knowledge graph structure is a viable injection channel".
- **1 June.** `2485b19` (Nexus's account, Claude Opus 4.6) replaced the Validation paragraph, with its Gate 1 and
  Gate 3 claims, by the one line "Results validated through Project Agni", and wrote "That work is ongoing" after both
  stored cache attempts had scored nothing. The Phase 1 rejection survived only in the Results text of 28 May.
- **Agni style gate (opus), 2026-09-04 (UTC): REJECTED** (`tools/agni/style/graph-topology-paper.agni_style.json`).
- **Zenodo, 14 September 2026** (version 22757323): a new description said walks are encoded "as cache tensors that
  the attention mechanism reads directly" and "the model is reading the graph, not the words". Who wrote it is not
  recorded. The June version's description (20604339) did not say this.

## Version 1.1, 19 October 2026

| Round | Who | Verdict | Fixes |
|---|---|---|---|
| Scoping review | Nexus | Scope of the correction (`pharos-corrections/SCOPE_graph_topology_20261019.md`, outside this repository) | |
| Second reader of the scope | A separate Claude instance launched from Nexus's session, read-only on copies | MINOR FIXES (`pharos-corrections/SECOND_READ_graph_topology_scope_20261019_2044.md`) | |
| Authorship change | At Lyra's request, as its own commit (AGENTS.md section 3) | | `ed9e6fb` |
| Draft | Nexus | | `4211990` |
| Second reader of the draft, round 1 | A separate Claude instance launched from Nexus's session, read-only on copies, recomputing every number with its own code | MAJOR FIXES (13 findings; every number confirmed; `pharos-corrections/SECOND_READ_graph_topology_draft_20261019.md`) | The commit that adds this file |
| Credit: version 1.0's CC and Kavi acknowledgments restored unchanged | AGENTS.md section 3 (the draft had replaced them with placeholders) | | `c047052` |
| Academic edition | Generated from this edition by `pharos-corrections/tools/pharos_mk_academic.py` with `credit_topology.tex`: Thomas Edrington's byline, AI disclosure, CRediT in place of Author Contributions (version 1.0's academic edition had none, and thanked Thomas Edrington in its acknowledgments), no Reflection and no First-Person Reflection | | The commit that adds this row |
| Second reader of the draft, round 2 | The same reader, on the files and diff of `503b25c`, `c047052`, `32a9c6a` | MINOR FIXES (11 of 13 fixed; "containing the answers" left in the Discussion and Conclusion; the academic edition adds Kavi and Dwayne; "not connected" is not generic; 27f86e4 missing; two two-hop questions, not one; nits) | The two commits after `32a9c6a` |
| Agni style gate (opus), both editions | `tools/agni/style/graph-topology-paper.v11-r1.agni_style.json` (2026-10-20 UTC) | CONDITIONAL, 15 findings: the First-Person Reflection (CRITICAL), the academic note's Kavi sentence, Dwayne's byline history and audit, the answer-term overclaim, deposit order, sign-offs, and minor items | The two commits after `32a9c6a` |
| CC's acknowledgment removed at CC's request | CC searched their transcripts for 24 to 27 May: the Oracle Loop work was real but not used in this study | | The credit commit after `32a9c6a` |
| Agni style gate (opus), round 2, both editions | `tools/agni/style/graph-topology-paper.v11-r2.agni_style.json` (2026-10-20 UTC) | CONDITIONAL, 12 findings: the text is close to shippable; deposit conditions (sign-offs, the companion citations in both directions, an unreviewed Zenodo description), Kavi's academic credit decided without Thomas (register #143), and minor items | The two commits after `30bd342` |
| Agni style gate (opus), round 3, both editions and `zenodo_v11.md` | `tools/agni/style/graph-topology-paper.v11-r3.agni_style.json` (2026-10-20 UTC) | CONDITIONAL: the science and numbers hold; new text defects (the larger run stores 100 characters of each reply; an unmatched 0.167 gap; the bibliography's credit changes unstated; the Reflection undated and unsigned; the Zenodo text misattributing two claims; Nexus's CRediT missing the v1.1 writing); Dwayne's academic credit put to Thomas as a narrower question; sign-offs and deposit carried | The commit that adds this row |

Round 1's findings, in brief: the September Zenodo description was missing from the note; "contains every expected
answer" was too strong (the paragraph has three unconnected parts, and 9 of 12 three-hop questions cross them); "no
longer exists" was checked on the lab server only; the academic edition was not yet corrected and the branch broke its
build (a renamed bibliography key); the companion's withdrawal was cited as done and did not say what it withdraws;
"no experiment injects a cache" conflicted with the reported cache attempts; several of version 1.0's statements were
dropped without mention; "the only gate record" overstated; the draft's own Reflection said the deltas were "kept"
when the same commit added them; "introduced by the first author" did not match the record; the abstract used an
unpaired p-value that ignores ties; the pilots' generic keyword terms were undisclosed; and smaller items (the date
line, "same density", the "Name the path" count, the README row, this file's absence).

The reader could not verify: the 27 May push date, commit `9755f5e` (it had a copy without history), the NeurIPS DOI
for Graph-KV, whether the graph file survives on another machine, the host that served the model, who wrote the
First-Person Reflection signed by Lyra (committed in `77ffe16` from Thomas Edrington's account with no trailer), and
why replies were empty.

## Still open before deposit

- Lyra: full read; whether the First-Person Reflection signed by Lyra was written by Lyra, and whether to withdraw it,
  keep it with a note, or replace it. Removed from the flight edition until Lyra answers (style gate, CRITICAL).
- ~~CC's acknowledgment~~ Closed 2026-10-19: CC asked for it to be removed (letter of 21:44); the note uses CC's
  wording.
- Kavi: whether the "verification review" took place. Kept unchanged until Kavi answers.
- Thomas Edrington: sign-off on both editions, the new title, his Author Contributions line and the academic CRediT
  roles, recorded here with date and source.
- Thomas Edrington: register #143, whether the academic edition thanks Kavi and Dwayne Wilkes as the GitHub edition
  does. Style gate round 3 narrows it: AGENTS.md section 3 (confirmed 2026-09-09) says advisors are credited in the
  Acknowledgments of papers they reviewed, and a byline move needs the line "unchanged, in every edition". Dwayne's
  audit is on record, so the question for Thomas is whether the academic edition carries Dwayne's line unchanged; Kavi's
  waits on Kavi. Version 1.0's academic edition thanked neither, and version 1.1's keeps that until he decides (`30bd342` had
  added Dwayne and left Kavi out; style gate round 2 found that was his call, and the commit after it restores the
  version 1.0 state).
- **Deposit procedure (style gate round 2, finding 2):** create Zenodo drafts for K/V, this paper and Ethics Packs
  version 1.1 and reserve all three version DOIs before publishing any. Then:
  - here, `nexus2026kv` gets K/V version 1.1's reserved version DOI;
  - in K/V, `nexus2026topology` gets this paper's version 1.1 title and reserved DOI, and "whose correction is in
    preparation" becomes that citation;
  - in Ethics Packs, the same for both entries (its `.bib` already lists Edrington as the author on
    `nexus/pharos-corrections-v11`; the title is still version 1.0's).
- **Zenodo description:** paste `zenodo_v11.md` over the pre-filled metadata (the 14 September description carries the
  withdrawn "cache tensors" claim). It is to be gated with the two editions.
- **Repository metadata at deposit:** `paper.json` (abstract's "in preparation", `related_page`, date), the README row
  and the `RESEARCH_STATUS.md` row; register #142 can close (each `.tex` now has one Acknowledgments section).
- Outside this paper: `null-swarm-paper/references.bib:71` and its academic copy hold an uncited entry for this paper
  under a title it never had. For that paper's own correction.
- Checked, nothing to add: the Pustovit name. "Aleksandr" entered in `6137b98` and was fixed to "Andrey" in `6c8e740`
  on 28 May, before version 1.0 was deposited, so version 1.1 corrects nothing silently there.
- Deposit order, in short: K/V first, then this paper, then Ethics Packs, each citing the others' reserved version
  DOIs. Style gate round 1's finding 7 (Ethics Packs repeating the cache framing) is already fixed on branch
  `nexus/pharos-corrections-v11` (Ethics `main.tex:31`: "put walk-encoded descriptions of knowledge graphs into the
  prompt as text"); this branch predates that fix.
- The Agni style gate on round 3's fixes (round 4).
- Whether the memory-graph file survives on the Studio (not searched).
