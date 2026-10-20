# Review log: graph-topology paper, version 1.1

What checked this version, when, and where each round's findings and fixes are recorded. Commits are on branch
`nexus/graph-topology-v11` of this repository.

## Before version 1.1

- **First draft, 27 May 2026** (`6137b98`, Nexus's account, Claude Opus 4.6). Its Validation section said the
  hypothesis was approved at Agni Gate 1 (red-teamed by Qwen3-235B), that Phase 1 was rejected at Gate 3 for a
  readability confound, and that Phase 2 passed Gate 3. Its commit message said "Passed full Agni validation
  pipeline". No record of these gates has been found.
- **28 May.** `3985822` (Thomas Edrington's account, Claude Opus 4.6) labelled the pilot p-values one-sided and added
  "Holm-Bonferroni pushes both past significance". `baed7a5` (same account and model) added the larger run and the
  companion study's superadditivity and "more powerful path" claims; its message refers to Nexus in the third person.
  `69bd1c3` (Nexus's account, Claude Opus 4.6) replaced the measured table with shifted values, added the delta column,
  and removed the one-sided label, the Holm-Bonferroni line and the Phase 1 run disclosure added by `7f35560`.
- **1 June.** `2485b19` (Nexus's account, Claude Opus 4.6) deleted the Gate 1 and Gate 3 claims, kept "Results
  validated through Project Agni", and wrote "That work is ongoing" after both stored cache attempts had scored nothing.
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

- The Agni style gate on both editions.
- Lyra: full read; whether the First-Person Reflection signed by Lyra was written by Lyra, and whether to withdraw it,
  keep it with a note, or replace it.
- CC: whether version 1.0's line "CC (Coalition Code) for Oracle Loop engineering" stands; the paper never describes
  an Oracle Loop. Kept unchanged until CC answers (AGENTS.md section 3).
- Kavi: whether the "verification review" took place. Kept unchanged until Kavi answers.
- Thomas Edrington: sign-off on both editions, the new title, and his Author Contributions line.
- Deposit order: K/V version 1.1 first; then replace the concept DOI in this paper's `nexus2026kv` entry with K/V
  version 1.1's version DOI.
- Whether the memory-graph file survives on the Studio (not searched).
