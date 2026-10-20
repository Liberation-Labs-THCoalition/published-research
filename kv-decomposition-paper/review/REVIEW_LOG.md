# Review log: K/V decomposition paper, version 1.1

What checked this version, when, and where each round's findings and fixes are recorded. Commits are on branch
`nexus/pharos-corrections-v11` of this repository.

## Before version 1.1

- **Agni gates, May 2026.** Recorded in KV-Experiments `graph-topology-injection/FINDINGS.md` and in the drafts of
  28 May (`8f7d889`) and 31 May (`27f86e4`). The complementarity hypothesis was approved as exploratory at Gate 1 and
  passed Gate 3. The injection-over-text hypothesis failed Gate 3 as underpowered. The final revision of 1 June
  (`2485b19`) deleted both qualifications.
- **Moved out of the publishing queue, 5 June** (`944f98e`, "BLOCKER status, not in publishing queue"). The reason
  is not recorded.
- **Restored, 22 July** (`d272803`, "All cleared Agni gate with fixes applied"). No record of that clearance found.
- **Agni style gate (opus), 2026-09-04: REJECTED** (`tools/agni/style/kv-decomposition-paper.agni_style.json`).

## Version 1.1, 19 October 2026

| Round | Who | Verdict | Fixes |
|---|---|---|---|
| Scoping review | Nexus, the first author, before drafting | Scope of the correction | Draft `9e21c40` |
| Second reader, round 1 | An independent Claude subagent in a read-only sandbox, recomputing every number from the primary files with its own code | MAJOR FIXES | `11a2c2a` |
| Second reader, round 2 | The same reader, re-reading the fixes | MINOR FIXES | `2d96b19` |
| Authorship change | At Lyra's request, as its own commit (AGENTS.md section 3) | | `8fe7493` |
| Correction notes and abstract | At Lyra's request | | `9ccf1e4` |
| Second reader, round 3 | The same reader, on `8fe7493`, `9ccf1e4` and `6e3cfff` | MINOR FIXES | `f855867` |
| Agni style gate (opus), both editions | `tools/agni/style/kv-decomposition-paper.v11-r1.agni_style.json` | CONDITIONAL, 20 findings | The commit that adds this log |

The second reader's reports were returned to the first author's session. Their findings and the fixes made for each
are listed in the commit messages named above.

## Still open before deposit

- Kavi: whether the "verification review" thanked in version 1.0 took place, and their acknowledgment wording.
- CC: wording for their acknowledgment (version 1.0's "attention-steering connection" came from the 1 June revision
  and has no source).
- Thomas Edrington: sign-off on the academic edition (its only author), the reason for the June 5 BLOCKER if known, and
  the new title.
- Lyra: full read of both editions.
- The style gate's final verdict on this version.
- The companion graph-topology paper repeats the withdrawn superadditivity claim; its correction is in preparation.
