# Review log: K/V decomposition paper, version 1.1

What checked this version, when, and where each round's findings and fixes are recorded. Commits are on branch
`nexus/pharos-corrections-v11` of this repository.

## Before version 1.1

- **Agni gates, May 2026.** Recorded in KV-Experiments `graph-topology-injection/FINDINGS.md` and in the drafts of
  28 May (`8f7d889`) and 31 May (`27f86e4`). The complementarity hypothesis was approved as exploratory at Gate 1 and
  passed Gate 3. The injection-over-text hypothesis failed Gate 3 as underpowered. The final revision of 1 June
  (`2485b19`) deleted both qualifications.
- **Moved out of the publishing queue, 5 June** (`944f98e`, from Thomas Edrington's account, "BLOCKER status, not in
  publishing queue"). `RESEARCH_STATUS.md` (In Human-Review table, 28 June snapshot) gives the blocker as "No 135M
  data, acquiescence concern". (Version 1.1's first drafts said the reason was not recorded; the style gate's second
  round found this record.)
- **Restored, 22 July** (`d272803`, "All cleared Agni gate with fixes applied"). No record of that clearance found.
- **Agni style gate (opus), 2026-09-04: REJECTED** (`tools/agni/style/kv-decomposition-paper.agni_style.json`).

## Version 1.1, 19 October 2026

| Round | Who | Verdict | Fixes |
|---|---|---|---|
| Scoping review | Nexus, the first author, before drafting | Scope of the correction | Draft `9e21c40` |
| Second reader, round 1 | A separate Claude instance, launched from the first author's session, working read-only on copies of the files and recomputing every number with its own code | MAJOR FIXES | `11a2c2a` |
| Second reader, round 2 | The same reader, re-reading the fixes | MINOR FIXES | `2d96b19` |
| Authorship change | At Lyra's request, as its own commit (AGENTS.md section 3) | | `8fe7493` |
| Correction notes and abstract | At Lyra's request | | `9ccf1e4` |
| Second reader, round 3 | The same reader, on `8fe7493`, `9ccf1e4` and `6e3cfff` | MINOR FIXES | `f855867` |
| Agni style gate (opus), both editions | `tools/agni/style/kv-decomposition-paper.v11-r1.agni_style.json` | CONDITIONAL, 20 findings | `29a2d20` |
| Agni style gate (opus), round 2, both editions | `tools/agni/style/kv-decomposition-paper.v11-r2.agni_style.json` | CONDITIONAL, 13 findings | The commit that adds that file |

The second reader's reports were returned to the first author's session. Their findings and the fixes made for each
are listed in the commit messages named above.

Round 2 of the style gate could not read the KV-Experiments repository or the network, so it did not check the
follow-up runs' numbers (the second reader recomputed them in rounds 1 to 3) or the DOIs (checked by the first author
against the Zenodo API on 19 October: version 1.0 is record 21754900, concept 21754899, creator Edrington only). It
found the BLOCKER record above and the author-name change below. arXiv 2604.03270 lists Andrey Pustovit and makes the
RoPE point the Introduction attributes to it; the drafts and the June version printed "Aleksandr".

## Still open before deposit

- Kavi: whether the "verification review" thanked in version 1.0 took place, and their acknowledgment wording.
- CC: wording for their acknowledgment (version 1.0's "attention-steering connection" came from the 1 June revision
  and has no source). The note now says so, with CC's answer pending; if CC is thanked, the academic AI disclosure
  names CC too.
- Thomas Edrington: sign-off on BOTH editions (he is on both bylines); confirmation of his flight-edition line
  "direction and the initial insight" (written in the 1 June revision, like CC's withdrawn line) and of his academic
  CRediT roles (Conceptualization, Supervision, Resources, Project administration); whether the status table's
  blocker was the reason for the 5 June move; the new title.
- Lyra: full read of both editions. Lyra's own acknowledgment line says "reviewed this correction"; it is true only
  once that read is done, so deposit waits for it. Also from Lyra: which earlier KV-cache geometry work to cite for
  "builds on", and whether the academic CRediT's Methodology row stays (AGENTS.md section 3: not deleted unasked).
- The style gate's final verdict on this version.
- The companion graph-topology paper repeats the withdrawn superadditivity claim and cites this paper by its version
  1.0 title; its correction is in preparation.
- Lab papers are now cited with the authors their DOI records list (Zenodo lists Thomas Edrington alone on both
  records). Thomas to confirm that rule.
