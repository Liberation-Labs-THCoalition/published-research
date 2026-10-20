# Review log: Ethics Packs paper, version 1.1

What checked this version, when, and where each round's findings and fixes are recorded. Commits are on branch
`nexus/pharos-corrections-v11` of this repository.

## Before version 1.1

- **First draft, 30 May** (`dc332bf`). Its validation paragraph credited CC with identifying the timeout confound
  through log analysis of the v3 run; CC's letter to Nexus that day (11:55, "MoReBench timeout confound") is the
  record.
- **Final revision, 1 June** (`2485b19`) removed the timeout table and its discussion and reduced CC's credit to
  "critical methodology feedback".
- **Moved out of the publishing queue, 5 June** (`944f98e`, from Thomas Edrington's account). `RESEARCH_STATUS.md`
  (In Human-Review table, 28 June snapshot) gives the blocker as "n=5, keyword-matching confound".
- **Restored, 22 July** (`d272803`, from Nexus's account, a Claude Opus 4.6 session: "All cleared Agni gate with fixes
  applied"). No record of that clearance found.
- **Revised, 24 and 28 July** (`a0fd1e2`, `c7e6f21`, from Thomas Edrington's account, one Claude Opus 4.6 session,
  the same Claude-Session ID on both). The invented reference metadata entered here.
- **Agni style gate (opus), 2026-09-04: REJECTED** (`tools/agni/style/ethics-pack-injection.agni_style.json`).

## Version 1.1, 19 October 2026

| Round | Who | Verdict | Fixes |
|---|---|---|---|
| Scoping review | Nexus, the first author, before drafting | Scope of the correction | Draft `4d68a86` |
| Second reader, round 1 | A separate Claude instance, launched from the first author's session, working read-only on copies of the files and recomputing every number with its own code | MAJOR FIXES | `11a2c2a` |
| Second reader, round 2 | The same reader, re-reading the fixes | MINOR FIXES | `2d96b19` |
| Authorship change | At Lyra's request, as its own commit (AGENTS.md section 3) | | `8fe7493` |
| Correction notes | At Lyra's request | | `9ccf1e4` |
| Second reader, round 3 | The same reader | MINOR FIXES | `f855867` |
| Agni style gate (opus), both editions | `tools/agni/style/ethics-pack-injection.v11-r1.agni_style.json` | CONDITIONAL, 16 findings (11 major) | The commit that adds this log |

Every finding was checked against the primary files before it was applied. The gate could not run Python, read the
pack files or the benchmark copy, or reach the network; its arithmetic was by hand. The first author recomputed its
two central findings from the run files: v3's 13 empty responses are 11 logged timeouts and 2 unlogged injected
empties (Gauthierian's program-funding dilemma, Aristotelian's sentencing dilemma); dropping only the 8 pairs with a
logged timeout leaves -0.051 over 17 pairs; the failures fall in the frameworks run second to fourth, in both arms.
DOIs checked against the Zenodo API (19 October): version 1.0 is record 21754904, concept 21754903, creator Edrington
only.

## Still open before deposit

- Thomas Edrington: sign-off on BOTH editions, and approval of the note's sentence on the invented reference metadata
  specifically (it names commits made under his account by a Claude session).
- Lyra: full read of both editions. Lyra's own acknowledgment line says "reviewed this correction"; it is true only
  once that read is done, so deposit waits for it. Also: whether "style guidance" (in version 1.0's acknowledgments)
  stays.
- ~~CC's credit~~ Closed 2026-10-19: CC confirmed (letter 20:55, from CC's own transcript of 30 May 11:41 and CC's
  letter to Nexus of 11:55 that day) that the 30 May credit is right and asked for it back, with CC named in the AI
  disclosure. Restored in both editions, the academic CRediT and the AI disclosure, in its own commit.
- Our MoReBench copy: record its source and SHA-256 (it is on the machine that ran the evaluation).
- Push, and pin the commit in Data and Code (the text has a placeholder).
- The style gate's final verdict on this version.
- Lab papers cited with the creators their Zenodo records list (Thomas Edrington alone). Thomas to confirm that rule.
