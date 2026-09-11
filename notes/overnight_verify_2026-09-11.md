# Overnight verification sweep — 2026-09-11, 03:47

**Read-only. No Zenodo writes, no commits, no pushes.** Purpose: confirm nothing drifted
after the republication pass of 2026-09-10/11.

## Verdict

**Everything checked passed. Nothing drifted.** One item of news: no reply from Arc yet.

---

## 1. Build gate — PASS, unchanged

```
Checked: 52  Stale: 0  Unverifiable: 0  Failed: 0  Skipped: 3
GATE: PASS
```

Identical to last night's result. The 3 skips are the long-standing never-built
`academic_main.tex` targets in formulary-paper, kv-cloak-defense-paper and
oracle-loop-paper — pre-existing, unchanged, and not caused by anything in this pass.

Repo state also unchanged: working tree clean (0 modified), nothing unpushed,
HEAD still `5044d9b`.

## 2. Zenodo concepts — 18 of 18 verified

Every concept still resolves to a record whose **file checksum matches the exact bytes we
shipped**, with no Wilkes in creators and the supersede note present. Checked
unauthenticated against the public API — what an outside reader sees, not what our own
tooling believes it did.

All 18: oracle-loop, formulary, kv-cloak-defense, lyra-technique-ii, decision-state,
emotion-accumulation, graph-topology, spectral-shape, user-model, delta-manifold,
cache-tracing, emotional-trajectory, identity-geometry, presence-metric, waystations,
null-swarm, meta-pattern, mine5-selective-sharpener.

## 3. Citation metadata — 150 of 150 live

**17/17 records correct; 150 `cites` relations live**, counts matching the manifest exactly.

### 3b. Retired duplicates — 8 of 8 still marked

All eight still carry both an `isIdenticalTo` relation pointing at the correct surviving
concept DOI **and** the duplicate notice in the description. Nothing reverted.

### 3c. mnemosyne-benchmark — correct

Concept `21801642` serves record `22704386`, creators `['Edrington, Thomas']`, file md5
`5c3929b5…` — an exact match for the **academic** edition shipped last night. The edition
swap held.

> **Gap worth recording:** `verify_all.py` covers only the 18 concepts in `vplan.json`. It
> does **not** cover mnemosyne-benchmark (deposited after that plan was written) or the 8
> retired duplicate lineages. I checked those separately for this sweep, but the tool would
> have reported a clean 18/18 while silently ignoring 9 other live records. A tool whose
> scope is narrower than the claim it appears to support is the same failure shape as a
> check that cannot fail. **Widen `verify_all.py` to read live state rather than a
> point-in-time plan file.**

## 4. Message to Arc — sent, no reply yet

`from_lyra_to_arc_20260910_persona_intensity_doi.md` (3,279 bytes, Sep 10 23:02) is the
**newest file in the inbox**. No reply has arrived in the ~4.75 hours since.

Nothing is wrong with the channel: prior Arc traffic exists
(`from_arc_helsinki_specs_for_nexus.md`, `from_arc_loam_watcher.md`, and four
`from_ang_arc_*` threads from July), so Arc does use this inbox. Four hours overnight is
simply not long enough to read anything into.

**No reply was sent, per instructions.** Ang's Persona Intensity DOI
(`10.5281/zenodo.20157722`) remains untouched.

---

## What this sweep did NOT check

Stating these so a clean report is not mistaken for total coverage:

- **Content of the republished PDFs.** Checksums prove the right *bytes* are being served;
  they say nothing about whether those bytes make the right claims. The claim-level
  corrections were verified last night at deposit time, not re-verified here.
- **The 46 older versions** that still list Wilkes in creators. That is correct by design —
  a version is a historical snapshot and its metadata should describe its own content — but
  no check here re-confirms it.
- **Whether anything external has begun citing the new DOIs.** Too early; indexes take days
  to weeks.

## Open for Thomas

1. **Upload the integrity edition to the website** — `mnemosyne-benchmark/main.pdf`. The
   only remaining item I cannot do.
2. **Rename and rotate `ZENODO_SANDBOX_TOKEN`** on MTH — it is a production credential.
3. Nothing else is blocked on you.
