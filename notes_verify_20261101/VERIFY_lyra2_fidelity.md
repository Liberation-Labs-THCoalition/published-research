# Verifier 2 (fidelity): The Lyra Technique II v1.1, PR #19

Checked at head 2514f7334f172b4319fe3fee2cf80305c610c999 (branch lyra/followup-withdrawn-claim-contacts).
Read-and-run only. 18 tool calls. The working tree was clean before and after.

## Result

The changes pass every check in the brief. The findings below are all low severity.

## Checks and their results

| Check | Result |
|---|---|
| Edition sync (`diff main.tex academic/main.tex`) | Byline, plus the random-direction-control paragraph sitting in a different place (identical text, inside the Withdrawn section in both). The same two hunks appear at d7c327e, so v1.1 adds no new drift. |
| PDFs match the .tex | Both editions rebuilt with MiKTeX latexmk into a scratch outdir: exit 0, 0 `!` errors, 0 undefined refs, 25 pages each. The `pdftotext -layout` diff between the committed and rebuilt PDFs is 0 lines for both editions. |
| Overfull boxes | 8 per edition, largest 26.80pt, in the committed and rebuilt logs alike. This matches the PR body. (My first grep reported 0 because of a quoting error. A second pattern found 8.) |
| verify_pdf, present direction | Control "Lyra Technique II". Present: "Version 1.1", "Persona Intensity in KV-Cache Geometry", "Injection Disambiguation (Withdrawn)". Exit 0 on both v1.1 PDFs; exit 1 on both d7c327e PDFs (all three present strings FAIL there). |
| verify_pdf, absent direction | Absent "Version 1.1" on the v1.1 PDF gives exit 1, and a planted nonsense control gives exit 1, so both failure paths work. |
| LF / tabs | CR=0 and TAB=0 in both .tex files, paper.json, README.md and RESEARCH_STATUS.md. `git diff --check d7c327e..HEAD` is clean. |
| Byline / credit | No added or removed line in d7c327e..HEAD touches `\author`, `\thanks` or the author lists. The edition byline diff is the same as at d7c327e. |
| Title change listed for Thomas | Yes, under "For Thomas: the title veto" in the PR body, with the old and new titles. It is isolated in e2d759b. Reverse-applying e2d759b's .tex/.json diff onto HEAD gives `git apply -R --check` rc=0. The PDFs need rebuilding after a revert, as the PR says. |
| Numbers vs source (R1R2_RESULTS.md 28, 35-40) | 0.358 vs 0.382 (line 28); S1 0.410 vs 0.370, Holm 0.071 (line 36); S2 0.992 vs 0.963, Holm 0.006 (line 37). All appear the same way in the Correction box, the abstract and paper.json. 0.348 and 0.872/L43 are not on those lines: I matched them to the brief, not to the source. |
| "May 2026" for v1.0 | Supported: paper.json has `"date": "2026-05"`, and the file was first committed on 2026-05-22. The old date line was `Draft --- \today`, as the Correction Note says. |
| Private-info scan of added lines | 689 added lines (tex/json/md) scanned for emails, Tailscale/LAN IPs, Windows paths, credentials, ports, host names, family names and tokens: 0 hits. Planting one line (email + 100.x IP + credentials path) gives 1 hit, so the scan can fire. |
| PR state | OPEN, isDraft=true, head = 2514f73, title "Follow-ups + The Lyra Technique II v1.1 correction". |

## Findings (ranked; all low)

1. **Edition drift beyond the byline (pre-existing).** The random-direction-control paragraph comes before
   the Linearity caveat in academic and after it in integrity (integrity main.tex:498-502 vs academic
   main.tex:489-493). The brief allows only the byline and CRediT to differ. This predates v1.1, and the PR
   body discloses it ("as before"). Both copies sit inside the Withdrawn section with identical text, so a
   reader is not misled. One move would remove the drift.
2. **The PR body slightly overstates the bib citations.** It says decision-state, emotion-accumulation and
   kv-cloak-defense "still cite the v1.0 title". The last two do (full title). decision-state's
   references.bib:134 cites the shorter "SVD Denoising and Directional Projection", which is the old
   paper.json title. I did not check "That is what the DOI published" against Zenodo
   (emotion-accumulation's entry carries doi 10.5281/zenodo.20604329).
3. **Hedge before the claim in the three-levels list** (integrity main.tex:1016-1023 and the academic twin).
   The item still opens with the assertion "Carries *directional alignment* with specific concept vectors"
   and only then says the readout is under correction and "proposed, not shown". Main.tex:1027-1030 ("A
   complete reading of model state ... requires all three") is qualified in the same sentence. Neither is
   wrong. The first could lead with the hedge.
4. **Mixed tense, "under correction" vs "corrected".** The intro (main.tex:231), the three-levels item
   (1020), the Discussion (1047), 1214 and 1455 say "under correction at source", while the Correction box
   says the source "found" the result. The two are consistent, because #17 is still unmerged, but a reader
   sees both terms. No change is needed while #17 is open. Recheck when it merges.
5. **A title veto also needs two status rows edited.** README.md:20 and RESEARCH_STATUS.md:158 (commit
   2514f73) say "title change proposed, pending Thomas". If Thomas vetoes, those rows need a one-word edit
   beyond "revert e2d759b and rebuild the two PDFs". This is cosmetic.

## Instrument notes (not findings)
- `verify_pdf --absent "Two new measurement channels"` FAILs on v1.1. The Correction Note quotes the v1.0
  abstract and intro (main.tex:1477, 1484), which is verify_pdf's documented case 4 (a correction quotes
  what it corrects), not a residual claim.
- `git diff --binary e2d759b^ e2d759b | git apply -R --check` fails on the PDFs, even though the blobs at
  e2d759b and HEAD are identical (f55f029). The PDF has `diff=astextplain` set. Only the .tex/.json revert
  check counts, and it passes.
- Scratch builds are in the session scratchpad (`v2fid/`), and nothing was written to the repo except
  this file.
