# VERIFY 2: fidelity, editions, PDFs, PII (PR #17 and PR #18)

> Committed copy (2026-11-01): local home paths and the Starship username are redacted. Otherwise verbatim.

Verifier 2. I only read and ran things; this report is the only file written. Scratch builds are in
`<home>\AppData\Local\Temp\claude\<home-slug>\fedc44e2-d3b2-49fe-9e89-5890ff7e37ba\scratchpad\v2\build\`.
No UMV-L1 or label-validity results were opened.

## What passed

- **Both PRs are drafts.** `gh pr view 17`: isDraft=true, OPEN, base master, head `34daa8d808644ed7a909c76022f588238b21097e`. `gh pr view 18`: isDraft=true, OPEN, base master, head `089bb2e919c123a74b4a01b5500320de7e9b77d1`.
- **PR body endings.** Both bodies end with the Claude Code line, a blank line, then `https://claude.ai/code/session_01GToYR61pPGkaidXwDEzyff` (checked with `od -c` on the tail of each body).
- **Commit trailers.** These commits end with the exact two required lines: `46750d7`, `533875d`, `d4f11da` and `34daa8d` (PR 17), and `089bb2e` (PR 18). See finding 6 for the 18 older commits.
- **PDF rebuilds.** I extracted each paper with `git archive HEAD` into a temp dir and ran `latexmk -pdf`. All six builds returned rc=0. Sizes match the committed PDFs exactly. The bytes differ, which is expected from timestamps and IDs. `pdftotext -layout` gives identical text for committed and rebuilt in all six PDFs: um flight/academic, decision-state flight/academic, emotion-accumulation flight/academic. Page counts: 26 / 27 / 15 / 16 / 10 / 10.
- **verify_pdf, checked both ways.** Control `Liberation Labs` is OK on every file. A planted needle `PLANTED-NOT-IN-ANY-PDF-xyzzy` FAILs on committed, rebuilt and master, so the check can fail.

  | PDF | needle | HEAD and rebuilt | master |
  |---|---|---|---|
  | um flight | `(Lyra, an AI system)` | OK | FAIL |
  | um flight | `Story Emotion in the KV Cache` | OK | FAIL |
  | um academic | `(Lyra, an AI system)` | OK | FAIL |
  | um academic | `AI disclosure: Lyra` | OK | FAIL |
  | ds flight | present `The detector does not sidestep` | OK | FAIL |
  | ds flight | absent `the geometry is the stable signal` | OK | OK (the flight edition never had it) |
  | ds academic | present `The detector does not sidestep` | OK | FAIL |
  | ds academic | absent `the geometry is the stable signal` | OK | FAIL (so it was removed) |
  | ea flight | `and its injection test is withdrawn` | OK | FAIL |
  | ea academic | `and its injection test is withdrawn` | OK | FAIL |

- **Editions changed together.** Every commit that touches a `.tex` file touches both editions, with one exception: `878c219`, a flight-only author footnote, named "AUTHOR LINE (flight edition only)". I diffed the added lines of each pair (sync.py, difflib). The only unmatched lines are rewraps between wordings that were already different (decision-state judge-reliability item), the two author footnotes, and the pre-existing limitations drift in finding 8. The residual-claim grep agrees across editions. "sidestep" is now negated in both: ds `main.tex:760`, `academic/main.tex:770`. The em-acc correction note is at `main.tex:136-152` and `academic/main.tex:135-151`. The em-acc `paper.json:8` abstract is updated.
- **LF and `git diff --check`.** No `.tex`, `.bib` or `paper.json` added in either PR contains a CR byte. Counted in Python on the HEAD blobs. `git diff --check` limited to tex/bib/md/json outside `correction_v2/` is clean in both. On PR 18 the full check returns rc=0.
- **d4f11da (credit disclosure).** No name is added to or removed from any byline. The diff changes only the label of the box ("First-Person Reflection" becomes "... (Lyra, an AI system)") and the text of Thomas's academic footnote.
- **4230a8a.** The `paper.json` "authors" list was only reformatted. It still reads `["Lyra", "Thomas Edrington"]`.

## Findings, ranked (8)

1. **A citation byline was changed in a commit not named as a credit change, and the PR does not flag it.** Commit `e9c16d8` ("fix factual errors Agni found in RETAINED claims") changes `lyra2026oracle` in `user-model-paper/references.bib:16`:
   - from `{Lyra and Edrington, Thomas and CC}`
   - to `{Lyra and Edrington, Thomas K. and Vera}`

   CC is removed and Vera is added. The new byline does match the primary: `oracle-loop-paper/main.tex:67-75` names Lyra, Thomas K. Edrington and Vera, and CC appears only in the acknowledgements at `:126`. So the fix is right. But the standing rule says a removal from a byline is flagged for Thomas and goes in its own commit named in the subject. The PR 17 body flags only `lyra2026detecting` and `lyra2026concordance` (body line 45). **Action:** add a line about this to the "for Thomas" list in the PR body.

2. **PII: local paths, usernames and host names in added files, in a PUBLIC repo.** `gh repo view` reports visibility PUBLIC. All hits are in PR 17's older commits (`2a1d23c`, `7064cf8` and the review records). None are in the four new commits, and PR 18 has none.
   - `user-model-paper/correction_v2/R1_INVESTIGATION.md:232`: a full `<home>\AppData\Local\Temp\claude\...\scratchpad\r1_inv\` path.
   - `user-model-paper/review/agni_paper_um_v2_corrections.json:6` and `..._r2.json:6`: `<home>/Desktop/LiberationLabs/Research/corrections/user_model_v2/...` and `/Users/<user>`.
   - `..._r2.json:10`: a `~/.claude` path.
   - `/Users/<starship-user>/...` appears 16 times, for example `PREREG_um_v2_R1R2.md:122` and `R1R2_DESIGN_NOTES.md:35, :669`. The ssh alias `<starship-user>` is also exposed.
   - `Starship` appears 57 times, for example `user-model-paper/README.md:97`. This is the name the papers already use, so it is likely acceptable.
   - `results_20261019/encode.log:7`: a home path (`<home>`). Also `~/observer-env`, `~/user_model_rerun`, `~/user_model_v2`, `~/agni_runs` and `~/lab`, all in `R1R2_DESIGN_NOTES.md`.

   The scan found no IPv4 addresses, keys or emails. Planted control: one line holding a Tailscale IP, a `C:\Users` path, a `~/.credentials` path and a host name was caught as ipv4, win_path, unix_home, key and host. A second control line caught the email pattern. Lines scanned: 1,165,268 added in PR 17, 227 added in PR 18. **Caveat:** the IPv4 pattern skips addresses whose first octet is a single digit, to avoid version strings. **Action:** Thomas decides whether the usernames and paths stay. Several of these files are under `SHA256SUMS`, so any redaction has to update those hashes.

3. **PR 18's citations print the superseded title, which itself states the withdrawn claim.**
   - `decision-state-paper/references.bib:127` (kept) and the new entry in `emotion-accumulation-paper/references.bib` (`@@ -44,3 +44,9`) both carry the title "Emotion Geometry in the User Model: {$W_K$} Projection Reads Valence from {KV}-Cache Keys".
   - PR 17 adopts "Story Emotion in the {KV} Cache: A Preregistered Test Against a Text ({TF-IDF}) Baseline" (`user-model-paper/README.md:103`).

   The bibliographies of all four rebuilt downstream PDFs therefore assert "W_K Projection Reads Valence", while their bodies withdraw it. **Action:** sync the title in PR 18, or add "(original title)" to the note. Either way the change must be coordinated with PR 17.

4. **The same claim survives in papers neither PR touches.** This is a gap in the sweep, not a defect in either diff.
   - `lyra-technique-ii/main.tex:480, :843, :1258` and `academic/main.tex:478, :841, :1256`: "W_K projection: AUROC 0.992", with the table row marked "Confirmed".
   - `lyra-technique-ii/paper.json:20`: "injection into neutral text confirms model state".
   - `meta-pattern/academic/main.tex:247`: "binary valence at AUROC 0.992" cites the user-model paper.
   - `graph-topology-paper/references.bib:50`: the user-model entry.
   - `RESEARCH_STATUS.md:157-158`.

   Source: `git grep` at HEAD of the downstream worktree. `REMEDIATION_REGISTER.md:672` already lists one part of the lyra-technique-ii problem.

5. **verify_pdf can't see an em dash in the emotion-accumulation PDFs, and can't see a dropped "fl" ligature.**
   - **Em dash:** in both em-acc editions, at HEAD and on master, `---` extracts as U+0016, not U+2014. The extraction holds 0 em dashes, and "emotional weather\x16the current turn" was found by Python repr. So any em-dash needle FAILs its control on the correct PDF; I watched it happen on all six em-acc runs. The brief's rule to use the em dash form for a `---` string cannot work on these two papers. Use dash-free needles there.
   - **Ligature:** `Reflection` FAILs on every user-model PDF, including the new ones. The extraction reads "First-Person Reection" (`um_flight.c.txt`), so the dropped-ligature tolerance in `tools/verify_pdf.py:64` (`needle_regex`) does not cover this case. That is a false FAIL; the check never reports success it hasn't earned.

6. **Trailers on the older PR 17 commits name earlier sessions.** 17 commits end with `Claude-Session: .../session_015e9ubekQVwLTMJc1bF9ciP` and `00fd803` ends with `.../session_01QDzFfLMsw9kW9SRTivSg8F`. The Co-Authored-By line is exact on all of them. I expect this, since earlier sessions wrote those commits. It is only a defect if the rule means every commit on the PR must carry this session.

7. **CRLF and `git diff --check` in `correction_v2/` (PR 17).** Nine files there contain CR bytes:
   - `um_v2_analysis.py` 1,073
   - `R1R2_DESIGN_NOTES.md` 879
   - `um_v2_dryrun.py` 726
   - `um_v2_encode.py` 719
   - `PREREG_um_v2_R1R2.md` 530
   - `prereg_um_v2_to_json.py` 167
   - `prereg_um_v2_R1R2.json` 158
   - `encode.log` 29
   - `chain.status` 1

   The full `git diff --check` returns rc=2 on these files and on the two user-model PDFs, which git diffs as text. This is deliberate: `correction_v2/.gitattributes` says the files are byte-exact copies whose hashes are registered, with `* -text`. Both worktrees have `core.autocrlf=true`, so that attribute is what protects them. It is documented, but it is an exception to "LF only", so name it in the PR body if it isn't there already.

8. **Pre-existing drift between the two user-model editions (not introduced here).**
   - Academic keeps the "Linear probes only" and "Single-turn design" limitation items live (`academic/main.tex:548, :556`).
   - Flight has them commented out (`main.tex:550, :559`).
   - It was the same on master: flight `:563, :582` commented, academic `:561, :577` live. The cause is `36c222f` ("Style guide: trim limitations").

   No action is needed in these PRs. It is listed so that nobody reads it as a regression.
