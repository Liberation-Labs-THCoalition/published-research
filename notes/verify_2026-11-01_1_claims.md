# VERIFY 1: claims against primaries (PR #17, PR #18)

> Committed copy (2026-11-01): local home paths and the Starship username are redacted, and one quoted pointer sentence is paraphrased. Otherwise verbatim.

Written 2026-11-01 by verifier 1 (read-only; this file is the only write). 23 tool calls.
PR #17 head `34daa8d`, PR #18 head `089bb2e`. Both are draft with base master (`gh pr view`). Both are 0 behind `origin/master` `f1ee90f`.
Primaries used:
- `corrections/user_model_2026a_correction_plan_20261014.md` (the plan)
- `corrections/user_model_v2/R1R2_RESULTS.md` (byte-identical to the released `correction_v2/R1R2_RESULTS.md`, per `cmp`)
- `data/qwen_emotion_bridge_summary.json`
- `code/emotion_geometry_bridge.py`
- Project-Oracle `d0c8395`
- the Key Numbers wording in the brief

No UMV-L1 results file was opened, and this report contains no UMV-L1 figure.

## Positive control (planted unlike case)
I wrote a checker (scratchpad `vchk/chk.py`) with four rules:
- every paragraph with 0.992 must also contain "300" and "arousal";
- 0.992 must not be attributed to a W_K probe or projection without a negation;
- any 12.3x must sit next to a TF-IDF or text-baseline qualifier;
- withdrawn phrases must not appear unless marked as previous or removed.

The plant was a copy of decision-state `main.tex` with the sentence "The W_K probe reads valence at AUROC 0.992" put back in. **It was caught** by the 0992_unqualified rule.

The 0992_as_WK rule did not fire on the plant, because my escaping put a tab into `\textsubscript`. It did fire on the May quotations in user-model App. A (flight :624/:628, academic :627/:631), so both rules can fail. On the real files, every user-model flag is May text that is quoted or attributed to "the May version". The real downstream flags are finding 1.

## Ranked findings

1. **PR #18: the 12.3x / 40.9% decode is cited without the corrected paper's central qualification.**
   - **Where:** both editions of both papers and `paper.json`:
     - decision-state `main.tex:130`, `academic/main.tex:131`;
     - emotion-accumulation `main.tex:63, :102`, `academic/main.tex:62, :101`;
     - `paper.json:8`.
   - **What's missing:** these lines say the citation "supports only the narrower point that key geometry carries the content of the text". They leave out the main corrected result: the decode is not above a text (TF-IDF) baseline.
   - **The primaries:**
     - R1R2_RESULTS.md:28 gives NOT_ABOVE_TEXT_TFIDF: 0.358 against 0.382, 95% interval [-6.4, +1.6].
     - R1R2_RESULTS.md:61-65 shows that v2's 40.9% depended on fitting FWL on the test fold. With FWL fitted on the training fold it reads 0.348.
     - The user-model paper's own text carries this at :70-75, :117, :307 and :331.
   - The decision-state correction note calls 40.9% "the decode the cited work supports". The corrected paper's headline is 35.8%, and that does not beat text (TF-IDF).
   - **Cause:** the Key Numbers row omits this too. The brief's wording was followed, but it falls short of the primary.
   - **Smallest fix:** in each of these sentences, add after the 12.3x clause: "; on the same stories it does not exceed a text (TF-IDF) baseline". Lyra may also want to add "not above text (TF-IDF)" to the Key Numbers row.

2. **PR #18: the bibliography title asserts the withdrawn claim.**
   - **Where:** decision-state `references.bib:127`, and emotion-accumulation `references.bib:48` (added in this PR).
   - **What's wrong:** both read "Emotion Geometry in the User Model: {$W_K$} Projection Reads Valence from {KV}-Cache Keys". That title states the claim the correction withdraws, and it does not match PR #17's adopted title. PR #17's title is "Story Emotion in the KV Cache: A Preregistered Test Against a Text (TF-IDF) Baseline" (`user-model-paper/paper.json:2`).
   - **Smallest fix:** use the adopted title in both `.bib` entries and rebuild the four PDFs.

3. **The Key Numbers W_K-bridge row disagrees with the primary. The paper is right.**
   - **What the paper says** (flight :86, :406, :410, :640; academic :85, :405, :409, :643): the peak is |rho| = 0.872 at L43. 0.862 is the L35 value. The Mantel 0.090 is unverified.
   - **What the primary says:** `data/qwen_emotion_bridge_summary.json` has `bridge_analysis/35/valence_pc1_rho` = 0.8621, `/43` = -0.8721, and `best_bridge_valence_rho` = -0.8721.
   - **Mantel:** `grep -ril mantel` over `data/` and `correction_v2/*.json` finds only a story sentence. No released file holds the Mantel value.
   - **So:** "does not predict cache geometry (Mantel 0.090)" states as established a number with no released source, and "0.862 at L35" is not the peak.
   - **Smallest fix:** change the Key Numbers row in MEMORY.md, not the paper. Neither PR cites the bridge downstream.

4. **PR #17: the released `correction_v2/R1R2_RESULTS.md` adds new UMV-L1 pointers.**
   - **Where:** :18 (the smoke test reproduced the loader hang) and :184-185 (a sentence that points into `label_validity/`). Both point into `label_validity/` files that are not in the release.
   - **Is any result in them?** No. The 84 / 816 counts are R1R2's own leak split, not label-validity results.
   - **Why it's still a problem:** the brief allows a pointer only where the existing text already had one, and this file is new to master.
   - **Smallest fix:** in the released copy, replace both sentences with "(a separate label-validity study, not part of this release)", then regenerate `SHA256SUMS:11`. The other option is Lyra's explicit OK to keep them.

5. **PR #17: the body of commit `26940f6` misdescribes the file.**
   - It says the released R1R2_RESULTS.md "keeps … a literal, unfilled" clock placeholder in its addendum heading.
   - `grep -n NOW` on the file returns nothing.
   - **Smallest fix:** correct it in the PR body. The commit is history.

6. **Low, out of the brief's scope: decision-state :66-68 calls its own encoding features "W_K projections".**
   - Whether that code applies W_K was not checked. It is the same class of mislabel the correction found.
   - **Smallest fix:** queue a code read. No change to this PR.

## Verified as correct (no finding)
- **The five errors, both editions.** The corresponding lines are identical between editions (Python comparison of flight/academic :183/182, :206/205, :320/319, :496/495, :521/520, :620/623, :648/651). Each one checked against its primary:
  - **Prompts:** quoted at :178 to match `code/emotion_geometry_bridge.py:230-243`. The blob `b5a36e42b6` is verified (`git rev-parse HEAD:…`).
  - **No W_K:** :206 matches plan §2 (class-mean keys, all positions, dot product, GroupKFold(5), FWL on the test fold, four layers).
  - **Valence:** :321-324 gives n = 300, the arousal confound and the Holm values as in R1R2_RESULTS.md:37-47.
  - **Injection:** :497-501 matches plan §4 (a fixed vector at all 16 layers, no forward pass, LR on raw keys) and is withdrawn.
  - **Scripts:** both are in Project-Oracle `d0c8395` on `origin/main`. Their CR-stripped sha256 values (`8a41036d…`, `c50782ac…`) equal the untracked originals. The paper cites them at :210, and the README at :41.
  - **"Untouched" sentence:** App. A :648/:651 is correct.
- **Abstract numbers match R1R2_RESULTS.md:** 35.8 / 38.2 / [-6.4, +1.6] / 0.985 vs 0.967 / 49.4 / 44.8.
- **12.3x arithmetic:** 0.409 = 368/900 and 0.410 = 369/900. 40.9/3.33 = 12.3.
- **The kappa sentence (PR #18):** it no longer ranks unlike quantities. The kappa is described as inter-judge agreement on responses, and the AUROC as a classifier against the design label. The phrase "encoding-phase cache features and generation-phase logit statistics" matches decision-state :66-69. The CI [0.8192, 0.9886] matches :69, :360 and :368. 0.9377 matches :71.
- **"W_K compass" claim:** the code at `emotional_dynamics.py:129-146` and `dose_response_accumulation.py:139-173` dot-products cached keys with directions. `k_proj` is used only for its shape.
- **"high fidelity":** the only remaining occurrences are the correction note's own mention (emotion-accumulation :145 / :144).
- **No UMV-L1 result anywhere:** I grepped the diff, the commit bodies and both PR bodies for label-validity, UMV-L1 and blind-reader patterns. Apart from finding 4, the hits are only "no label-validity numbers included" statements. One number-pattern hit is a cell in `tab:bridge` (layer-39 row), whose neighbours (0.859) are April bridge data, so it is not a leak.
- **Line endings:** 0 CR bytes in both user-model `.tex` files.
