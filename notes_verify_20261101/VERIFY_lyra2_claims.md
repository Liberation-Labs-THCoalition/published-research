# Verifier 1: claims and numbers, Lyra II v1.1 (PR #19, head 2514f73)

Read-only check, 2026-11-01. Scope: Lyra II (both editions, paper.json), cache-tracing, meta-pattern, README,
RESEARCH_STATUS, against `corrections/user_model_v2/R1R2_RESULTS.md` and the task rules.

**Verdict: no blocking errors.** Every corrected number matches its source. No sentence still asserts the W_K or
user-state claim. The tallies hold, and the spectral and SVD results are unchanged. There are four minor findings and
two notes on sources.

## Findings, ranked

1. **(minor, internal inconsistency) The body-count footnote quotes raw spectral emotion as 1.0x; the same table says 1.7x.**
   `lyra-technique-ii/main.tex:977` (footnote): "The readout's values were listed under Denoised beside raw spectral
   values ($1.0\times$, $0.70$)". The *Emotion (spectral)* row of the same table (`:951`) and Table emotion_methods
   (`Raw spectral (52-D) & $1.7\times$`) both give 1.7x. The 1.0x cell existed before this PR (`d7c327e`), but the new
   footnote now restates it as a raw spectral value. The PR also removed those two raw-spectral cells from the readout
   rows. That counts as a change to spectral cells, but the footnote documents it. Suggest "1.0x as then listed" or
   deleting the parenthetical.

2. **(minor) The Discussion still says the bridge "shows" |rho| = 0.862 at layer 35, beside the corrected 0.872 at L43.**
   `main.tex:1211-1216`: "emotion directions projected through the key weights ($W_K$) show PC1-valence correlation
   $|\rho| = 0.862$ at layer~35 ... (The source, now under correction, gives $|\rho|$ up to 0.872 at layer~43 ...)". The
   "We had reported" frame and the parenthetical make it non-assertive, but the main clause is still present tense
   and gives the superseded figure. The Correction box, §2.2 and the Correction Note all give only 0.872 at L43.
   Suggest "showed" or dropping 0.862.

3. **(minor, intentionally untouched, flagged for Lyra) The SVD method section still applies the shape-vs-direction framework to emotion.**
   `main.tex:~437-447` ("Massive for emotion and persona ... This pattern is \emph{predicted} by the shape-vs-direction
   framework") is SVD text, and the rule leaves it unchanged. The paper withdraws direction-level emotion everywhere
   else, so this paragraph now reads slightly ahead of the corrected position. It is not an error under the rules.

4. **(trivial) `paper.json` still carries the tag `"wk-projection"`** (line 17). The tag is metadata, not a claim. It
   is listed only for consistency with the "no W_K channel" claim.

5. **(source note, supported) 0.872 at L43 and "Mantel on an unreleased earlier run" are not in R1R2_RESULTS.md.**
   Neither appears in the lines the rules cite. Both trace to the source correction on branch
   `lyra/user-model-correction-v2-final`: `user-model-paper/paper/academic/main.tex:85-88` and `:405` ("layer~43
   ($|\rho| = 0.872$ ...)", "unreleased earlier run") and `user-model-paper/README.md:37,89`. The underlying value is
   `qwen_emotion_bridge_summary.json` H4 (-0.872), per Kavi flag 43. They are supported. MEMORY's key-number row
   (0.862 at L35) is the older figure and should be updated separately.

6. **(out of scope, consistent) RESEARCH_STATUS's User Model row still reads "Under correction (PR #17)".**
   That row belongs to the user-model PR. Its numbers agree with the Lyra II row and the paper.

## Numbers checked (all match)

| Claim in paper | Source |
|---|---|
| 30-way 0.410 vs 0.370, p = 0.036, Holm 0.071 n.s. | R1R2 S1 |
| 0.348 with FWL fitted on the training fold, "below text" | R1R2 exploratory, `R1.mean_trainFWL` |
| Described method 0.358 vs 0.382, McNemar 0.20, interval -6.4 to +1.6, excludes a gain of 3 points or more | R1R2 R2 primary |
| Binary 0.992 vs 0.963, Holm 0.006; described method 0.985 vs 0.967, Holm 0.027 | R1R2 S2, S3 |
| n = 300, ten emotions, confounded with arousal | R1R2 "narrower than its name" |
| Probe: raw class-mean keys, dot product, FWL fitted on test fold, no W_K | R1R2 exploratory paragraph |
| Multiples: 12.3x (0.410), 11.1x (0.370), 10.4x (0.348), 10.7x (0.358); margin 0.029; "0.02-0.03" | arithmetic |
| "first reported as 0.409" | the earlier 40.9% |

## Residual-claim grep (with a planted positive control)

Pattern: W_K projection reads/detects/captures/classifies, model/user (internal) state, near-perfect, "compass
reads", "readable through", 0.882, +0.110, "architectural, not", "Confirmed --- direction", dispositional state,
"confirmed instrument".

- **Positive control:** a planted sentence appended to a copy of main.tex ("Our W_K projection reads the model's
  internal emotional state and the user's state.") **was caught** (line 1639 of the copy).
- **Every hit in the real file falls into one of three groups:**
  - wording the Correction box or Correction Note quotes;
  - a section marked as kept for the record (Injection §, withdrawn Tier 1.5, §2.2 with its note);
  - a negation ("claims no reading of the model's or the user's state").
- **Not one unmarked assertion.**
- **Cache-tracing:**
  - the old sentence survives only inside a footnote, quoted as previous wording;
  - the footnote's numbers match R1R2.
- **Meta-pattern:**
  - no changes in 1f0ef8d..HEAD;
  - its own correction at :248 ("no W_K, labels are not the user's state") is consistent.

## Tallies

- **Meta-pattern:**
  - the *Discrete emotion* row still reads **Suspected** (`main.tex:209`, `academic/main.tex:208`);
  - the body count is 6 confirmed, 7 suspected, 10 falsified, 1 withdrawn (`:44`, both editions).
- **RESEARCH_STATUS** gives 6/7/10/1 at `:55` and `:224`. No count moved.
- **Lyra II** states no tally. Its three readout rows read Not above text, Suspected and Withdrawn, and none is
  Confirmed.

## Untouched: spectral and SVD

- **Lines removed d7c327e..HEAD that touch SVD, denoising or persona:**
  - the title and keyword lines;
  - prose about the convergence and shape/direction framing for emotion;
  - a reformat of the emotion table ("$\times$ chance" became "$\times$").
- **Values intact:**
  - SVD-denoised 12.9x / 0.960;
  - persona 0.611-0.751 to 0.899-0.992;
  - envelope 0/16;
  - *Emotion (spectral)* 1.7x to 12.9x.
- **Exception:** the raw-spectral cells removed from the readout rows (finding 1).
- **Edition parity:** main and academic differ by the same 18 diff lines before (d7c327e) and after (HEAD), so the
  two editions changed together.

## PDFs

Both v1.1 PDFs have 25 pages. Each contains "Version 1.1", "0.348", "Not Above Text on 30 Classes", "0.872 at layer
43", the withdrawn Tier 1.5 heading and the new title. The negative control passed: a planted absent string was
reported ABSENT.
