# R1 bounded investigation: user-model v2 correction run

**Scope (prereg §5.1):** the fingerprint, the dry run and port check on Starship, and one environment diff. No GPU, one
day at most. The model was not loaded and the encode was not rerun. Every computation below ran on Starship's CPU in
`/tmp/r1_inv/`, which has since been removed.
**Written:** 2026-10-19 15:53 PDT, by a Lyra subagent, after the runs below.
**The verdict is unchanged: R1 is NOT_REPRODUCED_CODE_PATH, as registered.** This file explains it as far as the scope
allows. It does not revise it.

## In short
- **The port is faithful.** v2's own code, executed from its AST on today's pooled keys, gives **369/900**. Its
  per-trial predictions match the port's 900 of 900, and all seven of v2's outputs are identical to the port's.
- **The pooling is bit-exact.** On the same input, the encode's pooling and v2:146-148 agree on 3,600 of 3,600 vectors.
- **sklearn is not the cause.** sklearn 1.8.0 (May's version, inferred from release dates) gives the same 369, the same
  predictions and the same solver iterations as 1.9.0.
- **Float precision is not the cause.** Switching v2's projection from float32 to float64 moves 30-way logits by at
  most 1.4e-5. The smallest decision margin is 5.1e-4.
- **v2's inputs changed.** The last-token CCS runs no classifier and no optimizer, and arithmetic changes would have to
  be 571× larger to move it. It still differs from May by 3 pair orderings, so the last-token keys v2 read on
  2026-05-21 are not bit-identical to today's encode. Every library in today's environment was released after May 21,
  and the environment v2 ran under no longer exists.
- **Not established:**
  - which trial moved, because May saved no per-trial predictions;
  - whether the one-trial gap comes from the keys or from scipy's L-BFGS-B. Testing scipy would be a second
    environment diff, which is out of scope.

## 1. Port check: does v2's own code give 368 or 369 on today's keys? **369.**

**Method** (`port_check.py`, scratch):
- It parses `~/user_model_rerun/code/wk_backtrack_probe_v2.py` with `ast` and executes v2's own nodes unmodified:
  - the module level: imports minus `transformers`, the constants, and v2:48-89;
  - from `main()`: v2:94-101 (the seed, and the trials loaded from v2's own `BRIDGE_DIR`), and every top-level
    statement in v2:176-332 (30-way, valence and CCS for mean and last, then the length baseline).
- It does not execute v2:93 (mkdir), v2:103-174 (the model and the extraction), or v2:335-359 (the summary and the
  write of v2's results file).
- **Inputs:**
  - **A:** `keys_pool` at model layers 3/7/11/15, `mean_all` and `last`, appended per layer as v2:151-152 does. These
    are the keys the port used. `n_tokens` comes from the shards.
  - **B:** v2's own extraction lines v2:139-163 (AST), replayed on each shard's `keys_full` (an fp16 copy of the bf16
    cache), so the pooling is v2's `k[0].mean(dim=1)`.
  - **C:** A cast to float64.

| v2 output | May (`b09f980e…`) | port R1 today | v2 code on A | on B | on C (float64) |
|---|---|---|---|---|---|
| mean, 30-way | **368** | 369 | **369** | 369 | 369 |
| mean, valence AUROC | 0.991778 | 0.991867 | 0.991867 | 0.991867 | 0.991867 |
| mean, CCS | 0.921556 | 0.921556 | 0.921556 | 0.921556 | 0.921556 |
| last, 30-way | 128 | 128 | 128 | 128 | 128 |
| last, valence AUROC | 0.922044 | 0.921733 | 0.921733 | 0.921733 | 0.921733 |
| last, CCS | 0.775556 | 0.774889 | 0.774889 | 0.774889 | 0.774889 |
| length baseline | 0.475111 | 0.475111 | 0.475111 | 0.475111 | 0.475111 |

- **Per trial:** A's 900 predictions equal `R1.mean.pred` in `um_v2_results.json` for all 900 trials.
- **The pooling** (`pool_order.py`): on the same `keys_full` input, v2:146-148 (`k[0].mean(dim=1).flatten()`) and the
  encode's `pool4` (`permute(1,0,2).reshape(T,-1).mean(0)`) agree bitwise on 3,600 of 3,600 vectors (max |Δ| = 0.0).
- **The positive control is not perfect, and the reason is known.** Recomputing the encode's own pooling from
  `keys_full` reproduces the stored `keys_pool` bitwise for 3,582 of 3,600 mean vectors and 3,587 of 3,600 last
  vectors. fp16 holds the smallest key elements only as subnormals, and the manifest records such elements in all 900
  trials. So B ran on keys within 6.0e-8 of A, differing in 18 of 3.7M elements. It gives the same seven numbers.

**Answer:** 369. The port is faithful through v2:176-332, and the encode's pooling equals v2's. By the registered
reading, the one-trial difference therefore comes from the keys or the libraries.

## 2. Which trial differs? **Cannot be answered: May saved no per-trial predictions.**

- May's results file (sha256 `b09f980e…`) holds seven summary numbers. `probe.log` holds only the printed summary.
  v2:242-249 never stores predictions.
- What today's run can show is labelled as **candidates, not identifications.**
  - The margin is the top-two logit gap from each fold's `decision_function`. It was captured by an instrumented
    subclass, and the result is asserted unchanged by the instrumentation.
- **Today's margins:**
  - minimum 5.1e-4;
  - none below 1e-4;
  - 3 below 1e-3, 17 below 1e-2 and 206 below 0.1;
  - median 0.26.
- **If the difference is a single flip,** the decision moved by at least 5.1e-4 logits. That is about 37 times what
  float32 against float64 arithmetic produces (§4), so it is not a rounding coincidence.

**Single flips that would turn 369 into 368** (correct today; smallest margins):

| trial | emotion | topic | runner-up | margin (logits) |
|---|---|---|---|---|
| 799 | restless | 6, "a traveler lost in a city where they don't speak the language" | terrified | 5.1e-4 |
| 820 | serene | 3, "a parent watching their child take first steps" | calm | 2.5e-3 |
| 74 | appreciative | 4, "a musician about to perform on stage for the first time" | content | 3.1e-3 |
| 302 | desperate | 0, "a scientist working alone in a laboratory late at night" | defeated | 3.3e-3 |

**Flips that would add one** (wrong today; the true class is the runner-up):
- 777 (resigned, topic 9), margin 8.4e-4;
- 176 (calm, topic 8), margin 8.6e-4.

A net change of −1 could also come from several flips.

## 3. One environment diff: scikit-learn 1.8.0 against 1.9.0. **No change.**

### What v2 ran under in May
- **Not today's observer-env:**
  - `~/observer-env` was created on 2026-07-30 (`pyvenv.cfg`).
  - It has no torchao. May's `probe.log` shows torchao's `KernelPreference` and `ScaleCalculationMode` enums being
    registered at import.
- **No surviving environment matches May's:**
  - The only torchao on Starship that defines `KernelPreference` is ACE-Step's 0.16.0. It was installed on
    2026-08-16, and that environment has no sklearn.
  - No launch command survives. Searched: the zsh histories and sessions, `~/user_model_rerun`,
    `~/models/research_results`, `~/lab`, the local Claude transcripts, and episodic memory. The May 21 session's
    chunks hold prose only.
- **Every library in observer-env was released after May 21** (PyPI first-upload dates):

  | library | today's version | its release date |
  |---|---|---|
  | sklearn | 1.9.0 | 2026-06-02 (rc1 on 05-20) |
  | scipy | 1.18.0 | 2026-06-19 |
  | numpy | 2.5.1 | 2026-07-04 |
  | torch | 2.13.0 | 2026-07-08 |
  | transformers | 5.14.1 | 2026-07-16 |

  So on May 21: sklearn ≤ 1.8.0, scipy ≤ 1.17.1, numpy ≤ 2.4.6, torch ≤ 2.12.0, transformers ≤ 5.9.0.
- **May's sklearn was 1.8.0.** This is inferred, not recorded:
  - 1.8.0 was the latest stable release on May 21 (released 2025-12-10, with no 1.8.x patches);
  - it is the version in all three May-era environments with sklearn: `miniforge/envs/apolaki` (05-11), the
    miniforge base (05-15) and `ai-toolkit/.venv` (05-17).
- `probe.log`'s torch line numbers (`_pytree.py:630`, `redirects.py:35`) match torch 2.12 and 2.13 alike, so they do
  not tell the versions apart.

### Why sklearn
- **It holds the only optimizer in the 30-way count.** lbfgs stops at `tol=1e-4` after 37-39 iterations per fold, so a
  float-level change in the loss can move the solution far beyond rounding.
- **1.9.0 changed the code lbfgs consumes.** `_linear_loss.py` now averages the loss via `xp.sum`/`sw_sum` instead of
  `np.average`, the multinomial loss goes through an array-API path, and `_logistic.py` adds an L-BFGS callback bridge.
- **numpy was the alternative, but test C already showed** that v2's projection gives the same results in float32 and
  float64.

### The run
- **The venv:** a throwaway `/tmp/r1_inv/venv` on Homebrew Python 3.12.13 (observer-env's base) with scikit-learn
  1.8.0. Everything else was pinned to observer-env's versions: numpy 2.5.1, scipy 1.18.0, joblib 1.5.3 and
  threadpoolctl 3.6.0.
- **The runner:** `venv_r1.py` runs the same AST selection on A. The unused `torch` alias was dropped from v2:23's
  import node, because the venv has no torch and no probe line uses it.
- **The control:** the same runner under observer-env reproduces the port check exactly: 369, the seven numbers, the
  per-trial predictions and the iterations.

**Result:**
- **369**;
- all seven numbers identical to 1.9.0;
- **0 of 900 trials change prediction**;
- the same lbfgs iterations per fold (39, 38, 39, 39, 37; `max_iter` 2000);
- valence probability gaps identical to 17 digits.

**sklearn is ruled out.**

## 4. What the seven numbers show (CPU, no new environment)

**A precision bound** (`numerics_bound.py`): v2's code was instrumented and run on float32 keys and on float64 keys.

| path | largest shift, float32 vs float64 | smallest gap it would have to cross | ratio |
|---|---|---|---|
| mean, 30-way logits | 1.36e-5 | 5.1e-4 (margin) | 37 |
| last CCS scores | ≤ 2.3e-6 | ≥ 1.28e-3 (pos/neg pair gap) | ≥ 571 |
| mean CCS scores | ≤ 2.1e-6 | ≥ 5.8e-3 | ≥ 2,700 |
| last valence LR | 9.4e-8 | 2.7e-5 | 291 |

**The last-token CCS differs from May** by 0.775556 − 0.774889. That is 3 pair orderings out of 4,500 (5 folds of
30 × 30). Nothing on its path can move it except the keys:
- **What the path contains:**
  - the last-token keys, which are a raw bf16 copy of the cache with no pooling;
  - v2:55-89 and v2:298-311, all numpy;
  - `roc_auc_score`.
  It has no classifier, no scaler and no scipy optimizer.
- **The labels, folds and token counts equal May's.** The length baseline is identical to 16 digits.
- **Arithmetic precision is excluded** by at least 571×.
- **AUROC is exact here:** there are no tied pairs.
- **What remains is the input:** the last-token keys v2 read on 2026-05-21 are not bit-identical to today's encode.
  That is established for the last-token keys. Whether the cause is the forward kernels (torch ≤ 2.12 and
  transformers ≤ 5.9 in May) or the rendering is not separable here.

**The mean-key outputs are mixed:**
- **Equal to May:** mean CCS, and v1's 470.
- **Different from May:** the mean 30-way count and mean valence (+2 of 22,500 pairs).
- **Two readings fit:**
  - mean keys that moved less than the last-token keys did, since pooling averages over every position;
  - a solver difference.
  These are not separated.

**The fingerprint is consistent with a stack change but cannot attribute it.** It was IN_BAND, done before this
investigation. Today's key norms differ from May's bridge run by up to 1.3e-4 relative (L3, trial 0: 379.6678 against
379.6635). May's bridge run used a different cache, though, so the drift cannot be attributed.

**Side observation: the band's repeats were float64 reruns, not perturbations.** With a measured RMS of 0,
`blocks + noise × 0` promotes float32 to float64, so all 30 "perturbed" repeats are float64 reruns
(`r1_band`/`_r1_k`, `_v1_k`). That is where v1's h = 1 comes from: 470 in float32, 471 in all 30 float64 runs. R1
gives 369 either way. Neither verdict changes: R1 is out of band at h = 0 or 1, and v1 is REPRODUCED at h = 0 or 1.

## What is established, and what is not

**Established:**
1. The port reproduces v2's own code exactly on today's keys: 369, with 900 of 900 predictions and all seven outputs
   identical.
2. The encode's pooling is bit-exact to v2:146-148.
3. sklearn 1.8.0 against 1.9.0 changes nothing.
4. Arithmetic precision cannot flip any 30-way decision (37× short) or any CCS pair (571× short).
5. v2's last-token keys from May differ from today's encode.
6. Every library in today's environment postdates May 21, and the environment v2 ran under no longer exists in its May
   state.

**Not established:**
1. Which trial or trials moved, because there are no May predictions.
2. Whether 368 against 369 comes from the keys (the May forward stack) or from scipy's L-BFGS-B (≤ 1.17.1 in May).
   - The Python wrapper diff from scipy 1.17.1 to 1.18.0 shows only the removed `iprint`/`disp` and ILP64 LAPACK
     detection.
   - The compiled solver was not compared.
3. May's exact numpy, scipy, torch and transformers versions.

**For reading the verdict:**
- **What the registered branch assumed:** NOT_REPRODUCED_CODE_PATH applies when R1 is out of band and v1 is
  REPRODUCED, on the reasoning that "today's keys match v2's evening, so look at the port".
- **What the investigation found:** the port is faithful, and that reasoning holds only at the resolution v1 can see. v1
  reproduces its count, but the last-token keys differ.
- **Why the band missed it:** the band (h = 0) measured today's run-to-run determinism. It was not built to see the
  May-to-today change of stack.
- **So:** the one-trial miss sits in a pipeline whose inputs demonstrably changed. The verdict stands as registered.
  Its label's implied cause, the code path, is not supported.

## What would settle it (outside this scope, not done)
- **scipy 1.17.1 as a second environment diff:** CPU, minutes.
- **May's keys:** these need May's forward stack (torch ≤ 2.12, transformers ≤ 5.9, torchao) and the GPU.

## Commands and files
- **Scratch scripts and outputs** (not in the Research tree): the session scratchpad
  `C:\Users\Thomas\AppData\Local\Temp\claude\C--Users-Thomas\fedc44e2-d3b2-49fe-9e89-5890ff7e37ba\scratchpad\r1_inv\`.
  - Scripts: `port_check.py`, `pool_order.py`, `venv_r1.py`, `numerics_bound.py`.
  - Outputs: `port_check.json`/`.log`, `r1_observer_env.json`, `r1_sklearn180.json`, `numerics_bound.json`.
  - This location is temporary.
- **On Starship:**
  ```
  ~/observer-env/bin/python -I port_check.py ~/user_model_v2/um_v2_results.json /tmp/r1_inv/port_check.json
  ~/observer-env/bin/python -I pool_order.py
  ~/observer-env/bin/python -I venv_r1.py /tmp/r1_inv/port_check.npz /tmp/r1_inv/r1_observer_env.json   # control
  /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv /tmp/r1_inv/venv
  /tmp/r1_inv/venv/bin/python -m pip install scikit-learn==1.8.0 numpy==2.5.1 scipy==1.18.0 joblib==1.5.3 threadpoolctl==3.6.0
  /tmp/r1_inv/venv/bin/python -I venv_r1.py /tmp/r1_inv/port_check.npz /tmp/r1_inv/r1_sklearn180.json
  ~/observer-env/bin/python -I numerics_bound.py /tmp/r1_inv/port_check.npz /tmp/r1_inv/numerics_bound.json
  rm -rf /tmp/r1_inv
  ```
- **Nothing in `~/user_model_v2` was written.** `find ~/user_model_v2 -newermt "2026-10-19 23:30:00"` returns 0
  entries, and `um_v2_results.json` still hashes `0aefed36…`.
- **Time used:** under an hour of CPU and wall clock. No GPU.
