# Data copied for this paper (2026-11-16)

**Source repository: `Liberation-Labs-THCoalition/Project-Oracle`** (private), not KV-Experiments. The folder
keeps the `kv_experiments` name that the 2026-11-16 copy uses across papers; the file below comes from
Project-Oracle.

| Published file | Source path @ commit | Source bytes / sha256 | Published bytes / sha256 |
|---|---|---|---|
| `oracle_generalization_findings.md` | `experiments/oracle_generalization_findings.md` @ `e68ee60709666cf5ad4d9043b60e22d7143e68b0` (2026-07-02) | 4,731 / `c5f761318e5a3dc15f21eef054135a6e528d6c861c1f754701ff9bde2757bf37` | 4,748 / `6e67a8fbe6120b4726301c067fdf8802d740edc62035ff588d343623c306a41f` |

The bytes were read from the commit (`git cat-file blob`), not from a working tree. The blob
(`bf6e7609d1a90c989c17205b36b4d6a55d6ebfc0`) is unchanged at Project-Oracle `origin/main` 5480499. Stored with LF
line endings (0 CR).

This is the write-up of the generalization test that the paper's limitations list cites: three non-laboratory
scenarios, 15 pairs each, with paired d per layer. The script that produced it is in
`../../code/kv_experiments/oracle_generalization_test.py`. Not dual-use: it holds per-layer summary statistics
and prose, with no vectors or prompt text.

## What was scrubbed

Two substitutions, on lines 3 and 5. Nothing else changed (the diff against the source is these two lines).
`<host>` and `<home>` below stand for a private machine name and a private home directory, which are not
reproduced here.

| Line | Before | After |
|---|---|---|
| 3 | `(run on <host>, oracle env)` | `(run on [lab GPU machine], oracle env)` |
| 5 | `` `<home>/oracle-experiments/results/oracle_generalization/` on <host>) `` | `` `[private home]/oracle-experiments/results/oracle_generalization/` on [lab GPU machine]) `` |

The hygiene scan covered home paths, private host names, IPv4 addresses, e-mail addresses, key and token
patterns, and the names of lab members who are not public authors. Before and after the scrub it was run with a
planted positive control (one hit for every pattern) and a clean negative control (no hits). The source had 4
hits (one home path and three host-name matches) on lines 3 and 5. The scrubbed file has none.

## Still available on request

- **The results JSON and raw activations that line 5 names** (`oracle_generalization_results.json`,
  `activations_*.npz`): these are not in git at `e68ee60` or at `origin/main` (checked with `git ls-tree`). They
  stay on internal hardware because they were never committed, not for dual-use reasons.
- **The Stage 1 to 5 raw activation captures**: held on internal hardware, as the paper's Code Availability appendix
  says.
